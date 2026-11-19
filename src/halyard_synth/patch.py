# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The patch: a complete, readable description of one hit (plan §4, §5).

Every field is required and the renderer holds no defaults, so a future
default can never change an old patch. A Patch validates itself whenever it is
constructed, by the parser or by an edit, so every Patch object lies inside the
valid domain; the parser and the editing commands reject everything else and
never clamp or repair. Problems name their fields by path, counting from 1:
`noise[2].amp[3].ms` is the time of the third knot of the second layer.

The editing commands (`tune`, `stretch`, `bursts`, `softer`, `flam`) rewrite points and
never add a parameter. They compute on the decimal values as written, so the
patch they print stays readable.
"""

from __future__ import annotations

import codecs
import dataclasses
import hashlib
import itertools
import math
import re
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .curve import amplitude_curve, pitch_curve

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterable, Iterator, Sequence

type Knots = tuple[tuple[float, float], ...]  # (ms, value) pairs, or (ratio, level_db) for modes

INSTRUMENT = "snare"
SCHEMA = 1
SAMPLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)
FORMATS = ("float32", "pcm24", "pcm16")
MODES = ("gain", "peak", "true-peak")  # the output's gain, or a target for the sample peak or the true peak
MAX_BODY_MS = 2000.0
MAX_NOISE_MS = 1000.0  # the noise repeats every second
MAX_LAYERS = 4
MAX_MODES = 4
MAX_RATIO = 8.0
MAX_KNOTS = 32
MIN_SPACING_MS = Decimal("0.01")  # 10 µs, compared on the decimal values as written
MAX_SEED = 2**63 - 1
MAX_BURSTS = 10
MAX_RIMS = 2
MAX_TAKE_DB, MAX_TAKE_MS = 6.0, 5.0  # how far takes may differ in strength and in timing
TAKES_KEY = b"snare/takes/v1"  # the stream every take's variation is drawn from
MAX_BRUSHES = 1  # a brush's band takes up to seven noise tables
SWEEP = (0.5, 2.0)  # a sweep's ratios, an octave either way
SWEEP_TRAVEL = 2.0  # and an octave between its lowest and highest
LOWEST_BAND_HZ = 20.0
MAX_RIM_MODES = 8
MAX_COMPRESSION = MAX_RATIO**3  # how much faster any mode may decay than its first: the body's 8 at damping 3
RIM_HZ = (100.0, 20000.0)  # where a rim's modes may lie
PAIR = 2  # a knot, a mode and a band are pairs
LOWEST_HZ, HIGHEST_HZ = 10.0, 20000.0  # where a body frequency may go


class PatchError(ValueError):
    """An invalid patch or edit. `problems` lists every reason, each with its field path."""

    def __init__(self, problems: Iterable[str]) -> None:
        """Keep every problem, and say them all, one per line."""
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


# Validation. Each check reports every problem it finds, with its path.


class _Problems(list[str]):
    """Every problem found, each with its field's path."""

    def add(self, path: str, message: str) -> None:
        """Record a problem."""
        self.append(f"{path}: {message}")

    def number(
        self,
        path: str,
        v: object,
        lo: float | None = None,
        hi: float | None = None,
        *,
        silence: bool = False,
    ) -> float | None:
        """Return v as a finite number within [lo, hi], or -inf where silence is allowed; None if invalid."""
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.add(path, f"expected a number, found {_kind(v)}")
            return None
        try:
            x = float(v)
        except OverflowError:
            self.add(path, "is too large")
            return None
        if silence and x == -math.inf:
            return x
        if not math.isfinite(x):
            self.add(path, "must be finite" + (" (or -inf for silence)" if silence else ""))
            return None
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            self.add(path, f"{x:g} is outside {lo:g} … {hi:g}" + (" (or -inf for silence)" if silence else ""))
            return None
        return x

    def integer(self, path: str, v: object, allowed: tuple[int, ...]) -> None:
        """Check that v is one of the allowed integers."""
        if isinstance(v, bool) or not isinstance(v, int):
            self.add(path, f"expected an integer, found {_kind(v)}")
        elif v not in allowed:
            self.add(path, f"{v} is not one of {', '.join(map(str, allowed))}")

    def choice(self, path: str, v: object, allowed: tuple[str, ...]) -> str | None:
        """Return v if it is one of the allowed strings; None if not."""
        if not isinstance(v, str) or v not in allowed:
            self.add(path, f"expected one of {', '.join(repr(a) for a in allowed)}, found {v!r}")
            return None
        return v

    def knots(self, path: str, knots: object, key: str, count: int, max_ms: float) -> list[tuple[float, float]] | None:
        """Return the knot pairs if their times are valid; the values are only checked to be numbers."""
        if not isinstance(knots, tuple) or not all(isinstance(k, tuple) and len(k) == PAIR for k in knots):
            self.add(path, f"expected (ms, {key}) knots")
            return None
        ok = count <= len(knots) <= MAX_KNOTS
        if not ok:
            self.add(path, f"needs {count} to {MAX_KNOTS} knots, found {len(knots)}")
        found = [self.number(f"{path}[{i}].ms", ms, 0.0, max_ms) for i, (ms, _) in enumerate(knots, 1)]
        given = [self.number(f"{path}[{i}].{key}", v) for i, (_, v) in enumerate(knots, 1)]
        times = [t for t in found if t is not None]
        values = [v for v in given if v is not None]
        if not ok or len(times) < len(found) or len(values) < len(given):
            return None
        for i in range(1, len(times)):
            a, b = Decimal(repr(times[i - 1])), Decimal(repr(times[i]))
            where = f"{path}[{i + 1}].ms"
            if b == a:
                self.add(where, "repeats the previous knot time")
            elif b < a:
                self.add(where, "is earlier than the previous knot")
            elif b - a < MIN_SPACING_MS:
                self.add(where, f"is {b - a} ms after the previous knot; knots need 0.01 ms (10 µs)")
        return list(zip(times, values, strict=True))

    def shape(self, path: str, knots: object, max_ms: float) -> None:
        """Check an amplitude shape: gains in [0, 1], 0 at both ends, highest point exactly 1."""
        pairs = self.knots(path, knots, "gain", 3, max_ms)
        if pairs is None:
            return
        gains = [g for _, g in pairs]
        bad = False
        for i, g in enumerate(gains, 1):
            if not 0.0 <= g <= 1.0:
                self.add(f"{path}[{i}].gain", f"{g:g} is outside 0 … 1")
                bad = True
        if gains[0] != 0.0:
            self.add(f"{path}[1].gain", "an amplitude shape must start at 0")
        if gains[-1] != 0.0:
            self.add(f"{path}[{len(gains)}].gain", "an amplitude shape must end at 0")
        if not bad and max(gains) != 1.0:
            self.add(path, f"the highest gain must be exactly 1, found {max(gains)!r}")


_KINDS: dict[type, str] = {bool: "a boolean", str: "a string", list: "an array", tuple: "an array", dict: "a table"}


def _kind(v: object) -> str:
    """Name what kind of value v is, as TOML would."""
    return _KINDS.get(type(v), type(v).__name__)


def _validate_pitch(
    pitch: list[tuple[float, float]],
    tune: float | None,
    ratios: list[float] | None,
    problems: _Problems,
) -> None:
    """Check the pitch curve: from 0 ms to ratio 1, positive, and every body frequency audible at every knot."""
    if pitch[0][0] != 0.0:
        problems.add("body.pitch[1].ms", "the pitch curve must start at 0 ms")
    if pitch[-1][1] != 1.0:
        problems.add(f"body.pitch[{len(pitch)}].ratio", "the pitch curve must end at ratio 1")
    for k, (_, r) in enumerate(pitch, 1):
        if r <= 0:
            problems.add(f"body.pitch[{k}].ratio", "must be positive")
        elif tune is not None and ratios is not None:
            # Every body frequency is extreme only at pitch knots, so the knots suffice.
            for i, ratio in enumerate(ratios, 1):
                f = tune * ratio * r
                if not LOWEST_HZ <= f <= HIGHEST_HZ:
                    where = "" if len(ratios) == 1 else f" for mode {i}"
                    problems.add(
                        f"body.pitch[{k}].ratio",
                        f"body frequency {f:g} Hz{where} is outside 10 Hz … 20 kHz",
                    )


def _validate_mode_list(
    modes: object,
    where: str,
    spec: tuple[str, float, float, int],
    plural: str,
    problems: _Problems,
) -> list[float] | None:
    """Check modes as (value, level_db) pairs: their count, values in range and increasing, the loudest exactly 0 dB.

    `spec` is the value's name, its range and the most modes allowed; return
    the values if every one is a number in range, else None.
    """
    key, lo, hi, most = spec
    if not isinstance(modes, tuple) or not all(isinstance(m, tuple) and len(m) == PAIR for m in modes):
        problems.add(where, f"expected ({key}, level_db) modes")
        return None
    if not 1 <= len(modes) <= most:
        problems.add(where, f"needs 1 to {most} modes, found {len(modes)}")
        return None
    values = [problems.number(f"{where}[{i}].{key}", v, lo, hi) for i, (v, _) in enumerate(modes, 1)]
    levels = [problems.number(f"{where}[{i}].level_db", v, -60.0, 0.0) for i, (_, v) in enumerate(modes, 1)]
    for i, (a, b) in enumerate(itertools.pairwise(values), 2):
        if a is not None and b is not None and not b > a:
            problems.add(f"{where}[{i}].{key}", f"{plural} must strictly increase")
    loudness = [v for v in levels if v is not None]
    if len(loudness) == len(levels) and max(loudness) != 0.0:
        problems.add(where, f"the loudest mode must be exactly 0 dB, found {max(loudness)!r}")
    valid = [v for v in values if v is not None]
    return valid if len(valid) == len(values) else None


def _validate_sweep(sweep: list[tuple[float, float]], band: object, path: str, problems: _Problems) -> None:
    """Check a sweep: ratios within 0.5 … 2, an octave apart at most, the band audible at every knot."""
    outside = [(k, r) for k, (_, r) in enumerate(sweep, 1) if not SWEEP[0] <= r <= SWEEP[1]]
    for k, r in outside:
        problems.add(f"{path}.sweep[{k}].ratio", f"{r:g} is outside {SWEEP[0]:g} … {SWEEP[1]:g}")
    if outside:
        return
    ratios = [r for _, r in sweep]
    low, high = min(ratios), max(ratios)
    if high / low > SWEEP_TRAVEL:
        problems.add(f"{path}.sweep", f"travels {high / low:g} times, more than an octave")
    # The band is extreme only at sweep knots, since each segment is monotone, so the knots suffice.
    if not isinstance(band, tuple) or len(band) != PAIR:
        return  # the layer's check has reported it
    lo, hi = band
    if all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in (lo, hi)):  # whole numbers too
        if lo * low < LOWEST_BAND_HZ:
            problems.add(f"{path}.sweep", f"takes the low edge to {lo * low:g} Hz, below 20 Hz")
        if hi * high > HIGHEST_HZ:
            problems.add(f"{path}.sweep", f"takes the high edge to {hi * high:g} Hz, above 20 kHz")


# Every field's check. Each reports every problem it finds, with its field's path, and returns what it checked,
# which the fields after it in the section may need: the pitch curve needs the tuning and the modes it moves.

type Check = Callable[[_Problems, str, Any, Any, dict[str, Any]], Any]  # (problems, path, value, section, checked)


def _range(lo: float, hi: float, *, silence: bool = False) -> Check:
    """Check a number within lo … hi, or -inf where silence is allowed."""
    return lambda problems, path, v, _section, _checked: problems.number(path, v, lo, hi, silence=silence)


def _one_of_numbers(allowed: tuple[int, ...]) -> Check:
    """Check an integer that is one of those allowed."""
    return lambda problems, path, v, _section, _checked: problems.integer(path, v, allowed)


def _one_of(allowed: tuple[str, ...]) -> Check:
    """Check a string that is one of those allowed."""
    return lambda problems, path, v, _section, _checked: problems.choice(path, v, allowed)


def _shape(max_ms: float) -> Check:
    """Check an amplitude shape ending by max_ms."""
    return lambda problems, path, v, _section, _checked: problems.shape(path, v, max_ms)


def _band(problems: _Problems, path: str, band: object, _section: object, _checked: dict[str, Any]) -> object:
    """Check a band: [low, high] in Hz, both audible, low below high."""
    if not isinstance(band, tuple) or len(band) != PAIR:
        problems.add(path, "expected [low, high] in Hz")
        return None
    lo = problems.number(f"{path}[1]", band[0], 20.0, 20000.0)
    hi = problems.number(f"{path}[2]", band[1], 20.0, 20000.0)
    if lo is not None and hi is not None and not lo < hi:
        problems.add(path, f"the low edge {lo:g} Hz must be below the high edge {hi:g} Hz")
    return band


def _body_modes(problems: _Problems, path: str, modes: object, _section: object, _checked: dict[str, Any]) -> object:
    """Check the body's modes and return their ratios, the first exactly 1, the rest up to 8."""
    ratios = _validate_mode_list(modes, path, ("ratio", 1.0, MAX_RATIO, MAX_MODES), "mode ratios", problems)
    if ratios is not None and ratios[0] != 1.0:
        problems.add(f"{path}[1].ratio", "the first mode defines the tuning: its ratio must be exactly 1")
    return ratios


def _body_pitch(problems: _Problems, path: str, pitch: object, _section: object, checked: dict[str, Any]) -> object:
    """Check the pitch curve, with every body frequency of the tuning and the modes audible at every knot."""
    knots = problems.knots(path, pitch, "ratio", 2, MAX_BODY_MS)
    if knots is not None:
        _validate_pitch(knots, checked["tune_hz"], checked["modes"], problems)
    return knots


def _rim_modes(problems: _Problems, path: str, modes: object, _section: object, checked: dict[str, Any]) -> object:
    """Check a rim's modes in Hz, whose highest the damping may not make decay too much faster than its first."""
    hz = _validate_mode_list(modes, path, ("hz", *RIM_HZ, MAX_RIM_MODES), "rim modes", problems)
    damping = checked["damping"]
    if hz is not None and damping is not None and (s := (hz[-1] / hz[0]) ** damping) > MAX_COMPRESSION:
        problems.add(
            f"{path.removesuffix('.modes')}.damping",
            f"its highest mode would decay {s:.4g} times faster than its first, more than "
            f"{MAX_COMPRESSION:g}; lower the damping or bring the modes closer",
        )
    return hz


def _sweep(problems: _Problems, path: str, sweep: object, section: Brush, _checked: dict[str, Any]) -> object:
    """Check a brush's sweep, which moves its band."""
    knots = problems.knots(path, sweep, "ratio", 2, MAX_NOISE_MS)
    if knots is not None:
        _validate_sweep(knots, section.band_hz, path.removesuffix(".sweep"), problems)
    return knots


def _output_db(problems: _Problems, path: str, db: object, _section: object, checked: dict[str, Any]) -> object:
    """Check the output's dB: a gain up to +24 dB, or a peak target up to 0 dBFS."""
    return problems.number(path, db, -60.0, 24.0 if checked["mode"] == "gain" else 0.0)


@dataclass(frozen=True)
class Spec:
    """How a section's field is read, checked and written: its TOML kind, a pair's keys, its comment and its check.

    A number is a float, written exactly; an integer and a choice are written as
    they are; a band is [low, high]; pairs are an array of two-key inline
    tables. The check sees the fields checked before it, for the rules between
    fields.
    """

    kind: str  # "number", "integer", "choice", "band" or "pairs"
    keys: tuple[str, str]  # a pairs field's two keys, in order
    note: str  # the canonical text's comment
    check: Check


def _spec(kind: str, check: Check, keys: tuple[str, str] = ("", ""), note: str = "") -> dict[str, Spec]:
    """Describe a field, as a dataclass field's metadata."""
    return {"spec": Spec(kind, keys, note, check)}


LEVEL = _spec("number", _range(-90.0, 24.0, silence=True))
RATTLE = _spec("number", _range(0.0, 1.0), note="0 … 1, gated at the body's fundamental")


@dataclass(frozen=True)
class Body:
    """The modal body: its tuning, level and damping, its modes, its pitch glide and its shape."""

    tune_hz: float = field(metadata=_spec("number", _range(40.0, 4000.0)))
    level_db: float = field(metadata=LEVEL)
    damping: float = field(metadata=_spec("number", _range(0.0, 3.0), note="mode i decays ratio^damping times faster"))
    modes: Knots = field(metadata=_spec("pairs", _body_modes, ("ratio", "level_db"), "loudest mode exactly 0 dB"))
    pitch: Knots = field(
        metadata=_spec("pairs", _body_pitch, ("ms", "ratio"), "ratio to tune_hz; starts at 0 ms, ends at 1"),
    )
    amp: Knots = field(
        metadata=_spec("pairs", _shape(MAX_BODY_MS), ("ms", "gain"), "shape: 0 at both ends, highest point exactly 1"),
    )


@dataclass(frozen=True)
class Layer:
    """A noise layer: its level, its band, its rattle depth and its shape."""

    level_db: float = field(metadata=LEVEL)
    band_hz: tuple[float, float] = field(metadata=_spec("band", _band))  # (low, high)
    rattle: float = field(metadata=RATTLE)
    amp: Knots = field(metadata=_spec("pairs", _shape(MAX_NOISE_MS), ("ms", "gain")))


@dataclass(frozen=True)
class Brush:
    """A brush on the head: a noise layer whose band moves during the hit, by its sweep."""

    level_db: float = field(metadata=LEVEL)
    band_hz: tuple[float, float] = field(metadata=_spec("band", _band, note="at sweep ratio 1"))
    rattle: float = field(metadata=RATTLE)
    sweep: Knots = field(metadata=_spec("pairs", _sweep, ("ms", "ratio"), "the band's edges times the ratio"))
    amp: Knots = field(metadata=_spec("pairs", _shape(MAX_NOISE_MS), ("ms", "gain")))


@dataclass(frozen=True)
class Rim:
    """The rim struck with or instead of the head: the hoop's ring and the stick's knock, as modes in Hz."""

    level_db: float = field(metadata=LEVEL)
    damping: float = field(
        metadata=_spec("number", _range(0.0, 3.0), note="mode i decays (hz_i / hz_1)^damping times faster"),
    )
    modes: Knots = field(metadata=_spec("pairs", _rim_modes, ("hz", "level_db"), "loudest mode exactly 0 dB"))
    amp: Knots = field(metadata=_spec("pairs", _shape(MAX_BODY_MS), ("ms", "gain")))


@dataclass(frozen=True)
class Drive:
    """The saturator's small-signal gain."""

    db: float = field(metadata=_spec("number", _range(0.0, 48.0), note="small-signal gain; 0 = bypass"))


@dataclass(frozen=True)
class Output:
    """How the file is written: its rate, format, level mode and dB, and polarity."""

    sample_rate: int = field(metadata=_spec("integer", _one_of_numbers(SAMPLE_RATES)))
    format: str = field(metadata=_spec("choice", _one_of(FORMATS), note='"float32" | "pcm24" | "pcm16"'))
    mode: str = field(metadata=_spec("choice", _one_of(MODES), note='"gain" | "peak" | "true-peak"'))
    db: float = field(metadata=_spec("number", _output_db, note="gain in dB, or peak target in dBFS"))
    polarity: int = field(metadata=_spec("integer", _one_of_numbers((1, -1))))


@dataclass(frozen=True)
class Takes:
    """How a patch's takes differ as played: in strength, and in the timing of every rise after the first."""

    strength_db: float = field(
        metadata=_spec("number", _range(0.0, MAX_TAKE_DB), note="each take this much harder or softer, at most"),
    )
    timing_ms: float = field(
        metadata=_spec("number", _range(0.0, MAX_TAKE_MS), note="each later rise this much earlier or later, at most"),
    )


@dataclass(frozen=True)
class Patch:
    """One hit, complete: its seed, body, layers, drive and output, and how its takes differ, if they do."""

    seed: int
    body: Body
    noise: tuple[Layer, ...]
    rim: tuple[Rim, ...]
    brush: tuple[Brush, ...]
    drive: Drive
    output: Output
    takes: Takes | None = None  # None: takes differ only in their noise

    def __post_init__(self) -> None:
        """Validate the patch, refusing it with every problem."""
        problems = _Problems()
        _validate(self, problems)
        if problems:
            raise PatchError(problems)

    @property
    def length_ms(self) -> float:
        """T_src: the latest end among the shapes of the audible sources (plan §3).

        A silent body still gates rattle, but only inside the layers it gates, so
        it never lengthens the sound. With nothing audible the length is 0.
        """
        shapes = [(s.level_db, s.amp) for s in self.noise] + [(r.level_db, r.amp) for r in self.rim]
        shapes += [(b.level_db, b.amp) for b in self.brush]
        ends = [amp[-1][0] for level_db, amp in shapes if level_db > -math.inf]
        if self.body.level_db > -math.inf:
            ends.append(self.body.amp[-1][0])
        return max(ends, default=0.0)


type Section = Body | Layer | Rim | Brush | Drive | Output | Takes


def _check(section: Section, path: str, problems: _Problems) -> dict[str, Any]:
    """Check a section's fields in order, each seeing those before it, and return what each checked."""
    checked: dict[str, Any] = {}
    for f in dataclasses.fields(section):
        spec: Spec = f.metadata["spec"]
        checked[f.name] = spec.check(problems, f"{path}.{f.name}", getattr(section, f.name), section, checked)
    return checked


def _validate_body(b: object, problems: _Problems) -> None:
    """Check the body."""
    if not isinstance(b, Body):
        problems.add("body", "expected a body section")
    else:
        _check(b, "body", problems)


def _validate_layers(layers: tuple[Layer, ...], problems: _Problems) -> None:
    """Check the layers: how many, and each."""
    if not 1 <= len(layers) <= MAX_LAYERS:
        problems.add("noise", f"needs 1 to {MAX_LAYERS} layers, found {len(layers)}")
    for j, layer in enumerate(layers, 1):
        _check(layer, f"noise[{j}]", problems)


def _validate_brushes(brushes: tuple[Brush, ...], problems: _Problems) -> None:
    """Check the brushes: how many, and each."""
    if len(brushes) > MAX_BRUSHES:
        problems.add("brush", f"at most {MAX_BRUSHES} brush, found {len(brushes)}")
    for j, b in enumerate(brushes, 1):
        _check(b, f"brush[{j}]", problems)


def _validate_rims(rims: tuple[Rim, ...], problems: _Problems) -> None:
    """Check the rims: how many, and each."""
    if len(rims) > MAX_RIMS:
        problems.add("rim", f"at most {MAX_RIMS} rims, found {len(rims)}")
    for j, r in enumerate(rims, 1):
        _check(r, f"rim[{j}]", problems)


def _validate_drive(d: object, problems: _Problems) -> None:
    """Check the drive."""
    if not isinstance(d, Drive):
        problems.add("drive", "expected a drive section")
    else:
        _check(d, "drive", problems)


def _validate_output(o: object, problems: _Problems) -> None:
    """Check the output section."""
    if not isinstance(o, Output):
        problems.add("output", "expected an output section")
    else:
        _check(o, "output", problems)


def _validate(p: Patch, problems: _Problems) -> None:
    """Check every part of a patch."""
    _validate_seed(p.seed, problems)
    _validate_body(p.body, problems)
    noise: object = p.noise
    if not isinstance(noise, tuple) or not all(isinstance(layer, Layer) for layer in noise):
        problems.add("noise", "expected noise layers")
    else:
        _validate_layers(noise, problems)
    rim: object = p.rim
    if not isinstance(rim, tuple) or not all(isinstance(r, Rim) for r in rim):
        problems.add("rim", "expected rims")
    else:
        _validate_rims(rim, problems)
    brush: object = p.brush
    if not isinstance(brush, tuple) or not all(isinstance(b, Brush) for b in brush):
        problems.add("brush", "expected brushes")
    else:
        _validate_brushes(brush, problems)
    _validate_drive(p.drive, problems)
    _validate_output(p.output, problems)
    if p.takes is not None:
        _validate_takes(p, problems)


def _validate_takes(p: Patch, problems: _Problems) -> None:
    """Check how takes differ: within range, and the hardest and softest take inside the valid domain."""
    t: object = p.takes
    if not isinstance(t, Takes):
        problems.add("takes", "expected a takes section")
        return
    strength = _check(t, "takes", problems)["strength_db"]
    if not strength or problems:  # the extremes are checked only on a patch with nothing else wrong
        return
    plain = dataclasses.replace(p, takes=None)
    for db, how in ((-strength, "harder"), (strength, "softer")):
        try:
            softer(plain, db)
        except PatchError as e:
            problems.add("takes.strength_db", f"a take struck {strength:g} dB {how} leaves the domain: {e.problems[0]}")


def _validate_seed(seed: object, problems: _Problems) -> None:
    """Check the seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.add("seed", f"expected an integer, found {_kind(seed)}")
    elif not 0 <= seed <= MAX_SEED:
        problems.add("seed", f"{seed} is outside 0 … 2**63 - 1")


# Parsing.

_TOP = ("instrument", "schema", "seed", "body", "noise", "drive", "output")
_OPTIONAL = ("rim", "brush", "takes")  # what may be absent: no rims, no brush, takes that differ only in noise


def _keys(problems: _Problems, prefix: str, table: dict[str, Any], expected: Sequence[str]) -> bool:
    """Report unknown and missing keys of a table; True when none is missing."""
    for key in table:
        if key not in expected:
            problems.add(f"{prefix}{key}", "unknown field")
    missing = [key for key in expected if key not in table]
    for key in missing:
        problems.add(f"{prefix}{key}", "missing")
    return not missing


def _float[T](v: T) -> T | float:
    """Return a TOML integer as a float, which floats accept; anything else is left for validation to name."""
    if isinstance(v, int) and not isinstance(v, bool):
        try:
            return float(v)
        except OverflowError:
            return v
    return v


def _pairs(problems: _Problems, path: str, items: object, keys: tuple[str, str]) -> Knots | None:
    """Return an array of inline tables with exactly these keys, as pairs; None after reporting what is wrong."""
    if not isinstance(items, list):
        problems.add(path, f"expected an array of {{ {', '.join(keys)} }} tables, found {_kind(items)}")
        return None
    out = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.add(f"{path}[{i}]", f"expected a {{ {', '.join(keys)} }} table, found {_kind(item)}")
            return None
        if not _keys(problems, f"{path}[{i}].", item, keys):
            return None
        out.append((_float(item[keys[0]]), _float(item[keys[1]])))
    return tuple(out)


def _section(problems: _Problems, path: str, table: object, fields: Sequence[str]) -> dict[str, Any] | None:
    """Return a table with exactly these fields; None after reporting what is wrong."""
    if not isinstance(table, dict):
        problems.add(path, f"expected a table, found {_kind(table)}")
        return None
    return table if _keys(problems, f"{path}.", table, fields) else None


def _read[S: (Body, Layer, Rim, Brush, Drive, Output, Takes)](
    problems: _Problems,
    path: str,
    table: object,
    section: type[S],
) -> S | None:
    """Build a section from its TOML table, by its fields; None after reporting what is wrong.

    Numbers become floats, a band a pair and pairs a tuple of pairs; whatever
    a value holds, the section's validation names what is wrong with it.
    """
    fields = dataclasses.fields(section)
    t = _section(problems, path, table, [f.name for f in fields])
    if t is None:
        return None
    values: dict[str, Any] = {}
    for f in fields:
        spec: Spec = f.metadata["spec"]
        v = t[f.name]
        if spec.kind == "number":
            values[f.name] = _float(v)
        elif spec.kind == "band":
            values[f.name] = tuple(map(_float, v)) if isinstance(v, list) else v
        elif spec.kind == "pairs":
            values[f.name] = _pairs(problems, f"{path}.{f.name}", v, spec.keys)
        else:
            values[f.name] = v
    if any(values[f.name] is None for f in fields if f.metadata["spec"].kind == "pairs"):
        return None
    return section(**values)


def _layer(problems: _Problems, path: str, table: dict[str, Any]) -> Layer | None:
    """Build a noise layer."""
    return _read(problems, path, table, Layer)


def _rim(problems: _Problems, path: str, table: dict[str, Any]) -> Rim | None:
    """Build a rim."""
    return _read(problems, path, table, Rim)


def _brush(problems: _Problems, path: str, table: dict[str, Any]) -> Brush | None:
    """Build a brush."""
    return _read(problems, path, table, Brush)


def _header(data: dict[str, Any]) -> None:
    """Refuse another instrument or schema, alone: such a patch may share nothing else."""
    instrument = data.get("instrument")
    if instrument is None:
        raise PatchError([f'instrument: missing; a snare patch starts with instrument = "{INSTRUMENT}"'])
    if instrument != INSTRUMENT:
        raise PatchError([f'instrument: this is a {instrument!r} patch, not a "{INSTRUMENT}" one'])
    schema = data.get("schema")
    if schema is None:
        raise PatchError(["schema: missing"])
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise PatchError([f"schema: expected an integer, found {_kind(schema)}"])
    if schema != SCHEMA:
        raise PatchError([f"schema: unsupported version {schema}; this renderer reads schema {SCHEMA}"])


def _tables_of[T](
    problems: _Problems,
    data: dict[str, Any],
    name: str,
    build: Callable[[_Problems, str, dict[str, Any]], T | None],
) -> tuple[T, ...] | None:
    """Build an array of tables, [[name]], each by `build`; absent means none, and None after reporting a problem.

    A required array that is absent has been reported as missing already.
    """
    tables = data.get(name, [])
    if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
        problems.add(name, f"expected an array of tables ([[{name}]]), found {_kind(tables)}")
        return None
    built = [build(problems, f"{name}[{j}]", t) for j, t in enumerate(tables, 1)]
    found = tuple(b for b in built if b is not None)
    return found if len(found) == len(built) else None


def _drive(problems: _Problems, data: dict[str, Any]) -> Drive | None:
    """Build and check the drive; None after reporting what is wrong."""
    if "drive" not in data or (drive := _read(problems, "drive", data["drive"], Drive)) is None:
        return None
    _validate_drive(drive, problems)
    return drive


def _output(problems: _Problems, data: dict[str, Any]) -> Output | None:
    """Build and check the output section; None after reporting what is wrong."""
    if "output" not in data or (output := _read(problems, "output", data["output"], Output)) is None:
        return None
    _validate_output(output, problems)
    return output


def from_dict(data: object) -> Patch:
    """Build a patch from parsed TOML, reporting every problem at once."""
    if not isinstance(data, dict):
        raise PatchError(["expected a table"])
    _header(data)
    problems = _Problems()
    _keys(problems, "", {k: v for k, v in data.items() if k not in _OPTIONAL}, _TOP)
    if "seed" in data:
        _validate_seed(data["seed"], problems)
    body = _read(problems, "body", data["body"], Body) if "body" in data else None
    if body is not None:
        _validate_body(body, problems)
    noise = _tables_of(problems, data, "noise", _layer) if "noise" in data else None
    rim, brush = _tables_of(problems, data, "rim", _rim), _tables_of(problems, data, "brush", _brush)
    if noise is not None:
        _validate_layers(noise, problems)
    if rim is not None:
        _validate_rims(rim, problems)
    if brush is not None:
        _validate_brushes(brush, problems)
    drive, output = _drive(problems, data), _output(problems, data)
    takes = _read(problems, "takes", data["takes"], Takes) if "takes" in data else None
    if takes is not None:
        _check(takes, "takes", problems)
    # Anything missing or unbuilt has been reported, so with no problem every part is there and checked.
    if problems or body is None or noise is None or rim is None or brush is None or drive is None or output is None:
        raise PatchError(problems)
    return Patch(data["seed"], body, noise, rim, brush, drive, output, takes)


def loads(text: str) -> Patch:
    """Parse a patch from TOML text."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PatchError([f"TOML: {e}"]) from None
    except RecursionError:  # arrays or tables nested deeper than any patch, made to exhaust the parser
        raise PatchError(["TOML: nested too deeply to be a patch"]) from None
    return from_dict(data)


def read_text(path: str | os.PathLike[str]) -> str:
    """Read a patch file's text: UTF-8 with or without a BOM, or UTF-16 with one, as Windows PowerShell's `>` writes."""
    raw = Path(path).read_bytes()
    return raw.decode("utf-16" if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig")


def load(path: str | os.PathLike[str]) -> Patch:
    """Read a patch from a TOML file, in UTF-8 or UTF-16."""
    try:
        text = read_text(path)
    except UnicodeDecodeError:
        raise PatchError(["not UTF-8 or UTF-16 text"]) from None
    return loads(text)


def _table(section: Body | Layer | Rim | Brush | Drive | Output | Takes) -> dict[str, Any]:
    """Return a section as the table its TOML parses to."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(section):
        spec: Spec = f.metadata["spec"]
        v = getattr(section, f.name)
        if spec.kind == "band":
            out[f.name] = list(v)
        elif spec.kind == "pairs":
            out[f.name] = [{spec.keys[0]: a, spec.keys[1]: b} for a, b in v]
        else:
            out[f.name] = v
    return out


def to_dict(p: Patch) -> dict[str, Any]:
    """Return the patch as the table its TOML parses to."""
    return {
        "instrument": INSTRUMENT,
        "schema": SCHEMA,
        "seed": p.seed,
        "body": _table(p.body),
        "noise": [_table(n) for n in p.noise],
        **({"rim": [_table(r) for r in p.rim]} if p.rim else {}),
        **({"brush": [_table(b) for b in p.brush]} if p.brush else {}),
        "drive": _table(p.drive),
        "output": _table(p.output),
        **({"takes": _table(p.takes)} if p.takes else {}),
    }


# Canonical TOML.

_COMMENT_COLUMN = 28


def dumps(p: Patch) -> str:
    """Write the complete patch as TOML with the canonical comments; loads(dumps(p)) == p exactly."""
    lines = [f'instrument = "{INSTRUMENT}"', f"schema = {SCHEMA}", f"seed = {p.seed}", "", "[body]", *_text(p.body)]
    for j, n in enumerate(p.noise, 1):
        lines += ["", _note("[[noise]]", f"noise[{j}]"), *_text(n)]
    for j, r in enumerate(p.rim, 1):
        lines += ["", _note("[[rim]]", f"rim[{j}]: the hoop's ring and the stick's knock"), *_text(r)]
    for j, b in enumerate(p.brush, 1):
        lines += ["", _note("[[brush]]", f"brush[{j}]: a noise layer whose band moves"), *_text(b)]
    lines += ["", "[drive]", *_text(p.drive), "", "[output]", *_text(p.output)]
    if p.takes is not None:
        lines += ["", _note("[takes]", "how takes differ as played"), *_text(p.takes)]
    return "\n".join(lines) + "\n"


def _text(section: Body | Layer | Rim | Brush | Drive | Output | Takes) -> list[str]:
    """Write a section's fields: each run of single values as aligned rows, each array of pairs on its own."""
    lines: list[str] = []
    rows: list[tuple[str, str, str]] = []
    for f in dataclasses.fields(section):
        spec: Spec = f.metadata["spec"]
        v = getattr(section, f.name)
        if spec.kind == "pairs":
            lines += _rows(rows) if rows else []
            rows = []
            lines += _array(f.name, [((spec.keys[0], a), (spec.keys[1], b)) for a, b in v], spec.note)
        else:
            rows.append((f.name, _value(spec.kind, v), spec.note))
    return lines + (_rows(rows) if rows else [])


def _value(kind: str, v: object) -> str:
    """Write a single value as TOML, by its field's kind: a number exactly, a choice quoted, a band as [low, high]."""
    if kind == "number":
        return _f(cast("float", v))
    if kind == "choice":
        return f'"{v}"'
    if kind == "band":
        low, high = cast("tuple[float, float]", v)
        return f"[{_f(low)}, {_f(high)}]"
    return str(v)


def _f(x: float) -> str:
    """Write a float exactly, as Python reads it back."""
    return repr(float(x))


def _note(line: str, note: str) -> str:
    """Add a comment to a line, at the comment column."""
    return f"{(line + ' ').ljust(_COMMENT_COLUMN)}# {note}" if note else line


def _rows(rows: Sequence[tuple[str, str, str]]) -> list[str]:
    """Write key = value rows, aligned, with their comments."""
    width = max(len(k) for k, _, _ in rows)
    return [_note(f"{k.ljust(width)} = {v}", note) for k, v, note in rows]


type Item = tuple[tuple[str, float], tuple[str, float]]  # an inline table of two keys


def _array(name: str, items: Sequence[Item], note: str) -> list[str]:
    """Write an array of inline tables, with the first column aligned."""
    firsts = [f"{items[0][0][0]} = {_f(a[1])}," for a, _ in items] if items else []
    width = max(map(len, firsts), default=0)
    body = [
        f"  {{ {first.ljust(width)} {key} = {_f(v)} }}," for first, (_, (key, v)) in zip(firsts, items, strict=True)
    ]
    return [_note(f"{name} = [", note), *body, "]"]


# Tuning vocabulary shared by `halyard init`, `halyard tune`, the report and the WAV root note.

_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_pitch(hz: float) -> float:
    """Return the fractional MIDI note number (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(midi: int) -> str:
    """Name a MIDI note, such as F#3."""
    return f"{_NOTES[midi % 12]}{midi // 12 - 1}"


def describe_tuning(hz: float) -> str:
    """Describe a frequency as its nearest note and the cents from it, such as 'F#3 -3.2 cents'."""
    p = midi_pitch(hz)
    n = round(p)
    cents = round(100 * (p - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{note_name(n)} {cents:+.1f} cents"


def parse_tune(text: str) -> float:
    """Parse a note name such as D3, C#4 or Eb3 (Hz rounded to 3 decimals), or a frequency in Hz."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text.strip())
    if m:
        midi = _LETTERS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) + 1)
        return round(float(440.0 * 2.0 ** ((midi - 69) / 12)), 3)
    msg = f"not a note name or frequency: {text!r}"
    try:
        hz = float(text)
    except ValueError:
        raise ValueError(msg) from None
    if not math.isfinite(hz):
        raise ValueError(msg)
    return hz


# Editing commands. Each returns a new, validated patch, or raises PatchError.

_US = Decimal("0.001")  # knot times are rewritten to 1 µs


def _dec(x: float) -> Decimal:
    """Return the decimal value as written."""
    return Decimal(repr(float(x)))


def _ms(x: Decimal) -> float:
    """Round a time to 1 µs."""
    return float(Decimal(x).quantize(_US))


def _significant(x: float, digits: int = 4) -> float:
    """Round to `digits` significant digits."""
    return float(f"{x:.{digits}g}")


def tune(p: Patch, text: str) -> Patch:
    """Set tune_hz from a note name or a frequency. Modes, glide and rattle follow, being ratios."""
    return dataclasses.replace(p, body=dataclasses.replace(p.body, tune_hz=parse_tune(text)))


def curve_names(p: Patch) -> tuple[str, ...]:
    """Name every curve of the patch by its path."""
    noise = [f"noise[{j}].amp" for j in range(1, len(p.noise) + 1)]
    rims = [f"rim[{j}].amp" for j in range(1, len(p.rim) + 1)]
    brushes = [f"brush[{j}].{c}" for j in range(1, len(p.brush) + 1) for c in ("sweep", "amp")]
    return ("body.pitch", "body.amp", *noise, *rims, *brushes)


def curve_knots(p: Patch, name: str) -> Knots:
    """Return a curve's knots by its path, one of `curve_names(p)`."""
    if name == "body.pitch":
        return p.body.pitch
    if name == "body.amp":
        return p.body.amp
    if name.startswith("rim["):
        return p.rim[int(name[len("rim[") : name.index("]")]) - 1].amp
    if name.startswith("brush["):
        b = p.brush[int(name[len("brush[") : name.index("]")]) - 1]
        return b.sweep if name.endswith(".sweep") else b.amp
    return p.noise[_layer_index(p, name[: -len(".amp")])].amp


def _set_curves(p: Patch, curves: dict[str, Knots]) -> Patch:
    """Return the patch with these curves replaced, by path."""
    body = p.body
    if "body.pitch" in curves:
        body = dataclasses.replace(body, pitch=curves["body.pitch"])
    if "body.amp" in curves:
        body = dataclasses.replace(body, amp=curves["body.amp"])
    noise = tuple(
        dataclasses.replace(layer, amp=curves[f"noise[{j}].amp"]) if f"noise[{j}].amp" in curves else layer
        for j, layer in enumerate(p.noise, 1)
    )
    rim = tuple(
        dataclasses.replace(r, amp=curves[f"rim[{j}].amp"]) if f"rim[{j}].amp" in curves else r
        for j, r in enumerate(p.rim, 1)
    )
    brush = tuple(
        dataclasses.replace(
            b,
            sweep=curves.get(f"brush[{j}].sweep", b.sweep),
            amp=curves.get(f"brush[{j}].amp", b.amp),
        )
        for j, b in enumerate(p.brush, 1)
    )
    return dataclasses.replace(p, body=body, noise=noise, rim=rim, brush=brush)


def _layer_index(p: Patch, name: str) -> int:
    """Return the 0-based index of a layer named as the patch paths name it: 'noise[2]' is the second layer."""
    m = re.fullmatch(r"noise\[(\d+)\]", name.strip())
    if not m or not 1 <= int(m.group(1)) <= len(p.noise):
        layers = ", ".join(f"noise[{j}]" for j in range(1, len(p.noise) + 1))
        raise PatchError([f"{name}: not a layer of this patch; it has {layers}"])
    return int(m.group(1)) - 1


def stretch(p: Patch, curves: str | Iterable[str], factor: float, after_ms: float | None = None) -> Patch:
    """Rescale knot times after each curve's peak (0 ms for pitch and sweeps), or after `after_ms`, by `factor`.

    `curves` names curves by path (`body.amp`, `body.pitch`, `noise[2].amp`,
    `brush[1].sweep`); `all` names every amplitude shape. Times are rewritten
    to 1 µs.
    """
    if not (math.isfinite(factor) and factor > 0):
        raise PatchError([f"stretch: the factor must be a positive number, found {factor!r}"])
    if after_ms is not None and not (math.isfinite(after_ms) and after_ms >= 0):
        raise PatchError([f"--after: must be a time in ms from 0 on, found {after_ms!r}"])
    names: list[str] = []
    for name in [curves] if isinstance(curves, str) else curves:
        if name == "all":
            names += [n for n in curve_names(p) if n.endswith(".amp")]
        elif name in curve_names(p):
            names.append(name)
        else:
            raise PatchError([f"{name}: not a curve of this patch; expected all or one of {', '.join(curve_names(p))}"])
    out: dict[str, Knots] = {}
    for name in dict.fromkeys(names):
        knots = curve_knots(p, name)
        if after_ms is not None:
            anchor = after_ms
        else:
            anchor = next(ms for ms, g in knots if g == 1.0) if name.endswith(".amp") else 0.0
        a, f = _dec(anchor), Decimal(repr(float(factor)))
        out[name] = tuple((ms if ms <= anchor else _ms(a + f * (_dec(ms) - a)), v) for ms, v in knots)
    return _set_curves(p, out)


SHOULDER_MS = 0.1  # the shoulder knot after each burst's peak (plan §2.2)
TAIL_DB = 60.0  # the last burst's exponential falls this far over its tail


@dataclass(frozen=True)
class Train:
    """A clap train: how many bursts, their spacing and its growth, and each burst's rise, dip, taper and tail."""

    count: int
    spacing: float
    start: float = 0.0
    growth: float = 1.0
    rise: float = 0.1
    dip: float = 0.1
    taper: float = 1.0
    tail: float = 80.0


def bursts(p: Patch, layer: str, train: Train) -> Patch:
    """Rewrite a layer's shape as a clap train of `count` bursts (plan §4).

    Burst i starts at b_i (b_1 = start, and each gap is `growth` times the
    previous, the first being `spacing`), rises to its peak in `rise`, and
    decays along an exponential: to `dip` times the lower of its own and the
    next burst's peak at the next start, or, for the last burst, by 60 dB over
    `tail`, ending at 0. A shoulder knot 0.1 ms after each peak keeps the crack
    sharp, and the last burst's tail has one knot 20 dB down. Peak i is
    taper^(i−1), scaled so the loudest is exactly 1.

    The result is 3N + 2 knots, or 3N + 3 with a start: the first knot, then
    each burst's peak, shoulder and dip (or tail knot), then the end. Times are
    written to 1 µs and gains to 4 significant digits.
    """
    j = _layer_index(p, layer)
    _check_train(train)
    knots = _train_knots(train)
    if len(knots) > MAX_KNOTS:
        msg = f"{layer}.amp: {train.count} bursts need {len(knots)} knots, more than the {MAX_KNOTS} allowed"
        raise PatchError([msg])
    return _set_curves(p, {f"noise[{j + 1}].amp": knots})


def _check_train(train: Train) -> None:
    """Refuse a train's settings out of range, naming each."""
    problems = _Problems()
    count: object = train.count
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_BURSTS:
        problems.add("--count", f"needs 1 to 10 bursts, found {count!r}")
    for name, value, ok in (
        ("--spacing", train.spacing, train.spacing > 0),
        ("--start", train.start, train.start >= 0),
        ("--growth", train.growth, train.growth > 0),
        ("--rise", train.rise, train.rise >= float(MIN_SPACING_MS)),
        ("--dip", train.dip, 0 <= train.dip < 1),
        ("--taper", train.taper, train.taper > 0),
        ("--tail", train.tail, train.tail > 3 * (SHOULDER_MS + float(MIN_SPACING_MS))),
    ):
        if not (math.isfinite(value) and ok):
            problems.add(name, f"{value!r} is out of range")
    if problems:
        raise PatchError(problems)


def _train_knots(train: Train) -> Knots:
    """Return the knots of a valid train, times to 1 µs."""
    count, rise, dip, tail, taper = train.count, train.rise, train.dip, train.tail, train.taper
    peaks = [taper ** (i - (0 if taper <= 1 else count - 1)) for i in range(count)]
    peaks = [1.0 if pk == max(peaks) else _significant(pk) for pk in peaks]
    starts, gap = [train.start], train.spacing
    for _ in range(count - 1):
        starts.append(starts[-1] + gap)
        gap *= train.growth
    knots = [(0.0, 0.0), *([(train.start, 0.0)] if train.start > 0 else [])]
    for i, (b, pk) in enumerate(zip(starts, peaks, strict=True)):
        top = b + rise
        if i + 1 < count:
            nxt = starts[i + 1]
            floor = dip * min(pk, peaks[i + 1])
            if not nxt - top >= SHOULDER_MS + 2 * float(MIN_SPACING_MS):
                msg = (
                    f"--rise: burst {i + 1} rises for {rise:g} ms, which leaves no room for its "
                    f"shoulder and dip before burst {i + 2} starts {nxt - b:g} ms later"
                )
                raise PatchError([msg])
            decay = math.log(pk / max(floor, pk * 10 ** (-TAIL_DB / 20))) / (nxt - top)
            ends = [(nxt, _significant(floor))]
        else:
            decay = math.log(10) * TAIL_DB / 20 / tail
            ends = [(top + tail / 3, _significant(pk * 10 ** (-TAIL_DB / 60))), (top + tail, 0.0)]
        knots += [(top, pk), (top + SHOULDER_MS, _significant(pk * math.exp(-decay * SHOULDER_MS))), *ends]
    return tuple((_ms(Decimal(repr(ms))), g) for ms, g in knots)


def rises(values: Sequence[float]) -> list[tuple[int, int]]:
    """Return the (first, last) knot indices of each rise: a maximal run of strictly increasing knots.

    Both ends of a rise have zero slope, the first because nothing before it is
    lower and the last because nothing after it is higher, so a rise can be
    lengthened without changing the rest of the curve.
    """
    out: list[tuple[int, int]] = []
    i, n = 0, len(values)
    while i < n - 1:
        if values[i + 1] > values[i]:
            k = i
            while k < n - 1 and values[k + 1] > values[k]:
                k += 1
            out.append((i, k))
            i = k
        else:
            i += 1
    return out


def _lengthen_rises(knots: Knots, factor: Decimal) -> Knots:
    """Lengthen every rise by `factor` and shift later knots by the added time, all to 1 µs.

    The shift after each rise is the exact added time so far, rounded once, so
    the rounding never accumulates over a train of bursts, and knots between
    rises move together, keeping their spacing exactly.
    """
    values = [v for _, v in knots]
    times = [_dec(ms) for ms, _ in knots]
    new, added, shift, done = list(times), Decimal(0), Decimal(0), 0
    for a, b in rises(values):
        for k in range(done, a + 1):
            new[k] = times[k] + shift
        added += (factor - 1) * (times[b] - times[a])
        shift = added.quantize(_US)
        new[b] = times[b] + shift
        for k in range(a + 1, b):  # knots inside a rise keep their place in it
            new[k] = (new[a] + (new[b] - new[a]) * (times[k] - times[a]) / (times[b] - times[a])).quantize(_US)
        done = b + 1
    for k in range(done, len(times)):
        new[k] = times[k] + shift
    return tuple((float(t), v) for t, (_, v) in zip(new, knots, strict=True))


def softer(p: Patch, db: float, glide: float = 2.0, contact: float = 0.2) -> Patch:
    """Return the same drum struck `db` softer, by the three rules of plan §2.7.

    With v = 10^(−db/20):
      * Level: every source level falls by db (silent sources stay silent).
      * Glide: the pitch curve's excess over 1 scales by v^glide.
      * Contact: every rise lengthens by v^(−contact), later knots shifting by
        the added time, and every layer's and brush's upper band edge scales by v^contact.
    Nothing else changes. An exponent of 0 turns its rule off. Values are
    written to 1 µs, ratios to 6 decimals, band edges to 0.1 Hz and levels to
    6 decimals of a dB.
    """
    for name, x in (("DB", db), ("--glide", glide), ("--contact", contact)):
        if not math.isfinite(x):
            raise PatchError([f"{name}: must be a finite number, found {x!r}"])
    if db == 0:
        return p
    v = 10 ** (-db / 20)

    def level(x: float) -> float:
        """Lower a level by db, keeping silence silent, to 6 decimals."""
        return x if x == -math.inf else float(round(_dec(x) - Decimal(repr(float(db))), 6))

    def rises(amp: Knots) -> Knots:
        return _struck_rises(amp, v, contact)

    def struck[L: (Layer, Brush)](layer: L) -> L:
        """Strike a layer or a brush softer: its level, its rises and, by contact, its upper band edge."""
        lo, hi = layer.band_hz
        band = (lo, float(round(_dec(hi) * Decimal(repr(v**contact)), 1)) if contact else hi)
        return dataclasses.replace(layer, level_db=level(layer.level_db), band_hz=band, amp=rises(layer.amp))

    b = p.body
    return dataclasses.replace(
        p,
        body=dataclasses.replace(
            b,
            level_db=level(b.level_db),
            pitch=_struck_glide(b.pitch, v, glide),
            amp=rises(b.amp),
        ),
        noise=tuple(struck(layer) for layer in p.noise),
        rim=tuple(dataclasses.replace(r, level_db=level(r.level_db), amp=rises(r.amp)) for r in p.rim),
        brush=tuple(struck(brush) for brush in p.brush),
    )


def _struck_rises(amp: Knots, v: float, contact: float) -> Knots:
    """Lengthen a shape's rises by v^(−contact), as a strike softer by v does."""
    return _lengthen_rises(amp, Decimal(repr(v**-contact))) if contact else amp


def _struck_glide(pitch: Knots, v: float, glide: float) -> Knots:
    """Scale a pitch curve's excess over 1 by v^glide, as a strike softer by v does, to 6 decimals."""
    if not glide:
        return pitch
    scale = Decimal(repr(v**glide))
    return tuple((ms, float(round(1 + (_dec(r) - 1) * scale, 6))) for ms, r in pitch)


def flam(p: Patch, gap_ms: float, grace_db: float = 12.0, glide: float = 2.0, contact: float = 0.2) -> Patch:
    """Return the drum struck twice: a grace note `grace_db` softer, then the main stroke `gap_ms` later.

    Two strokes, as `strokes` writes them: the grace is the drum struck softer by
    softer's rules, its level in its shapes so the drive still hears it softer.
    """
    for name, x, ok in (("--gap", gap_ms, gap_ms > 0), ("--grace", grace_db, grace_db > 0)):
        if not (math.isfinite(x) and ok):
            raise PatchError([f"{name}: must be a positive number, found {x!r}"])
    return strokes(p, [(0.0, grace_db), (gap_ms, 0.0)], glide, contact, name="--grace")


MIN_STROKES, MAX_STROKES = 2, 16  # a roll's strokes; the 32 knots of a shape usually end it first


@dataclass(frozen=True)
class Roll:
    """A roll or a build: how many strokes, their spacing and its growth, and how much softer the first is."""

    count: int
    spacing: float  # ms, from the first stroke to the second
    growth: float = 1.0  # each gap over the one before: below 1 the roll speeds up
    crescendo: float = 0.0  # dB softer the first stroke is than the last, the strokes between in even steps


def roll(p: Patch, r: Roll, glide: float = 2.0, contact: float = 0.2) -> Patch:
    """Return the drum struck `r.count` times, a roll or a build, as `strokes` writes it, by softer's rules."""
    problems = _Problems()
    count: object = r.count
    if isinstance(count, bool) or not isinstance(count, int) or not MIN_STROKES <= count <= MAX_STROKES:
        problems.add("--count", f"needs {MIN_STROKES} to {MAX_STROKES} strokes, found {count!r}")
    for name, value, ok in (
        ("--spacing", r.spacing, r.spacing > 0),
        ("--growth", r.growth, r.growth > 0),
        ("--crescendo", r.crescendo, r.crescendo >= 0),
    ):
        if not (math.isfinite(value) and ok):
            problems.add(name, f"{value!r} is out of range")
    if problems:
        raise PatchError(problems)
    starts, gap, at = [], _dec(r.spacing), Decimal(0)
    for _ in range(r.count):
        starts.append(at)
        at, gap = at + gap, gap * Decimal(repr(float(r.growth)))
    levels = [r.crescendo * (1 - k / (r.count - 1)) for k in range(r.count)]
    return strokes(p, [(float(_ms(t)), db) for t, db in zip(starts, levels, strict=True)], glide, contact)


def strokes(
    p: Patch,
    hits: Sequence[tuple[float, float]],
    glide: float = 2.0,
    contact: float = 0.2,
    *,
    name: str = "--crescendo",
) -> Patch:
    """Return the drum struck at each (start in ms, dB softer) of `hits`, as ordinary knots.

    The first stroke starts at 0 ms and the loudest is 0 dB softer. Each stroke
    is the drum struck that much softer by softer's rules, its rises and glide,
    with its level written into its shapes so the drive hears it softer; the
    bands and the brush's sweep are shared. Every shape is each stroke's in
    turn: one stroke's until the next stroke's first knot, where it takes the
    previous stroke's value and goes on as the next. The pitch curve restarts at
    each stroke's start and reaches its glide at the body's peak, since the
    glide follows the head's motion. The decay compresses from the shape's
    first knot at gain 1, the first of the loudest strokes, so a softer stroke
    before it rings its higher modes as long as its fundamental. Times are
    written to 1 µs, gains and ratios to 6 decimals. It fails rather than drop
    knots, and rather than start a stroke before the one before it has peaked.
    """
    levels = [10 ** (-db / 20) for _, db in hits]
    for (_, db), v in zip(hits, levels, strict=True):
        if db and round(Decimal(repr(v)), 6) >= 1:
            raise PatchError([f"{name}: {db!r} dB is too small for a stroke to stay softer, written to 6 decimals"])
    starts = [_dec(t) for t, _ in hits]

    def stroke(knots: Knots, k: int) -> list[tuple[float, float]]:
        """Return stroke k's shape, softer by its dB, from its own start."""
        if not hits[k][1]:
            return list(knots)
        scale = Decimal(repr(levels[k]))
        return [(ms, float(round(_dec(g) * scale, 6))) for ms, g in _struck_rises(knots, levels[k], contact)]

    # A stroke's strike is the earliest peak among its sounding shapes: the next stroke must not start before it,
    # or the strokes would run into each other and one would be lost.
    sounding = [c for c in curve_names(p) if c.endswith(".amp") and _sounds(p, c)]
    for k in range(1, len(hits)):
        strike = (
            min(
                _dec(next(ms for ms, g in shape if g == max(g for _, g in shape))) - _dec(shape[0][0])
                for shape in (stroke(curve_knots(p, c), k - 1) for c in sounding)
            )
            if sounding
            else Decimal(0)
        )
        if starts[k] - starts[k - 1] < strike + MIN_SPACING_MS:
            raise PatchError(
                [
                    (
                        f"stroke {k + 1}: starts {_ms(starts[k] - starts[k - 1])} ms after stroke {k}, which strikes "
                        f"{_ms(strike)} ms in; space the strokes further apart"
                    ),
                ],
            )

    def joined(path: str) -> Knots:
        main = curve_knots(p, path)
        out = stroke(main, 0)
        for k in range(1, len(hits)):
            start = _dec(main[0][0]) + starts[k]
            # The joined curve so far is what sounds just before this stroke, so its value there is the join.
            curve = amplitude_curve([ms / 1000 for ms, _ in out], [g for _, g in out])
            residual = float(round(Decimal(repr(float(curve(float(start) / 1000)))), 6))
            mine = [(_ms(_dec(ms) + starts[k]), g) for ms, g in stroke(main, k)]
            out = [*(kn for kn in out if _dec(kn[0]) <= start - MIN_SPACING_MS), (_ms(start), residual), *mine[1:]]
        return tuple(out)

    glided = [
        p.body.pitch if not db else _struck_glide(p.body.pitch, v, glide)
        for (_, db), v in zip(hits, levels, strict=True)
    ]
    pitch = list(glided[0])
    for k in range(1, len(hits)):
        body = stroke(p.body.amp, k)  # this stroke's own body, its rise lengthened when it is softer
        peak = _dec(next(ms for ms, g in body if g == max(g for _, g in body)))
        curve = pitch_curve([ms / 1000 for ms, _ in pitch], [r for _, r in pitch])
        residual = float(round(Decimal(repr(float(curve(float(starts[k]) / 1000)))), 6))
        mine = [(_ms(_dec(ms) + starts[k] + peak), r) for ms, r in glided[k]]
        pitch = [*(kn for kn in pitch if _dec(kn[0]) <= starts[k] - MIN_SPACING_MS), (_ms(starts[k]), residual), *mine]
    curves = {c: joined(c) for c in curve_names(p) if c.endswith(".amp")}
    return _set_curves(p, {**curves, "body.pitch": tuple(pitch)})


def _sounds(p: Patch, path: str) -> bool:
    """Whether the source an amplitude shape belongs to is audible."""
    if path == "body.amp":
        return p.body.level_db > -math.inf
    kind, index = path[: path.index("[")], int(path[path.index("[") + 1 : path.index("]")]) - 1
    sources: dict[str, tuple[Layer | Rim | Brush, ...]] = {"noise": p.noise, "rim": p.rim, "brush": p.brush}
    return sources[kind][index].level_db > -math.inf


def stack(a: Patch, b: Patch, db: float = 0.0) -> Patch:
    """Return one patch that sounds both: `a` and `b`, its sources `db` louder, through one drive.

    The layers, rims and brushes of both are kept, a's first, within the limits
    of a patch; the body is the one that sounds, and two sounding bodies are
    refused, since a patch is one drum's head. The drive, the output and the
    seed are a's, so b's noise is the same noise in its own bands, drawn for
    a's seed, and its layers' rattle follows the body the stack keeps. How its
    takes differ is whichever of the two says, and two that differ are
    refused. Levels are written to 6 decimals.
    """
    if not math.isfinite(db):
        raise PatchError([f"--db: must be a finite number, found {db!r}"])
    louder = Decimal(repr(float(db)))

    def level(x: float) -> float:
        """Raise one of b's levels by db, keeping silence silent."""
        return x if x == -math.inf else float(round(_dec(x) + louder, 6))

    if a.takes is not None and b.takes is not None and a.takes != b.takes:
        raise PatchError(["takes: both patches say how their takes differ, and differently; keep one [takes]"])
    sounding = [p.body for p in (a, b) if p.body.level_db > -math.inf]
    if len(sounding) > 1:
        raise PatchError(["body: both patches have a sounding body, and a patch is one drum's head; silence one"])
    body = (
        a.body
        if not sounding or sounding[0] is a.body
        else dataclasses.replace(b.body, level_db=level(b.body.level_db))
    )
    return dataclasses.replace(
        a,
        body=body,
        noise=(*a.noise, *(dataclasses.replace(n, level_db=level(n.level_db)) for n in b.noise)),
        rim=(*a.rim, *(dataclasses.replace(r, level_db=level(r.level_db)) for r in b.rim)),
        brush=(*a.brush, *(dataclasses.replace(r, level_db=level(r.level_db)) for r in b.brush)),
        takes=a.takes or b.takes,
    )


def played(p: Patch) -> Patch:
    """Return the take p's seed names, as played: the patch itself when its takes differ only in their noise.

    Every take is drawn from SHAKE-256 of its own key and the seed, so a take is
    the same in every version and on every platform. It is the drum struck
    softer by d dB, by softer's rules, d drawn evenly from ±`strength_db`. Then
    time itself moves at every later event of the hit: a rise after the first
    of any shape, such as a clap's later burst or a roll's next stroke, with
    every rise in the other shapes within 1 ms of it. Each event moves by a
    draw within ±`timing_ms`, and everything at or after it moves with it,
    the body, its glide, the layers, rims and brush and its sweep alike, so a
    stroke stays one stroke. No move brings two knots of a curve within 10 µs
    or a curve past its limit, and the hit's first rise stays on the beat. The
    result has no [takes]: it is that one take.
    """
    t = p.takes
    if t is None:
        return p
    words = hashlib.shake_256(TAKES_KEY + p.seed.to_bytes(8, "little")).digest(8 * _TAKE_DRAWS)
    draws = iter(int.from_bytes(words[i : i + 8], "little") >> 11 for i in range(0, len(words), 8))
    q = dataclasses.replace(p, takes=None)
    softness = t.strength_db * (2 * next(draws) * 2.0**-53 - 1)
    if softness:
        q = softer(q, softness)
    if not t.timing_ms:
        return q
    names = curve_names(q)
    times = _warped(q, names, _dec(t.timing_ms), draws)
    curves = {c: tuple((_ms(x), v) for x, (_, v) in zip(times[c], curve_knots(q, c), strict=True)) for c in names}
    return _set_curves(q, curves)


def _events(q: Patch, names: Sequence[str]) -> list[Decimal]:
    """Return a hit's later events: the start of every rise after the first of a shape, those within 1 ms as one."""
    starts = sorted(
        _dec(curve_knots(q, c)[first][0])
        for c in names
        if c.endswith(".amp")
        for first, _ in rises([g for _, g in curve_knots(q, c)])[1:]
    )
    events: list[Decimal] = []
    for x in starts:
        if not events or x - events[-1] > EVENT_MS:
            events.append(x)
    return events


def _warped(q: Patch, names: Sequence[str], reach: Decimal, draws: Iterator[int]) -> dict[str, list[Decimal]]:
    """Return every curve's knot times with each event and all after it moved by a draw within ±reach."""
    times = {c: [_dec(ms) for ms, _ in curve_knots(q, c)] for c in names}
    moved = Decimal(0)
    for event in _events(q, names):
        at = event + moved  # where the event is now, after the moves before it
        move = (reach * (2 * Decimal(next(draws)) / Decimal(2**53) - 1)).quantize(_US)
        for c in names:
            later = [i for i, x in enumerate(times[c]) if x >= at]
            if later and later[0]:  # never within 10 µs of the knot before
                move = max(move, MIN_SPACING_MS - (times[c][later[0]] - times[c][later[0] - 1]))
            if later:  # never past the curve's limit
                move = min(move, _limit(c) - times[c][-1])
        for c in names:
            times[c] = [x + move if x >= at else x for x in times[c]]
        moved += move
    return times


EVENT_MS = Decimal("1.0")  # rises this close belong to one event of a take's timing
_TAKE_DRAWS = 1 + 9 * MAX_KNOTS  # a strength, and a move for every event, at most one per rise of every shape


def _limit(curve: str) -> Decimal:
    """Return how late a curve's last knot may be: 2 s for the body and the rims, 1 s for the noise's."""
    return Decimal(repr(MAX_BODY_MS if curve.startswith(("body", "rim")) else MAX_NOISE_MS))
