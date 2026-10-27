# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The patch: a complete, readable description of one hit (plan §4, §5).

Every field is required and the renderer holds no defaults, so a future
default can never change an old patch. A Patch validates itself whenever it is
constructed, by the parser or by an edit, so every Patch object lies inside the
valid domain; the parser and the editing commands reject everything else and
never clamp or repair. Problems name their fields by path, counting from 1:
`noise[2].amp[3].ms` is the time of the third knot of the second layer.

The editing commands (`tune`, `stretch`, `bursts`, `softer`) rewrite points and
never add a parameter. They compute on the decimal values as written, so the
patch they print stays readable.
"""

from __future__ import annotations

import dataclasses
import math
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal

INSTRUMENT = "snare"
SCHEMA = 1
SAMPLE_RATES = (44100, 48000, 88200, 96000)
FORMATS = ("float32", "pcm24", "pcm16")
MAX_BODY_MS = 2000.0
MAX_NOISE_MS = 1000.0  # the noise repeats every second
MAX_LAYERS = 4
MAX_MODES = 4
MAX_RATIO = 8.0
MAX_KNOTS = 32
MIN_SPACING_MS = Decimal("0.01")  # 10 µs, compared on the decimal values as written
MAX_SEED = 2**63 - 1


class PatchError(ValueError):
    """An invalid patch or edit. `problems` lists every reason, each with its field path."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class Body:
    tune_hz: float
    level_db: float
    damping: float
    modes: tuple  # ((ratio, level_db), ...)
    pitch: tuple  # ((ms, ratio), ...)
    amp: tuple  # ((ms, gain), ...)


@dataclass(frozen=True)
class Layer:
    level_db: float
    band_hz: tuple  # (low, high)
    rattle: float
    amp: tuple  # ((ms, gain), ...)


@dataclass(frozen=True)
class Drive:
    db: float


@dataclass(frozen=True)
class Output:
    sample_rate: int
    format: str
    mode: str
    db: float
    polarity: int


@dataclass(frozen=True)
class Patch:
    seed: int
    body: Body
    noise: tuple  # (Layer, ...)
    drive: Drive
    output: Output

    def __post_init__(self):
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
        ends = [layer.amp[-1][0] for layer in self.noise if layer.level_db > -math.inf]
        if self.body.level_db > -math.inf:
            ends.append(self.body.amp[-1][0])
        return max(ends, default=0.0)


# Validation. Each check reports every problem it finds, with its path.


class _Problems(list):
    def add(self, path: str, message: str) -> None:
        self.append(f"{path}: {message}")

    def number(self, path, v, lo=None, hi=None, *, silence=False):
        """A finite number within [lo, hi], or -inf where silence is allowed; None if invalid."""
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

    def integer(self, path, v, allowed):
        if isinstance(v, bool) or not isinstance(v, int):
            self.add(path, f"expected an integer, found {_kind(v)}")
        elif v not in allowed:
            self.add(path, f"{v} is not one of {', '.join(map(str, allowed))}")

    def choice(self, path, v, allowed):
        if not isinstance(v, str) or v not in allowed:
            self.add(path, f"expected one of {', '.join(repr(a) for a in allowed)}, found {v!r}")
            return None
        return v

    def knots(self, path, knots, key, count, max_ms):
        """Knot pairs with valid times; the values are only checked to be numbers."""
        if not isinstance(knots, tuple) or not all(isinstance(k, tuple) and len(k) == 2 for k in knots):
            self.add(path, f"expected (ms, {key}) knots")
            return None
        ok = count <= len(knots) <= MAX_KNOTS
        if not ok:
            self.add(path, f"needs {count} to {MAX_KNOTS} knots, found {len(knots)}")
        times = [self.number(f"{path}[{i}].ms", ms, 0.0, max_ms) for i, (ms, _) in enumerate(knots, 1)]
        values = [self.number(f"{path}[{i}].{key}", v) for i, (_, v) in enumerate(knots, 1)]
        if not ok or None in times or None in values:
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
        return list(zip(times, values))

    def shape(self, path, knots, max_ms):
        """An amplitude shape: gains in [0, 1], 0 at both ends, highest point exactly 1."""
        knots = self.knots(path, knots, "gain", 3, max_ms)
        if knots is None:
            return None
        gains = [g for _, g in knots]
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
        return knots


def _kind(v) -> str:
    return {bool: "a boolean", str: "a string", list: "an array", tuple: "an array", dict: "a table"}.get(
        type(v), type(v).__name__
    )


def _validate(p: Patch, problems: _Problems) -> None:
    _validate_seed(p.seed, problems)
    _validate_body(p.body, problems)
    if not isinstance(p.noise, tuple) or not all(isinstance(layer, Layer) for layer in p.noise):
        problems.add("noise", "expected noise layers")
    else:
        _validate_layers(p.noise, problems)
    _validate_drive(p.drive, problems)
    _validate_output(p.output, problems)


def _validate_seed(seed, problems: _Problems) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.add("seed", f"expected an integer, found {_kind(seed)}")
    elif not 0 <= seed <= MAX_SEED:
        problems.add("seed", f"{seed} is outside 0 … 2**63 - 1")


def _validate_body(b: Body, problems: _Problems) -> None:
    if not isinstance(b, Body):
        problems.add("body", "expected a body section")
        return
    tune = problems.number("body.tune_hz", b.tune_hz, 40.0, 4000.0)
    problems.number("body.level_db", b.level_db, -90.0, 24.0, silence=True)
    problems.number("body.damping", b.damping, 0.0, 3.0)
    ratios = _validate_modes(b.modes, problems)
    pitch = problems.knots("body.pitch", b.pitch, "ratio", 2, MAX_BODY_MS)
    if pitch is not None:
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
                    if not 10.0 <= f <= 20000.0:
                        where = "" if len(ratios) == 1 else f" for mode {i}"
                        problems.add(
                            f"body.pitch[{k}].ratio", f"body frequency {f:g} Hz{where} is outside 10 Hz … 20 kHz"
                        )
    problems.shape("body.amp", b.amp, MAX_BODY_MS)


def _validate_modes(modes, problems: _Problems):
    """The mode ratios if they are valid, else None."""
    if not isinstance(modes, tuple) or not all(isinstance(m, tuple) and len(m) == 2 for m in modes):
        problems.add("body.modes", "expected (ratio, level_db) modes")
        return None
    if not 1 <= len(modes) <= MAX_MODES:
        problems.add("body.modes", f"needs 1 to {MAX_MODES} modes, found {len(modes)}")
        return None
    ratios = [problems.number(f"body.modes[{i}].ratio", r, 1.0, MAX_RATIO) for i, (r, _) in enumerate(modes, 1)]
    levels = [problems.number(f"body.modes[{i}].level_db", v, -60.0, 0.0) for i, (_, v) in enumerate(modes, 1)]
    if ratios[0] is not None and ratios[0] != 1.0:
        problems.add("body.modes[1].ratio", "the first mode defines the tuning: its ratio must be exactly 1")
    for i in range(1, len(ratios)):
        if ratios[i] is not None and ratios[i - 1] is not None and not ratios[i] > ratios[i - 1]:
            problems.add(f"body.modes[{i + 1}].ratio", "mode ratios must strictly increase")
    if None not in levels and max(levels) != 0.0:
        problems.add("body.modes", f"the loudest mode must be exactly 0 dB, found {max(levels)!r}")
    return None if None in ratios else ratios


def _validate_layers(layers: tuple, problems: _Problems) -> None:
    if not 1 <= len(layers) <= MAX_LAYERS:
        problems.add("noise", f"needs 1 to {MAX_LAYERS} layers, found {len(layers)}")
    for j, layer in enumerate(layers, 1):
        _validate_layer(layer, f"noise[{j}]", problems)


def _validate_layer(layer: Layer, path: str, problems: _Problems) -> None:
    problems.number(f"{path}.level_db", layer.level_db, -90.0, 24.0, silence=True)
    band = layer.band_hz
    if not isinstance(band, tuple) or len(band) != 2:
        problems.add(f"{path}.band_hz", "expected [low, high] in Hz")
    else:
        lo = problems.number(f"{path}.band_hz[1]", band[0], 20.0, 20000.0)
        hi = problems.number(f"{path}.band_hz[2]", band[1], 20.0, 20000.0)
        if lo is not None and hi is not None and not lo < hi:
            problems.add(f"{path}.band_hz", f"the low edge {lo:g} Hz must be below the high edge {hi:g} Hz")
    problems.number(f"{path}.rattle", layer.rattle, 0.0, 1.0)
    problems.shape(f"{path}.amp", layer.amp, MAX_NOISE_MS)


def _validate_drive(d: Drive, problems: _Problems) -> None:
    if not isinstance(d, Drive):
        problems.add("drive", "expected a drive section")
    else:
        problems.number("drive.db", d.db, 0.0, 48.0)


def _validate_output(o: Output, problems: _Problems) -> None:
    if not isinstance(o, Output):
        problems.add("output", "expected an output section")
        return
    problems.integer("output.sample_rate", o.sample_rate, SAMPLE_RATES)
    problems.choice("output.format", o.format, FORMATS)
    mode = problems.choice("output.mode", o.mode, ("gain", "peak"))
    problems.number("output.db", o.db, -60.0, 0.0 if mode == "peak" else 24.0)
    problems.integer("output.polarity", o.polarity, (1, -1))


# Parsing.

_FIELDS = {
    "": ("instrument", "schema", "seed", "body", "noise", "drive", "output"),
    "body": ("tune_hz", "level_db", "damping", "modes", "pitch", "amp"),
    "noise": ("level_db", "band_hz", "rattle", "amp"),
    "drive": ("db",),
    "output": ("sample_rate", "format", "mode", "db", "polarity"),
}


def _keys(problems: _Problems, prefix: str, table: dict, expected) -> bool:
    for key in table:
        if key not in expected:
            problems.add(f"{prefix}{key}", "unknown field")
    missing = [key for key in expected if key not in table]
    for key in missing:
        problems.add(f"{prefix}{key}", "missing")
    return not missing


def _float(v):
    """TOML integers are accepted for floats; anything else is left for validation to name."""
    if isinstance(v, int) and not isinstance(v, bool):
        try:
            return float(v)
        except OverflowError:
            return v
    return v


def _pairs(problems: _Problems, path: str, items, keys):
    """An array of inline tables with exactly these keys, as tuples of numbers."""
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
        out.append(tuple(_float(item[k]) for k in keys))
    return tuple(out)


def _section(problems: _Problems, path: str, table, fields):
    if not isinstance(table, dict):
        problems.add(path, f"expected a table, found {_kind(table)}")
        return None
    return table if _keys(problems, f"{path}.", table, fields) else None


def _body(problems: _Problems, table):
    t = _section(problems, "body", table, _FIELDS["body"])
    if t is None:
        return None
    modes = _pairs(problems, "body.modes", t["modes"], ("ratio", "level_db"))
    pitch = _pairs(problems, "body.pitch", t["pitch"], ("ms", "ratio"))
    amp = _pairs(problems, "body.amp", t["amp"], ("ms", "gain"))
    if None in (modes, pitch, amp):
        return None
    return Body(_float(t["tune_hz"]), _float(t["level_db"]), _float(t["damping"]), modes, pitch, amp)


def _layer(problems: _Problems, path: str, table):
    t = _section(problems, path, table, _FIELDS["noise"])
    if t is None:
        return None
    amp = _pairs(problems, f"{path}.amp", t["amp"], ("ms", "gain"))
    if amp is None:
        return None
    band = tuple(map(_float, t["band_hz"])) if isinstance(t["band_hz"], list) else t["band_hz"]
    return Layer(_float(t["level_db"]), band, _float(t["rattle"]), amp)


def from_dict(data) -> Patch:
    """Build a patch from parsed TOML, reporting every problem at once."""
    if not isinstance(data, dict):
        raise PatchError(["expected a table"])
    # The instrument and schema come first: another instrument or a newer schema may share nothing else.
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

    problems = _Problems()
    _keys(problems, "", data, _FIELDS[""])
    seed = data.get("seed")
    if "seed" in data:
        _validate_seed(seed, problems)
    body = _body(problems, data["body"]) if "body" in data else None
    if body is not None:
        _validate_body(body, problems)
    noise = None
    if "noise" in data:
        tables = data["noise"]
        if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
            problems.add("noise", f"expected an array of tables ([[noise]]), found {_kind(tables)}")
        else:
            layers = [_layer(problems, f"noise[{j}]", t) for j, t in enumerate(tables, 1)]
            if None not in layers:
                noise = tuple(layers)
                _validate_layers(noise, problems)
    drive = output = None
    if "drive" in data and (t := _section(problems, "drive", data["drive"], _FIELDS["drive"])) is not None:
        drive = Drive(_float(t["db"]))
        _validate_drive(drive, problems)
    if "output" in data and (t := _section(problems, "output", data["output"], _FIELDS["output"])) is not None:
        output = Output(t["sample_rate"], t["format"], t["mode"], _float(t["db"]), t["polarity"])
        _validate_output(output, problems)
    if problems:
        raise PatchError(problems)
    return Patch(seed, body, noise, drive, output)


def loads(text: str) -> Patch:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PatchError([f"TOML: {e}"]) from None
    return from_dict(data)


def load(path) -> Patch:
    with open(path, encoding="utf-8") as f:
        return loads(f.read())


def to_dict(p: Patch) -> dict:
    b = p.body
    return {
        "instrument": INSTRUMENT,
        "schema": SCHEMA,
        "seed": p.seed,
        "body": {
            "tune_hz": b.tune_hz,
            "level_db": b.level_db,
            "damping": b.damping,
            "modes": [{"ratio": r, "level_db": v} for r, v in b.modes],
            "pitch": [{"ms": m, "ratio": r} for m, r in b.pitch],
            "amp": [{"ms": m, "gain": g} for m, g in b.amp],
        },
        "noise": [
            {
                "level_db": n.level_db,
                "band_hz": list(n.band_hz),
                "rattle": n.rattle,
                "amp": [{"ms": m, "gain": g} for m, g in n.amp],
            }
            for n in p.noise
        ],
        "drive": {"db": p.drive.db},
        "output": {
            "sample_rate": p.output.sample_rate,
            "format": p.output.format,
            "mode": p.output.mode,
            "db": p.output.db,
            "polarity": p.output.polarity,
        },
    }


# Canonical TOML.

_COMMENT_COLUMN = 28


def dumps(p: Patch) -> str:
    """The complete patch as TOML with the canonical comments; loads(dumps(p)) == p exactly."""
    b, o = p.body, p.output
    lines = [f'instrument = "{INSTRUMENT}"', f"schema = {SCHEMA}", f"seed = {p.seed}", "", "[body]"]
    lines += _rows(
        [
            ("tune_hz", _f(b.tune_hz), ""),
            ("level_db", _f(b.level_db), ""),
            ("damping", _f(b.damping), "mode i decays ratio^damping times faster"),
        ]
    )
    lines += _array("modes", [(("ratio", r), ("level_db", v)) for r, v in b.modes], "loudest mode exactly 0 dB")
    lines += _array(
        "pitch", [(("ms", m), ("ratio", r)) for m, r in b.pitch], "ratio to tune_hz; starts at 0 ms, ends at 1"
    )
    lines += _array(
        "amp", [(("ms", m), ("gain", g)) for m, g in b.amp], "shape: 0 at both ends, highest point exactly 1"
    )
    for j, n in enumerate(p.noise, 1):
        lines += ["", _note("[[noise]]", f"noise[{j}]")]
        lines += _rows(
            [
                ("level_db", _f(n.level_db), ""),
                ("band_hz", f"[{_f(n.band_hz[0])}, {_f(n.band_hz[1])}]", ""),
                ("rattle", _f(n.rattle), "0 … 1, gated at the body's fundamental"),
            ]
        )
        lines += _array("amp", [(("ms", m), ("gain", g)) for m, g in n.amp], "")
    lines += ["", "[drive]"] + _rows([("db", _f(p.drive.db), "small-signal gain; 0 = bypass")])
    lines += ["", "[output]"]
    lines += _rows(
        [
            ("sample_rate", str(o.sample_rate), ""),
            ("format", f'"{o.format}"', '"float32" | "pcm24" | "pcm16"'),
            ("mode", f'"{o.mode}"', '"gain" | "peak"'),
            ("db", _f(o.db), "gain in dB, or peak target in dBFS"),
            ("polarity", str(o.polarity), ""),
        ]
    )
    return "\n".join(lines) + "\n"


def _f(x: float) -> str:
    return repr(float(x))


def _note(line: str, note: str) -> str:
    return f"{(line + ' ').ljust(_COMMENT_COLUMN)}# {note}" if note else line


def _rows(rows) -> list[str]:
    width = max(len(k) for k, _, _ in rows)
    return [_note(f"{k.ljust(width)} = {v}", note) for k, v, note in rows]


def _array(name, items, note) -> list[str]:
    """An array of inline tables, with the first column aligned."""
    firsts = [f"{items[0][0][0]} = {_f(a[1])}," for a, _ in items] if items else []
    width = max(map(len, firsts), default=0)
    body = [f"  {{ {first.ljust(width)} {key} = {_f(v)} }}," for first, (_, (key, v)) in zip(firsts, items)]
    return [_note(f"{name} = [", note), *body, "]"]


# Tuning vocabulary shared by `halyard init`, `halyard tune`, the report and the WAV root note.

_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_pitch(hz: float) -> float:
    """Fractional MIDI note number (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(midi: int) -> str:
    return f"{_NOTES[midi % 12]}{midi // 12 - 1}"


def describe_tuning(hz: float) -> str:
    """For example 'F#3 -3.2 cents'."""
    p = midi_pitch(hz)
    n = round(p)
    cents = round(100 * (p - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{note_name(n)} {cents:+.1f} cents"


def parse_tune(text: str) -> float:
    """A note name such as D3, C#4 or Eb3 (Hz rounded to 3 decimals), or a frequency in Hz."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text.strip())
    if m:
        midi = _LETTERS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) + 1)
        return round(440.0 * 2.0 ** ((midi - 69) / 12), 3)
    try:
        hz = float(text)
    except ValueError:
        raise ValueError(f"not a note name or frequency: {text!r}") from None
    if not math.isfinite(hz):
        raise ValueError(f"not a note name or frequency: {text!r}")
    return hz


# Editing commands. Each returns a new, validated patch, or raises PatchError.

_US = Decimal("0.001")  # knot times are rewritten to 1 µs


def _dec(x: float) -> Decimal:
    """The decimal value as written."""
    return Decimal(repr(float(x)))


def _ms(x) -> float:
    return float(Decimal(x).quantize(_US))


def _significant(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def tune(p: Patch, text: str) -> Patch:
    """Set tune_hz from a note name or a frequency. Modes, glide and rattle follow, being ratios."""
    return dataclasses.replace(p, body=dataclasses.replace(p.body, tune_hz=parse_tune(text)))


def curve_names(p: Patch) -> tuple[str, ...]:
    return ("body.pitch", "body.amp", *(f"noise[{j}].amp" for j in range(1, len(p.noise) + 1)))


def _get_curve(p: Patch, name: str) -> tuple:
    if name == "body.pitch":
        return p.body.pitch
    if name == "body.amp":
        return p.body.amp
    return p.noise[_layer_index(p, name[: -len(".amp")])].amp


def _set_curves(p: Patch, curves: dict) -> Patch:
    body = p.body
    if "body.pitch" in curves:
        body = dataclasses.replace(body, pitch=curves["body.pitch"])
    if "body.amp" in curves:
        body = dataclasses.replace(body, amp=curves["body.amp"])
    noise = tuple(
        dataclasses.replace(layer, amp=curves[f"noise[{j}].amp"]) if f"noise[{j}].amp" in curves else layer
        for j, layer in enumerate(p.noise, 1)
    )
    return dataclasses.replace(p, body=body, noise=noise)


def _layer_index(p: Patch, name: str) -> int:
    """0-based index of a layer named as the patch paths name it: 'noise[2]' is the second layer."""
    m = re.fullmatch(r"noise\[(\d+)\]", name.strip())
    if not m or not 1 <= int(m.group(1)) <= len(p.noise):
        layers = ", ".join(f"noise[{j}]" for j in range(1, len(p.noise) + 1))
        raise PatchError([f"{name}: not a layer of this patch; it has {layers}"])
    return int(m.group(1)) - 1


def stretch(p: Patch, curves, factor: float, after_ms: float | None = None) -> Patch:
    """Rescale knot times after each curve's peak (0 ms for pitch), or after `after_ms`, by `factor`.

    `curves` names curves by path (`body.amp`, `body.pitch`, `noise[2].amp`);
    `all` names every amplitude shape. Times are rewritten to 1 µs.
    """
    if not (math.isfinite(factor) and factor > 0):
        raise PatchError([f"stretch: the factor must be a positive number, found {factor!r}"])
    if after_ms is not None and not (math.isfinite(after_ms) and after_ms >= 0):
        raise PatchError([f"--after: must be a time in ms from 0 on, found {after_ms!r}"])
    names = []
    for name in [curves] if isinstance(curves, str) else curves:
        if name == "all":
            names += [n for n in curve_names(p) if n.endswith(".amp")]
        elif name in curve_names(p):
            names.append(name)
        else:
            raise PatchError([f"{name}: not a curve of this patch; expected all or one of {', '.join(curve_names(p))}"])
    out = {}
    for name in dict.fromkeys(names):
        knots = _get_curve(p, name)
        if after_ms is not None:
            anchor = after_ms
        else:
            anchor = 0.0 if name == "body.pitch" else next(ms for ms, g in knots if g == 1.0)
        a, f = _dec(anchor), Decimal(repr(float(factor)))
        out[name] = tuple((ms if ms <= anchor else _ms(a + f * (_dec(ms) - a)), v) for ms, v in knots)
    return _set_curves(p, out)


SHOULDER_MS = 0.1  # the shoulder knot after each burst's peak (plan §2.2)
TAIL_DB = 60.0  # the last burst's exponential falls this far over its tail


def bursts(
    p: Patch,
    layer: str,
    count: int,
    spacing: float,
    *,
    start: float = 0.0,
    growth: float = 1.0,
    rise: float = 0.1,
    dip: float = 0.1,
    taper: float = 1.0,
    tail: float = 80.0,
) -> Patch:
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
    problems = _Problems()
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10:
        problems.add("--count", f"needs 1 to 10 bursts, found {count!r}")
    for name, value, ok in (
        ("--spacing", spacing, spacing > 0),
        ("--start", start, start >= 0),
        ("--growth", growth, growth > 0),
        ("--rise", rise, rise >= float(MIN_SPACING_MS)),
        ("--dip", dip, 0 <= dip < 1),
        ("--taper", taper, taper > 0),
        ("--tail", tail, tail > 3 * (SHOULDER_MS + float(MIN_SPACING_MS))),
    ):
        if not (math.isfinite(value) and ok):
            problems.add(name, f"{value!r} is out of range")
    if problems:
        raise PatchError(problems)

    peaks = [taper ** (i - (0 if taper <= 1 else count - 1)) for i in range(count)]
    peaks = [1.0 if pk == max(peaks) else _significant(pk) for pk in peaks]
    starts, gap = [start], spacing
    for _ in range(count - 1):
        starts.append(starts[-1] + gap)
        gap *= growth
    knots = [(0.0, 0.0)] + ([(start, 0.0)] if start > 0 else [])
    for i, (b, pk) in enumerate(zip(starts, peaks)):
        top = b + rise
        if i + 1 < count:
            nxt = starts[i + 1]
            floor = dip * min(pk, peaks[i + 1])
            if not nxt - top >= SHOULDER_MS + 2 * float(MIN_SPACING_MS):
                raise PatchError(
                    [
                        f"--rise: burst {i + 1} rises for {rise:g} ms, which leaves no room for its "
                        f"shoulder and dip before burst {i + 2} starts {nxt - b:g} ms later"
                    ]
                )
            decay = math.log(pk / max(floor, pk * 10 ** (-TAIL_DB / 20))) / (nxt - top)
            ends = [(nxt, _significant(floor))]
        else:
            decay = math.log(10) * TAIL_DB / 20 / tail
            ends = [(top + tail / 3, _significant(pk * 10 ** (-TAIL_DB / 60))), (top + tail, 0.0)]
        knots += [(top, pk), (top + SHOULDER_MS, _significant(pk * math.exp(-decay * SHOULDER_MS)))] + ends
    knots = tuple((_ms(Decimal(repr(ms))), g) for ms, g in knots)
    if len(knots) > MAX_KNOTS:
        raise PatchError([f"{layer}.amp: {count} bursts need {len(knots)} knots, more than the {MAX_KNOTS} allowed"])
    return _set_curves(p, {f"noise[{j + 1}].amp": knots})


def rises(values) -> list[tuple[int, int]]:
    """(first, last) knot indices of each rise: a maximal run of strictly increasing knots.

    Both ends of a rise have zero slope, the first because nothing before it is
    lower and the last because nothing after it is higher, so a rise can be
    lengthened without changing the rest of the curve.
    """
    out, i, n = [], 0, len(values)
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


def _lengthen_rises(knots: tuple, factor: Decimal) -> tuple:
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
    return tuple((float(t), v) for t, (_, v) in zip(new, knots))


def softer(p: Patch, db: float, glide: float = 2.0, contact: float = 0.2) -> Patch:
    """The same drum struck `db` softer, by the three rules of plan §2.7.

    With v = 10^(−db/20):
      * Level: every source level falls by db (silent sources stay silent).
      * Glide: the pitch curve's excess over 1 scales by v^glide.
      * Contact: every rise lengthens by v^(−contact), later knots shifting by
        the added time, and every layer's upper band edge scales by v^contact.
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

    def level(x):
        return x if x == -math.inf else float(round(_dec(x) - Decimal(repr(float(db))), 6))

    b = p.body
    pitch = b.pitch
    if glide:
        scale = Decimal(repr(v**glide))
        pitch = tuple((ms, float(round(1 + (_dec(r) - 1) * scale, 6))) for ms, r in b.pitch)
    body_amp = b.amp
    layers = []
    for layer in p.noise:
        amp, band = layer.amp, layer.band_hz
        if contact:
            amp = _lengthen_rises(amp, Decimal(repr(v**-contact)))
            band = (band[0], float(round(_dec(band[1]) * Decimal(repr(v**contact)), 1)))
        layers.append(dataclasses.replace(layer, level_db=level(layer.level_db), band_hz=band, amp=amp))
    if contact:
        body_amp = _lengthen_rises(b.amp, Decimal(repr(v**-contact)))
    body = dataclasses.replace(b, level_db=level(b.level_db), pitch=pitch, amp=body_amp)
    return dataclasses.replace(p, body=body, noise=tuple(layers))
