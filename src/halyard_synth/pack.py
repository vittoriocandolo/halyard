# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Packs: velocity layers with round-robin takes and an SFZ mapping (docs/design.md: Packs).

Each patch is one velocity layer, loudest first, and each layer gets the same
number of takes: take j of a layer plays that layer's seed + j − 1. Every file
of a pack shares one gain, so the layers keep their levels against each other; in peak
mode that gain brings the loudest file to the target.

The velocity mapping follows a square law, 40·log₁₀(v/127) dB. A layer's
velocities run up to its own, the velocity where the law reaches its loudness
relative to the loudest, from just above the softer layer's own. Each layer's
group carries a curve point at every velocity in its range, so the level
follows the law across layer boundaries and no velocity plays a take louder
than it was rendered.

A pack is checked whole before any file is written: it is refused if any file has a sample
over the format's ceiling or a true peak above 0 dBTP, or a first or last
sample above -90 dBFS, which a sampler would hear as a click where the file
starts or stops.
"""

from __future__ import annotations

import dataclasses
import math
import multiprocessing
import os
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from . import __version__, cores, wav
from .loudness import momentary_max
from .patch import Patch, PatchError, load
from .render import (
    Domain,
    RenderError,
    Stamp,
    ceiling,
    db,
    domain,
    level_gain,
    output_stage,
    synthesize,
    take_seeds,
    write_take,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]
    type Audio = NDArray[np.float32]

SNARE_KEY, CLAP_KEY = 38, 39  # General MIDI: acoustic snare, hand clap
TOP_KEY = 127  # the highest MIDI key
UNEVEN_DB = 1.0  # a take spread over this suggests level matching
EDGE_DB = -90.0  # how quiet the first and last samples must be: below the smallest step of 16-bit PCM


class PackError(RuntimeError):
    """A pack that cannot be built, with what to change."""


SFZ_SYNTAX = ("<", ">", "=", "$", "\\", "//", "/*")  # what a sampler reads as syntax even inside a sample's name


def sfz_name_problem(name: str) -> str | None:
    """Return why `name` cannot stand for itself in an SFZ map, or None when it can.

    A sampler reads a map line by line: `<` opens a header, `word=` an opcode,
    `$` a variable, `//` and `/*` a comment and a backslash a folder, and the
    space around a value is trimmed. A name holding any of these, a control
    character or a line break would be read as something else: a crafted file
    name could add regions or opcodes to the map.
    """
    breaks = sorted({f"U+{ord(c):04X}" for c in name if unicodedata.category(c) in {"Cc", "Zl", "Zp"}})
    found = [*breaks, *(s for s in SFZ_SYNTAX if s in name)]
    if found:
        return f"{name!r} cannot name a file in an SFZ map: it holds {' '.join(found)}; rename it"
    if name != name.strip():
        return f"{name!r} cannot name a file in an SFZ map: a sampler trims the space around it; rename it"
    return None


@dataclass(frozen=True)
class Options:
    """How a pack is rendered: takes per layer, its MIDI key (None to choose it), and level matching."""

    takes: int = 1
    key: int | None = None
    level_match: bool = False


DEFAULTS = Options()


@dataclass(frozen=True)
class Layer:
    """One velocity layer: its patch, loudness, velocities, curve, and its takes' loudness, gains and domains."""

    name: str  # the patch file's stem
    patch: Patch
    loudness: float  # its takes' together, the mean of their powers, LUFS, in the pack's files
    relative_db: float  # to the loudest layer
    lovel: int
    hivel: int
    curve: tuple[float, ...]  # amplitude at lovel … hivel, at most 1
    take_loudness: tuple[float, ...]
    take_gain_db: tuple[float, ...]
    take_true_peak: tuple[float, ...]  # in the pack's files, linear
    domains: tuple[Domain, ...]

    @property
    def spread(self) -> float:
        """The range of the takes' loudness, in dB."""
        values = [x for x in self.take_loudness if x > -math.inf]
        return max(values) - min(values) if values else 0.0

    def files(self, stem: str) -> list[str]:
        """Return the names of the layer's files, one per take."""
        return [f"{stem}_v{self.hivel}_rr{j}.wav" for j in range(1, len(self.take_loudness) + 1)]


def velocity_ranges(relative_db: list[float]) -> list[tuple[int, int]]:
    """Return (lovel, hivel) per layer, loudest first, from each layer's loudness relative to the loudest.

    A layer's velocities run up to its own, the velocity where 40·log₁₀(v/127)
    reaches its level, from just above the softer layer's own. So the map only
    ever lowers a take, never plays it louder than it was rendered, and the
    law holds at every velocity.
    """
    own = [math.floor(127.0 * 10.0 ** (d / 40.0) + 1e-9) for d in relative_db]  # the law's inverse
    lows = [*(hi + 1 for hi in own[1:]), 1]
    ranges = list(zip(lows, own, strict=True))
    for i, (lo, hi) in enumerate(ranges):
        if lo > hi:
            msg = (
                f"layers {i + 1} and {i + 2} are within a velocity step of each other in loudness"
                if i + 1 < len(ranges)
                else f"layer {i + 1} is {-relative_db[i]:.1f} dB under the loudest, below the square law at velocity 1"
            )
            raise PackError(msg)
    return ranges


def curve_for(relative_db: float, lovel: int, hivel: int) -> tuple[float, ...]:
    """Return the amplitude over lovel … hivel that puts a layer on the square law, never above 1.

    At velocity v the level should be 40·log₁₀(v/127) dB, and the layer's own
    level is relative_db, so its gain is (v/127)² / 10^(relative_db/20): at most
    1 up to the layer's own velocity, where its range ends.
    """
    return tuple((v / 127.0) ** 2 / 10.0 ** (relative_db / 20.0) for v in range(lovel, hivel + 1))


def _synthesize(p: Patch) -> tuple[Floats, tuple[float, float], Domain]:
    """Render one take at unity gain, and return it, its sample and true peaks, and its judgement."""
    s = synthesize(p)
    return s.out, (s.peak("peak"), s.true_peak), domain(s)


def check_outputs(patches: list[tuple[str, Patch]]) -> None:
    """Refuse layers whose output settings differ."""
    first_name, first = patches[0]
    for name, p in patches[1:]:
        if p.output != first.output:
            fields = [
                f.name
                for f in dataclasses.fields(p.output)
                if getattr(p.output, f.name) != getattr(first.output, f.name)
            ]
            msg = (
                f"{name}: its output settings differ from {first_name}'s ({', '.join(f'output.{f}' for f in fields)}); "
                "the layers of a pack share one output"
            )
            raise PackError(msg)


@dataclass(frozen=True)
class Pack:
    """A rendered pack: its name, key, takes, one gain, its layers and their audio, ready to write."""

    stem: str
    key: int
    takes: int
    set_gain_db: float
    layers: tuple[Layer, ...]
    audio: tuple[tuple[Audio, ...], ...]  # per layer, per take
    level_match: bool
    render_s: float

    def sfz(self) -> str:
        """Write the SFZ mapping."""
        layers, takes = "s" if len(self.layers) > 1 else "", "s" if self.takes > 1 else ""
        head = f"{len(self.layers)} velocity layer{layers} × {self.takes} take{takes}, written by halyard {__version__}"
        lines = [
            f"// {self.stem}: {head}",
            "// Velocity follows a square law, 40·log10(v/127) dB, across layer boundaries.",
            "",
            "<global>",
            f"key={self.key}",
            "pitch_keytrack=0",
            "amp_veltrack=100",
            "loop_mode=one_shot",  # a drum plays out whatever the note's length
        ]
        for layer in self.layers:
            lines += [
                "",
                f"// {layer.name}: {layer.loudness:.1f} LUFS, {layer.relative_db:+.2f} dB",
                "<group>",
                f"lovel={layer.lovel}",
                f"hivel={layer.hivel}",
                f"seq_length={self.takes}",
            ]
            lines += [
                f"amp_velcurve_{v}={c:.9f}"  # nine decimals: velocity 1 within 1e-4 dB of the law
                for v, c in zip(range(layer.lovel, layer.hivel + 1), layer.curve, strict=True)
            ]
            for j, name in enumerate(layer.files(self.stem), 1):
                lines.append(f"<region> sample={name} seq_position={j}")
        return "\n".join(lines) + "\n"

    def lines(self, out_dir: str | os.PathLike[str]) -> list[str]:
        """Describe the pack, a labelled line at a time."""
        files = len(self.layers) * self.takes
        match = ", level-matched within each layer" if self.level_match else ""
        what = (
            f"{len(self.layers)} velocity layers × {self.takes} takes = {files} files in {out_dir}, "
            f"key {self.key}, one gain of {self.set_gain_db:+.2f} dB{match}"
        )
        rows = [("pack", what)]
        for i, layer in enumerate(self.layers, 1):
            worst = min(d.margin_db for d in layer.domains)
            outside = sum(not d.inside for d in layer.domains)
            status = (
                f"{outside} take(s) OUTSIDE the validated domain; check with `halyard verify`"
                if outside
                else f"inside, worst margin {worst:.1f} dB"
            )
            spread = f", take spread {layer.spread:.2f} dB" + (
                " (over 1 dB: consider --level-match)" if layer.spread > UNEVEN_DB and not self.level_match else ""
            )
            levels = (
                f"{layer.name}: {layer.loudness:.1f} LUFS ({layer.relative_db:+.1f} dB), "
                f"velocities {layer.lovel}–{layer.hivel}{spread}; {status}"
            )
            rows.append((f"layer {i}", levels))
        rows.append(("render", f"{self.render_s:.1f} s"))
        return [f"{label:<9}{text}" for label, text in rows]


def _render(
    patches: list[tuple[str, Patch]],
    takes: int,
    workers: int | None,
) -> tuple[list[list[Floats]], list[list[tuple[float, float]]], list[list[Domain]]]:
    """Render every take of every layer at unity gain, in parallel processes: each one, its peaks and its judgement."""
    seeds = [take_seeds(p, takes) for _, p in patches]
    jobs = [dataclasses.replace(p, seed=s) for (_, p), ss in zip(patches, seeds, strict=True) for s in ss]
    n = min(len(jobs), workers or os.cpu_count() or 1)
    if n > 1:
        # Workers start from a fork server, or fresh where there is none (Windows), never by forking the caller: a
        # thread of the caller holding a lock, such as the table cache's, would leave a worker deadlocked on it.
        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        context = multiprocessing.get_context(method)
        with ProcessPoolExecutor(n, mp_context=context, initializer=cores.share, initargs=(n,)) as pool:
            done = list(pool.map(_synthesize, jobs))
    else:
        done = [_synthesize(p) for p in jobs]
    outs = [[done[i * takes + j][0] for j in range(takes)] for i in range(len(patches))]
    peaks = [[done[i * takes + j][1] for j in range(takes)] for i in range(len(patches))]
    doms = [[done[i * takes + j][2] for j in range(takes)] for i in range(len(patches))]
    return outs, peaks, doms


def _set_gain(first: Patch, peaks: list[list[tuple[float, float]]], gains: list[list[float]]) -> float:
    """Return the pack's one gain: the patches' own, or the one that brings the loudest file to the target."""
    scaled = [
        (s * g, t * g) for row, grow in zip(peaks, gains, strict=True) for (s, t), g in zip(row, grow, strict=True)
    ]
    top = ceiling(first.output.format)
    return level_gain(first.output, [s for s, _ in scaled], [t for _, t in scaled], top)


def mean_loudness(loudness: Sequence[float]) -> float:
    """Return takes' loudness together, in LUFS: the mean of their powers; -inf when every one is silent."""
    powers = [10.0 ** (x / 10.0) for x in loudness if x > -math.inf]
    return 10.0 * math.log10(sum(powers) / len(loudness)) if powers else -math.inf


def _check_order(patches: list[tuple[str, Patch]], firsts: list[float]) -> None:
    """Refuse a silent layer, and layers not given loudest first."""
    if any(x == -math.inf for x in firsts):
        msg = f"{patches[firsts.index(-math.inf)][0]}: a silent layer cannot be mapped"
        raise PackError(msg)
    for i in range(1, len(firsts)):
        if not firsts[i] < firsts[i - 1]:
            msg = (
                f"{patches[i][0]} ({firsts[i]:.1f} LUFS) is not softer than {patches[i - 1][0]} "
                f"({firsts[i - 1]:.1f} LUFS): give the layers loudest first"
            )
            raise PackError(msg)


def build(
    patches: list[tuple[str, Patch]],
    options: Options = DEFAULTS,
    *,
    stem: str | None = None,
    workers: int | None = None,
) -> Pack:
    """Render every layer's takes and derive the mapping; nothing is written."""
    start = time.perf_counter()
    if not patches:
        msg = "a pack needs at least one patch"
        raise PackError(msg)
    names = dict.fromkeys([stem or patches[0][0], *(name for name, _ in patches)])
    if unnamable := [problem for name in names if (problem := sfz_name_problem(name))]:
        raise PackError("\n".join(unnamable))
    check_outputs(patches)
    first = patches[0][1]
    outs, peaks, doms = _render(patches, options.takes, workers)
    if options.level_match and (varied := [n for n, p in patches if p.takes and p.takes.strength_db]):
        msg = f"{', '.join(varied)}: level matching would undo how much harder or softer [takes] strikes each take"
        raise PackError(msg)
    rate = first.output.sample_rate
    loud = [[momentary_max(o, rate) for o in row] for row in outs]
    # A layer is as loud as its takes together, so no one take, struck harder or softer, decides where it goes.
    firsts = [mean_loudness(row) for row in loud]
    gains = [
        [(10.0 ** ((m - x) / 20.0) if options.level_match and m > -math.inf and x > -math.inf else 1.0) for x in row]
        for row, m in zip(loud, firsts, strict=True)
    ]
    set_gain = _set_gain(first, peaks, gains)
    _check_order(patches, firsts)
    relative = [x - firsts[0] for x in firsts]
    ranges = velocity_ranges(relative)
    level = db(set_gain)
    layers: list[Layer] = []
    audio: list[tuple[Audio, ...]] = []
    for i, ((name, p), (lo, hi)) in enumerate(zip(patches, ranges, strict=True)):
        curve = curve_for(relative[i], lo, hi)
        files = tuple(output_stage(o, set_gain * g, p.output.polarity) for o, g in zip(outs[i], gains[i], strict=True))
        audio.append(files)
        layers.append(
            Layer(
                name,
                p,
                firsts[i] + level,
                relative[i],
                lo,
                hi,
                curve,
                tuple(x + level + db(g) for x, g in zip(loud[i], gains[i], strict=True)),
                tuple(db(g) for g in gains[i]),
                tuple(t * set_gain * g for (_, t), g in zip(peaks[i], gains[i], strict=True)),
                tuple(doms[i]),
            ),
        )
    key = options.key
    if key is None:
        key = CLAP_KEY if all(p.body.level_db == -math.inf for _, p in patches) else SNARE_KEY
    return Pack(
        stem or patches[0][0],
        key,
        options.takes,
        level,
        tuple(layers),
        tuple(audio),
        options.level_match,
        time.perf_counter() - start,
    )


def gates(pk: Pack) -> list[str]:
    """Return why the pack cannot be written, a line per file and problem: nothing, when every file passes."""
    top = ceiling(pk.layers[0].patch.output.format)
    return [
        f"{name}: {problem}"
        for layer, files in zip(pk.layers, pk.audio, strict=True)
        for name, a, true_peak in zip(layer.files(pk.stem), files, layer.take_true_peak, strict=True)
        for problem in file_problems(a, true_peak, top)
    ]


def file_problems(a: Audio, true_peak: float, top: float) -> list[str]:
    """Return why one file cannot go into a pack: nothing, when it passes every gate."""
    problems = []
    peak = float(np.max(np.abs(a), initial=0.0))
    if peak > top:
        problems.append(f"over the format's ceiling by {db(peak / top):.4f} dB")
    if true_peak > 1.0:
        problems.append(f"true peak {db(true_peak):+.2f} dBTP, above 0 dBTP")
    for edge, sample in (("first", a[0]), ("last", a[-1])):
        level = db(abs(float(sample)))  # a render always has samples: at least the decimator's tail
        if level > EDGE_DB:
            problems.append(f"{edge} sample at {level:.1f} dBFS, above {EDGE_DB:g} dBFS: it would click")
    return problems


def reduction_db(pk: Pack) -> float:
    """Return how far output.db must come down for every file to fit the ceiling and 0 dBTP: 0 when they do."""
    top = ceiling(pk.layers[0].patch.output.format)
    worst = max(
        max(float(np.max(np.abs(a), initial=0.0)) / top, true_peak)
        for layer, files in zip(pk.layers, pk.audio, strict=True)
        for a, true_peak in zip(files, layer.take_true_peak, strict=True)
    )
    return max(0.0, db(worst))


def write(pk: Pack, folder: str | os.PathLike[str]) -> list[Path]:
    """Write every file of the pack and its SFZ mapping, and return their paths."""
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (layer, files) in enumerate(zip(pk.layers, pk.audio, strict=True), 1):
        for j, (name, a) in enumerate(zip(layer.files(pk.stem), files, strict=True), 1):
            path = out_dir / name
            write_take(
                path,
                a,
                layer.patch,
                Stamp(layer.patch.seed + j - 1, j, pk.takes, pk.set_gain_db, layer.take_gain_db[j - 1]),
                extra={
                    "pack": {
                        "layer": i,
                        "layers": len(pk.layers),
                        "lovel": layer.lovel,
                        "hivel": layer.hivel,
                        "key": pk.key,
                    },
                },
            )
            written.append(path)
    sfz = out_dir / f"{pk.stem}.sfz"
    wav.save(sfz, pk.sfz().encode("utf-8"))  # the same bytes on every platform, whole or not at all
    written.append(sfz)
    return written


def _load(paths: Iterable[str]) -> list[tuple[str, Patch]]:
    """Read every layer's patch, or refuse the pack naming the file that cannot be read and why."""
    patches = []
    for path in paths:
        try:
            patches.append((Path(path).stem, load(path)))
        except OSError as e:
            msg = f"{path}: {e.strerror}"
            raise PackError(msg) from None
        except PatchError as e:
            raise PackError("\n".join([f"{path}: invalid patch", *(f"  {m}" for m in e.problems)])) from None
    return patches


def _check_key(key: int | None) -> None:
    """Refuse a key that is not a MIDI key."""
    if key is not None and not 0 <= key <= TOP_KEY:
        msg = f"--key: {key} is not a MIDI key (0 … 127)"
        raise PackError(msg)


def pack(
    paths: Iterable[str],
    out_dir: str | os.PathLike[str],
    options: Options = DEFAULTS,
    *,
    out: TextIO | None = None,
) -> int:
    """Run `halyard pack`: render the layers and takes, and write the files and the SFZ mapping."""
    report = out or sys.stdout
    try:
        patches = _load(paths)
        _check_key(options.key)
        pk = build(patches, options)
    except (PackError, RenderError) as e:
        sys.stderr.write(f"{e}\n")
        return 2
    if problems := gates(pk):
        report.write("\n".join(pk.lines(out_dir)) + "\n")
        lines = ["not written: these files fail the pack's gates", *(f"  {p}" for p in problems)]
        if (lower := reduction_db(pk)) > 0:
            lines.append(f"lower output.db by at least {lower:.2f} dB in every layer")
        sys.stderr.write("\n".join(lines) + "\n")
        return 1
    try:
        written = write(pk, out_dir)  # every file first, so a closed output never leaves a pack half written
    except OSError as e:
        sys.stderr.write(f"{e.filename or out_dir}: {e.strerror or e}\n")
        return 2
    report.write("\n".join(pk.lines(out_dir)) + "\n")
    report.write("".join(f"wrote {path}\n" for path in written))
    return 0
