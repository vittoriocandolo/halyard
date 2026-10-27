# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Packs: velocity layers with round-robin takes and an SFZ mapping (plan §7.3).

Each patch is one velocity layer, loudest first, and each layer gets the same
number of takes: take j of a layer plays that layer's seed + j − 1. Every file
of a pack shares one gain, so the layers keep their natural levels; in peak
mode that gain brings the loudest file to the target.

The velocity mapping follows a square law, 40·log₁₀(v/127) dB. Each layer
sits at the velocity where the law reaches its loudness relative to the
loudest, and layers meet halfway between, in dB. Each layer's group carries a
curve point at every velocity in its range, so the level follows the law
across layer boundaries.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from . import __version__
from .loudness import momentary_max
from .patch import Patch, PatchError, load
from .render import Domain, RenderError, Stamp, ceiling, db, domain, output_stage, synthesize, take_seeds, write_take

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]
    type Audio = NDArray[np.float32]

SNARE_KEY, CLAP_KEY = 38, 39  # General MIDI: acoustic snare, hand clap
TOP_KEY = 127  # the highest MIDI key
LOUD_VOLUME_DB = 6.0  # the group volume some players stop at
UNEVEN_DB = 1.0  # a take spread over this suggests level matching


class PackError(RuntimeError):
    """A pack that cannot be built, with what to change."""


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
    loudness: float  # its first take's, LUFS, in the pack's files
    relative_db: float  # to the loudest layer
    lovel: int
    hivel: int
    volume_db: float  # the group's volume, which with the curve places the layer on the square law
    curve: tuple[float, ...]  # amplitude at lovel … hivel, at most 1
    take_loudness: tuple[float, ...]
    take_gain_db: tuple[float, ...]
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

    Layer i sits where 40·log₁₀(v/127) reaches its level; neighbours meet at the
    geometric mean of their velocities, halfway in dB.
    """
    natural = [127.0 * 10.0 ** (d / 40.0) for d in relative_db]
    bounds = [127.0, *(math.sqrt(a * b) for a, b in itertools.pairwise(natural)), 0.0]
    ranges = [(math.floor(bounds[i + 1]) + 1, math.floor(bounds[i])) for i in range(len(natural))]
    for i, (lo, hi) in enumerate(ranges):
        if lo > hi:
            msg = f"layers {i} and {i + 1} are too close in loudness to get velocities of their own"
            raise PackError(msg)
    return ranges


def curve_for(relative_db: float, lovel: int, hivel: int) -> tuple[float, tuple[float, ...]]:
    """Return the group volume in dB and the amplitude curve over lovel … hivel that put a layer on the square law.

    At velocity v the level should be 40·log₁₀(v/127) dB, and the layer's own
    level is relative_db, so its gain is (v/127)² / 10^(relative_db/20). The
    curve is that gain over its largest value, which becomes the volume.
    """
    gains = [(v / 127.0) ** 2 / 10.0 ** (relative_db / 20.0) for v in range(lovel, hivel + 1)]
    top = max(gains)
    return 20.0 * math.log10(top), tuple(g / top for g in gains)


def _synthesize(p: Patch) -> tuple[Floats, Domain]:
    """Render one take at unity gain, and judge it."""
    s = synthesize(p)
    return s.out, domain(s)


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
        ]
        for layer in self.layers:
            lines += [
                "",
                f"// {layer.name}: {layer.loudness:.1f} LUFS, {layer.relative_db:+.2f} dB",
                "<group>",
                f"lovel={layer.lovel}",
                f"hivel={layer.hivel}",
                f"seq_length={self.takes}",
                f"volume={layer.volume_db:.4f}",
            ]
            lines += [
                f"amp_velcurve_{v}={c:.6f}"
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
            if layer.volume_db > LOUD_VOLUME_DB:
                loud = (
                    f"its group volume is {layer.volume_db:+.1f} dB; some players stop at +6 dB, so "
                    "keep neighbouring layers within 12 dB"
                )
                rows.append(("", loud))
        rows.append(("render", f"{self.render_s:.1f} s"))
        return [f"{label:<9}{text}" for label, text in rows]


def _render(
    patches: list[tuple[str, Patch]],
    takes: int,
    workers: int | None,
) -> tuple[list[list[Floats]], list[list[Domain]]]:
    """Render every take of every layer at unity gain, in parallel processes, and judge each."""
    seeds = [take_seeds(p, takes) for _, p in patches]
    jobs = [dataclasses.replace(p, seed=s) for (_, p), ss in zip(patches, seeds, strict=True) for s in ss]
    n = min(len(jobs), workers or os.cpu_count() or 1)
    if n > 1:
        # Workers start from a fork server, or fresh where there is none (Windows), never by forking the caller: a
        # thread of the caller holding a lock, such as the table cache's, would leave a worker deadlocked on it.
        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        with ProcessPoolExecutor(n, mp_context=multiprocessing.get_context(method)) as pool:
            done = list(pool.map(_synthesize, jobs))
    else:
        done = [_synthesize(p) for p in jobs]
    outs = [[done[i * takes + j][0] for j in range(takes)] for i in range(len(patches))]
    doms = [[done[i * takes + j][1] for j in range(takes)] for i in range(len(patches))]
    return outs, doms


def _set_gain(first: Patch, outs: list[list[Floats]], gains: list[list[float]]) -> float:
    """Return the pack's one gain: the patches' own, or the one that brings the loudest file to the target."""
    if first.output.mode == "gain":
        return float(10.0 ** (first.output.db / 20.0))
    peak = max(
        float(np.max(np.abs(o), initial=0.0)) * g
        for row, grow in zip(outs, gains, strict=True)
        for o, g in zip(row, grow, strict=True)
    )
    target = min(10.0 ** (first.output.db / 20.0), ceiling(first.output.format))
    return target / peak if peak > 0 else 1.0


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
    check_outputs(patches)
    first = patches[0][1]
    outs, doms = _render(patches, options.takes, workers)
    rate = first.output.sample_rate
    loud = [[momentary_max(o, rate) for o in row] for row in outs]
    gains = [
        [
            (10.0 ** ((row[0] - x) / 20.0) if options.level_match and row[0] > -math.inf and x > -math.inf else 1.0)
            for x in row
        ]
        for row in loud
    ]
    set_gain = _set_gain(first, outs, gains)
    firsts = [row[0] for row in loud]
    _check_order(patches, firsts)
    relative = [x - firsts[0] for x in firsts]
    ranges = velocity_ranges(relative)
    level = db(set_gain)
    layers: list[Layer] = []
    audio: list[tuple[Audio, ...]] = []
    for i, ((name, p), (lo, hi)) in enumerate(zip(patches, ranges, strict=True)):
        volume, curve = curve_for(relative[i], lo, hi)
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
                volume,
                curve,
                tuple(x + level + db(g) for x, g in zip(loud[i], gains[i], strict=True)),
                tuple(db(g) for g in gains[i]),
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


def over(pk: Pack) -> float:
    """Return the largest sample over the format's ceiling, or 0 when the pack fits."""
    top = ceiling(pk.layers[0].patch.output.format)
    peak = max(float(np.max(np.abs(a), initial=0.0)) for row in pk.audio for a in row)
    return peak if peak > top else 0.0


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
    sfz.write_text(pk.sfz(), encoding="utf-8", newline="\n")  # the same bytes on every platform
    written.append(sfz)
    return written


def pack(
    paths: Iterable[str],
    out_dir: str | os.PathLike[str],
    options: Options = DEFAULTS,
    *,
    out: TextIO | None = None,
) -> int:
    """Run `halyard pack`: render the layers and takes, and write the files and the SFZ mapping."""
    report = out or sys.stdout
    patches = []
    for path in paths:
        try:
            patches.append((Path(path).stem, load(path)))
        except OSError as e:
            sys.stderr.write(f"{path}: {e.strerror}\n")
            return 2
        except PatchError as e:
            sys.stderr.write(f"{path}: invalid patch\n" + "".join(f"  {m}\n" for m in e.problems))
            return 2
    if options.key is not None and not 0 <= options.key <= TOP_KEY:
        sys.stderr.write(f"--key: {options.key} is not a MIDI key (0 … 127)\n")
        return 2
    try:
        pk = build(patches, options)
    except (PackError, RenderError) as e:
        sys.stderr.write(f"{e}\n")
        return 2
    peak = over(pk)
    if peak:
        report.write("\n".join(pk.lines(out_dir)) + "\n")
        top = ceiling(patches[0][1].output.format)
        sys.stderr.write(
            f"not written: the loudest file peaks at {db(peak):+.2f} dBFS, over {db(top):.2f} dBFS; lower output.db by "
            f"at least {db(peak / top):.2f} dB in every layer\n",
        )
        return 1
    written = write(pk, out_dir)  # every file first, so a closed output never leaves a pack half written
    report.write("\n".join(pk.lines(out_dir)) + "\n")
    report.write("".join(f"wrote {path}\n" for path in written))
    return 0
