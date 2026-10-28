# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the documentation's generated parts from the code: figures, sounds and the references' gallery.

Every figure and sound is rendered from a shipped reference patch, so the
documentation shows what the code does, and `tests/test_docs.py` fails when
any of it is out of date:

  * `docs/figures/<reference>.svg`: the reference's level over time and its spectrum;
  * `docs/figures/sweep.svg`, `softer.svg` and `flam.svg`: what the brush's sweep,
    `softer` and `flam` do, measured on the references;
  * `docs/audio/<reference>.wav`: the reference as 24-bit PCM, matched in loudness;
  * `docs/references.md`: the gallery, each reference with its sound, figure, report
    and patch, described by the user guide's table of references.

The figures are SVG written here, with no plotting library, and follow the
reader's light or dark theme.

    uv run python -m docs.build          # write them
    uv run python -m docs.build --check  # list what is out of date, and fail if any is
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from halyard_synth import patch, wav
from halyard_synth.cli import REFERENCES, reference_text
from halyard_synth.curve import Curve
from halyard_synth.render import Stamp, render, write_take
from tools import say

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]

DOCS = Path(__file__).resolve().parent
FIGURES = DOCS / "figures"
AUDIO = DOCS / "audio"
GALLERY = DOCS / "references.md"
GUIDE = DOCS / "GUIDE.md"

FLOOR_DB = -72.0  # the figures' lowest level
WINDOW_MS = 2.0  # the level's RMS window
HOP_MS = 0.25
SMOOTH_OCTAVES = 1 / 6  # the spectrum's smoothing
LOWEST_HZ, HIGHEST_HZ = 20.0, 20000.0
TAKES = 8  # the takes a centroid is averaged over
CONCEPTS = ("sweep", "softer", "flam")  # the figures of what the brush's sweep, softer and flam do
DOCS_FORMAT = "pcm24"  # the gallery's sounds: small, and exact far below audibility

# Measurements.


def level_db(x: Floats, rate: int) -> tuple[Floats, Floats]:
    """Return the times in ms and the level in dB below its peak: RMS over 2 ms windows every 0.25 ms."""
    w, hop = round(WINDOW_MS / 1000 * rate), round(HOP_MS / 1000 * rate)
    padded = np.concatenate((np.zeros(w // 2), x, np.zeros(w)))
    power = np.convolve(padded**2, np.ones(w) / w, mode="valid")[: x.size : hop]
    db: Floats = 10 * np.log10(np.maximum(power / power.max(), 10 ** (FLOOR_DB / 10)))
    times: Floats = np.arange(db.size, dtype=np.float64) * HOP_MS
    return times, db


def spectrum_db(x: Floats, rate: int) -> tuple[Floats, Floats]:
    """Return frequencies and the hit's whole spectrum in dB below its peak, smoothed over a sixth of an octave."""
    n = 1 << max(16, math.ceil(math.log2(x.size)))
    power = np.abs(np.fft.rfft(x, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / rate)
    centres = LOWEST_HZ * 2.0 ** np.arange(0, math.log2(HIGHEST_HZ / LOWEST_HZ) + 1e-9, 1 / 24)
    half = 2.0 ** (SMOOTH_OCTAVES / 2)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    lo, hi = np.searchsorted(f, centres / half), np.searchsorted(f, centres * half)
    mean = (cumulative[hi] - cumulative[lo]) / np.maximum(hi - lo, 1)
    db: Floats = 10 * np.log10(np.maximum(mean / mean.max(), 10 ** (FLOOR_DB / 10)))
    return centres, db


def mean_centroid(p: patch.Patch, rate: int) -> tuple[Floats, Floats]:
    """Return the times in ms and the spectral centroid of the patch's first takes' summed power, over 20 ms windows.

    Summing the takes' spectra before taking the centroid averages away the
    noise's own wandering, so what is left is the band's motion. Windows where
    the hit is more than 26 dB down are left out.
    """
    takes = [_audio(dataclasses.replace(p, seed=p.seed + k)) for k in range(TAKES)]
    size = min(x.size for x in takes)
    w, hop = round(0.02 * rate), round(0.005 * rate)
    f = np.fft.rfftfreq(w, 1 / rate)
    window = np.hanning(w)
    starts = np.arange(0, size - w, hop)
    power = np.zeros((starts.size, f.size))
    for x in takes:
        frames = np.stack([x[a : a + w] * window for a in starts])
        power += np.abs(np.fft.rfft(frames, axis=1)) ** 2
    total = power.sum(axis=1)
    keep = total >= 10 ** (-26 / 10) * total.max()
    times: Floats = 1000 * (starts[keep] + w / 2) / rate
    centroid: Floats = (power[keep] @ f) / total[keep]
    return times, centroid


# Figures.


class Series(NamedTuple):
    """One line of a panel: its points and its style, `line` or `line2`, and its label."""

    x: Floats
    y: Floats
    style: str = "line"
    label: str = ""


@dataclasses.dataclass(frozen=True)
class Panel:
    """A plot: its series, its axes' ranges, ticks and titles; a logarithmic x axis if `log`."""

    title: str
    series: Sequence[Series]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_ticks: Sequence[tuple[float, str]]
    y_ticks: Sequence[tuple[float, str]]
    x_label: str
    log: bool = False


WIDTH, HEIGHT = 720, 260
MARGIN = {"left": 52, "right": 16, "top": 30, "bottom": 40}
GAP = 44
LABEL_EM = 6.2  # a legend character's width, about, at 11 px

STYLE = """
  .bg { fill: #ffffff; }
  .frame { fill: none; stroke: #8c959f; stroke-width: 1; }
  .grid { stroke: #d0d7de; stroke-width: 0.5; }
  .line { fill: none; stroke: #0969da; stroke-width: 1.5; stroke-linejoin: round; }
  .line2 { fill: none; stroke: #bc4c00; stroke-width: 1.5; stroke-linejoin: round; }
  .line3 { fill: none; stroke: #1a7f37; stroke-width: 1.5; stroke-linejoin: round; }
  text { fill: #1f2328; font: 11px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .title { font-weight: 600; font-size: 12px; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; }
    .frame { stroke: #6e7681; }
    .grid { stroke: #30363d; }
    .line { stroke: #4493f8; }
    .line2 { stroke: #f0883e; }
    .line3 { stroke: #3fb950; }
    text { fill: #e6edf3; }
  }
"""


def _n(v: float) -> str:
    """Write a coordinate to 0.1 px."""
    return f"{v:.1f}"


def svg(panels: Sequence[Panel], description: str) -> str:
    """Return panels side by side as an SVG document that follows the reader's theme."""
    inner = (WIDTH - MARGIN["left"] - MARGIN["right"] - GAP * (len(panels) - 1)) / len(panels)
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}"'
            f' height="{HEIGHT}" role="img" aria-label="{description}">'
        ),
        f"<style>{STYLE}</style>",
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}"/>',
    ]
    for i, p in enumerate(panels):
        left = MARGIN["left"] + i * (inner + GAP)
        right = left + inner

        def sx(v: float, p: Panel = p, left: float = left) -> float:
            a, b = p.x_range
            u = (math.log(v / a) / math.log(b / a)) if p.log else (v - a) / (b - a)
            return left + u * inner

        def sy(v: float, p: Panel = p) -> float:
            a, b = p.y_range
            return bottom - (v - a) / (b - a) * (bottom - top)

        for v, text in p.y_ticks:
            y = sy(v)
            parts.append(f'<line class="grid" x1="{_n(left)}" y1="{_n(y)}" x2="{_n(right)}" y2="{_n(y)}"/>')
            parts.append(f'<text x="{_n(left - 6)}" y="{_n(y + 4)}" text-anchor="end">{text}</text>')
        for v, text in p.x_ticks:
            x = sx(v)
            parts.append(f'<line class="grid" x1="{_n(x)}" y1="{_n(top)}" x2="{_n(x)}" y2="{_n(bottom)}"/>')
            parts.append(f'<text x="{_n(x)}" y="{_n(bottom + 15)}" text-anchor="middle">{text}</text>')
        parts.append(
            f'<rect class="frame" x="{_n(left)}" y="{_n(top)}" width="{_n(inner)}" height="{_n(bottom - top)}"/>',
        )
        parts.append(f'<text class="title" x="{_n(left)}" y="{_n(top - 10)}">{p.title}</text>')
        parts.append(
            f'<text x="{_n((left + right) / 2)}" y="{_n(bottom + 32)}" text-anchor="middle">{p.x_label}</text>',
        )
        for k, s in enumerate(p.series):
            keep = (s.x >= p.x_range[0]) & (s.x <= p.x_range[1])
            ys = np.clip(s.y[keep], *p.y_range)
            points = " ".join(f"{_n(sx(float(a)))},{_n(sy(float(b)))}" for a, b in zip(s.x[keep], ys, strict=True))
            parts.append(f'<polyline class="{s.style}" points="{points}"/>')
            if s.label:
                y = top + 14 + 14 * k
                start = right - 34 - LABEL_EM * max(len(t.label) for t in p.series)
                key = f'x1="{_n(start)}" y1="{_n(y - 4)}" x2="{_n(start + 18)}" y2="{_n(y - 4)}"'
                parts.append(f'<line class="{s.style}" {key}/>')
                parts.append(f'<text x="{_n(start + 24)}" y="{_n(y)}">{s.label}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _time_ticks(length_ms: float) -> tuple[float, list[tuple[float, str]]]:
    """Return a round span for a time axis and its ticks."""
    for span, step in (
        (10, 2),
        (25, 5),
        (50, 10),
        (100, 20),
        (200, 50),
        (300, 50),
        (500, 100),
        (800, 200),
        (1000, 200),
    ):
        if length_ms <= span:
            return float(span), [(float(t), f"{t:g}") for t in range(0, span + 1, step)]
    span = math.ceil(length_ms / 500) * 500
    return float(span), [(float(t), f"{t:g}") for t in range(0, span + 1, 500)]


LEVEL_TICKS = [(float(v), f"{v:g}".replace("-", "−")) for v in range(0, -73, -12)]
HZ_TICKS = [(20.0, "20"), (100.0, "100"), (1000.0, "1k"), (10000.0, "10k"), (20000.0, "20k")]


def _level_panel(series: Sequence[Series], length_ms: float, title: str = "Level, dB below the peak") -> Panel:
    """Return a panel of levels over time."""
    span, ticks = _time_ticks(length_ms)
    return Panel(title, series, (0.0, span), (FLOOR_DB, 0.0), ticks, LEVEL_TICKS, "ms")


def _spectrum_panel(series: Sequence[Series], title: str = "Spectrum, dB below its peak") -> Panel:
    """Return a panel of spectra."""
    return Panel(title, series, (LOWEST_HZ, HIGHEST_HZ), (FLOOR_DB, 0.0), HZ_TICKS, LEVEL_TICKS, "Hz", log=True)


def reference_figure(name: str, x: Floats, rate: int) -> str:
    """Return a reference's figure: its level over time and its spectrum."""
    t, lv = level_db(x, rate)
    f, sp = spectrum_db(x, rate)
    return svg(
        [_level_panel([Series(t, lv)], 1000 * x.size / rate), _spectrum_panel([Series(f, sp)])],
        f"The {name} reference: its level over time and its spectrum",
    )


def _audio(p: patch.Patch) -> Floats:
    """Render a patch at its own gain."""
    a, _ = render(p)
    return a.astype(np.float64)


def concept_figures() -> dict[str, str]:
    """Return the figures of the brush's sweep, of `softer` and of `flam`, each measured on a reference."""
    out = {}
    brush = patch.loads(reference_text("brush"))
    alone = dataclasses.replace(
        brush,
        body=dataclasses.replace(brush.body, level_db=-math.inf),
        noise=tuple(dataclasses.replace(n, level_db=-math.inf) for n in brush.noise),
    )
    held = dataclasses.replace(alone, brush=(dataclasses.replace(alone.brush[0], sweep=((0.0, 1.0), (1.0, 1.0))),))
    rate = brush.output.sample_rate
    swept_t, swept = mean_centroid(alone, rate)
    held_t, steady = mean_centroid(held, rate)
    knots = patch.curve_knots(brush, "brush[1].sweep")
    curve = Curve.pchip([ms / 1000 for ms, _ in knots], [r for _, r in knots], first=0.0, last=0.0)
    ratio_t = np.linspace(0.0, 400.0, 401)
    span = (1000.0 * math.floor(min(swept.min(), steady.min()) / 1000), 1000.0 * math.ceil(swept.max() / 1000))
    centroid_ticks = [(v, f"{v / 1000:g}k") for v in np.arange(span[0], span[1] + 1, 1000.0)]
    time_ticks = [(float(v), f"{v:g}") for v in range(0, 401, 100)]
    out["sweep"] = svg(
        [
            Panel(
                f"Spectral centroid, the brush alone, {TAKES} takes",
                [Series(swept_t, swept, "line", "swept"), Series(held_t, steady, "line2", "held at 1")],
                (0.0, 400.0),
                span,
                time_ticks,
                centroid_ticks,
                "ms",
            ),
            Panel(
                "Its sweep: both band edges times",
                [Series(ratio_t, curve(ratio_t / 1000), "line3")],
                (0.0, 400.0),
                (0.5, 2.0),
                time_ticks,
                [(v, f"{v:g}") for v in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)],
                "ms",
            ),
        ],
        "The brush reference's spectral centroid follows its sweep, and stays put with the sweep held",
    )
    crack = patch.loads(reference_text("crack"))
    rate = crack.output.sample_rate
    styles = ("line", "line2", "line3")
    ladder = [(db, _audio(patch.softer(crack, db))) for db in (0, 12, 24)]
    peak = max(float(np.max(np.abs(x))) for _, x in ladder)
    level_series, spectrum_series = [], []
    for (db, x), style in zip(ladder, styles, strict=True):
        t, lv = level_db(x, rate)
        lv = lv + 20 * math.log10(float(np.max(np.abs(x))) / peak)  # each against the loudest
        f, sp = spectrum_db(x, rate)
        label = "as written" if db == 0 else f"{db} dB softer"
        level_series.append(Series(t, lv, style, label))
        spectrum_series.append(Series(f, sp, style, label))
    out["softer"] = svg(
        [
            _level_panel(level_series, 120.0, "Level, dB below the hardest hit"),
            _spectrum_panel(spectrum_series, "Spectrum, each against its own peak"),
        ],
        "The crack reference struck 12 and 24 dB softer: quieter, its rise slower and its top darker",
    )
    snare = patch.loads(reference_text("tone"))
    rate = snare.output.sample_rate
    x, flammed = _audio(snare), _audio(patch.flam(snare, 25.0))
    peak = float(np.max(np.abs(flammed)))
    t1, l1 = level_db(x, rate)
    t2, l2 = level_db(flammed, rate)
    out["flam"] = svg(
        [
            _level_panel(
                [
                    Series(t2, l2 + 20 * math.log10(float(np.max(np.abs(flammed))) / peak), "line", "flam --gap 25"),
                    Series(
                        t1 + 25.0,
                        l1 + 20 * math.log10(float(np.max(np.abs(x))) / peak),
                        "line2",
                        "one stroke, 25 ms on",
                    ),
                ],
                200.0,
                "Level, dB below the flam's peak",
            ),
        ],
        "The tone reference as a flam: a grace note 12 dB softer, then the main stroke 25 ms later",
    )
    return out


# The gallery.

TABLE_ROW = re.compile(r"^\| `(?P<name>[a-z-]+)` \| (?P<what>[^|]+) \| (?P<try>[^|]+) \|$")


def descriptions() -> dict[str, tuple[str, str]]:
    """Return each reference's description and what to try, from the user guide's table: one source for both."""
    text = GUIDE.read_text(encoding="utf-8")
    section = text[text.index("## The references") : text.index("\n## ", text.index("## The references") + 1)]
    found = {}
    for line in section.splitlines():
        m = TABLE_ROW.match(line)
        if m:
            found[m["name"]] = (m["what"].strip(), m["try"].strip())
    return found


def report_text(p: patch.Patch) -> str:
    """Return a render's report as `halyard render` prints it, without the render time, which varies."""
    _, report = render(p)
    lines = [line for line in report.lines() if not line.startswith("render ")]
    return "\n".join(lines)


def gallery() -> str:
    """Return the references' gallery, in markdown."""
    found = descriptions()
    missing = [name for name in REFERENCES if name not in found]
    if missing:
        msg = f"the user guide's table of references lacks {', '.join(missing)}"
        raise ValueError(msg)
    lines = [
        "# The references",
        "",
        "<!-- Generated by `uv run python -m docs.build` from the shipped references and the user guide's table of",
        "     references. Do not edit by hand. -->",
        "",
        f"These {len(REFERENCES)} starting points are rendered by the code that ships. Each sound is the reference's",
        "own take, matched in loudness at −27 LUFS like every reference, so turn them up together;",
        "the report is what `halyard render` prints for it, and the patch is what `halyard init`",
        "writes. The figures show the level over time, in dB below the peak, and the whole",
        "hit's spectrum, smoothed over a sixth of an octave.",
        "",
        *(f"- [`{name}`](#{name})" for name in REFERENCES),
    ]
    for name in REFERENCES:
        what, tries = found[name]
        text = reference_text(name)
        p = patch.loads(text)
        lines += [
            "",
            f"## {name}",
            "",
            f"{what}. Try: {tries}.",
            "",
            f"![The {name} reference: its level over time and its spectrum](figures/{name}.svg)",
            "",
            f"[Listen to {name}](audio/{name}.wav) · `halyard init {name} > {name}.toml`",
            "",
            "```",
            report_text(p),
            "```",
            "",
            "<details><summary>The patch</summary>",
            "",
            "```toml",
            text.rstrip("\n"),
            "```",
            "",
            "</details>",
        ]
    return "\n".join(lines) + "\n"


# Building.


def sound(name: str, path: Path) -> None:
    """Write a reference's own take as 24-bit PCM."""
    p = patch.loads(reference_text(name))
    p = dataclasses.replace(p, output=dataclasses.replace(p.output, format=DOCS_FORMAT))
    audio, report = render(p)
    take = report.takes[0]
    write_take(path, audio, p, Stamp(take.seed, take.number, len(report.takes), report.set_gain_db, take.gain_db))


def outputs() -> dict[Path, Callable[[Path], None]]:
    """Return every generated file and how to write it."""
    out: dict[Path, Callable[[Path], None]] = {}
    for name in REFERENCES:
        p = patch.loads(reference_text(name))

        def figure(path: Path, name: str = name, p: patch.Patch = p) -> None:
            path.write_text(reference_figure(name, _audio(p), p.output.sample_rate), encoding="utf-8")

        def audio(path: Path, name: str = name) -> None:
            sound(name, path)

        out[FIGURES / f"{name}.svg"] = figure
        out[AUDIO / f"{name}.wav"] = audio
    concepts: dict[str, str] = {}

    def concept(path: Path) -> None:
        if not concepts:  # rendered once for all three
            concepts.update(concept_figures())
        path.write_text(concepts[path.stem], encoding="utf-8")

    for name in CONCEPTS:
        out[FIGURES / f"{name}.svg"] = concept

    def page(path: Path) -> None:
        path.write_text(gallery(), encoding="utf-8")

    out[GALLERY] = page
    return out


def build(root: Path | None = None) -> list[Path]:
    """Write every generated file, under `root` instead of docs/ if given; return what was written."""
    written = []
    for path, write in outputs().items():
        target = path if root is None else root / path.relative_to(DOCS)
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target)
        written.append(target)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Write the generated documentation, or with --check list what is out of date and fail if any is."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    args = ap.parse_args(argv)
    if not args.check:
        for path in build():
            say(str(path.relative_to(DOCS.parent)))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        stale = [path for path in build(Path(tmp)) if not current(path, DOCS / path.relative_to(tmp))]
    for path in stale:
        say(f"out of date: {path.relative_to(tmp)}")
    return 1 if stale else 0


def current(fresh: Path, recorded: Path) -> bool:
    """Whether a recorded file matches a fresh build: text exactly, sounds to 2 LSB of 24 bits.

    Text is compared exactly: a figure rounds differently on another platform
    only if it lies within about 10⁻¹² of a rounding boundary.
    """
    if not recorded.exists():
        return False
    if fresh.suffix == ".wav":
        a, b = wav.read(fresh), wav.read(recorded)
        return (
            a.rate == b.rate
            and a.samples.size == b.samples.size
            and bool(
                np.max(np.abs(a.samples - b.samples)) <= 2 * 2.0**-23,
            )
        )
    return fresh.read_text(encoding="utf-8") == recorded.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
