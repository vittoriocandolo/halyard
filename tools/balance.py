# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The balance against Ballast's kicks that the user guide gives (docs/guide.md: In a DAW), measured.

Each preset is set against one of Ballast's presets in a loop of eight bars of 4/4: the kick on every beat, at
`output.db = −10`, from the WAVs in tools/balance/ that Ballast rendered (each holds its patch); the preset on two
and four. Each part's loudness is its largest 400 ms momentary loudness (ITU-R BS.1770), as `halyard render`
reports it. A preset is raised until it reads 5 dB under the kick, but never so far that its file peaks above
−1 dBFS.

    uv run python -m tools.balance          # print the table
    uv run python -m tools.balance --write  # write it into docs/guide.md
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from docs.build import markdown
from halyard_synth import cli, patch, wav
from halyard_synth.loudness import momentary_max
from halyard_synth.render import db, render

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]

ROOT = Path(__file__).resolve().parent.parent
KICKS = ROOT / "tools" / "balance"
GUIDE = ROOT / "docs" / "guide.md"
TEMPOS_BPM = {"house": 124.0, "tech-house": 126.0, "trance": 138.0}  # each kick's tempo
KICK_DB = -10.0  # the kicks' output.db
BARS = 8
BEATS = (2, 4)  # where a snare or clap plays in each bar
UNDER_DB = 5.0  # how far under the kick a snare or clap reads
PEAK_DBFS = -1.0  # no file is raised past this
AGAINST = {  # the Ballast preset each preset is set against
    "tone": "house",
    "crack": "house",
    "fat": "house",
    "driven": "house",
    "clap": "house",
    "snap": "house",
    "machine": "house",
    "house": "house",
    "short-clap": "tech-house",
    "rim-click": "tech-house",
    "big-clap": "trance",
    "long-snare": "trance",
    "rimshot": "house",
    "cross-stick": "house",
    "brush": "house",
    "zap": "trance",
    "reverse": "trance",
    "closing": "house",
}
HEADER = (
    "| Preset | Against Ballast's | Raise `output.db` by | Note |",
    "|---|---|---|---|",
)


class Row(NamedTuple):
    """One preset's balance: the kick it is set against, how far to raise it, and how far its peak holds it short."""

    preset: str
    kick: str
    raise_db: float
    short_db: float  # 0 when the peak does not hold it


def loop(hit: Floats, rate: int, bpm: float, beats: Sequence[float]) -> Floats:
    """Return BARS bars of the hit, struck at each of `beats` (counted from 0) of the whole loop."""
    beat = 60 / bpm * rate
    out = np.zeros(round(BARS * 4 * beat) + hit.size)
    for at in beats:
        start = round(at * beat)
        out[start : start + hit.size] += hit
    return out[: round(BARS * 4 * beat)]


def ballast_patch(path: Path) -> dict[str, object]:
    """Return the patch a WAV that Ballast wrote holds, from its `blst` chunk."""
    data = path.read_bytes()
    at = 12
    while at + 8 <= len(data):
        kind, size = data[at : at + 4], struct.unpack_from("<I", data, at + 4)[0]
        if kind == b"blst":
            return tomllib.loads(json.loads(data[at + 8 : at + 8 + size])["patch"])
        at += 8 + size + (size & 1)
    msg = f"{path}: no patch from Ballast in it"
    raise ValueError(msg)


def kick(name: str) -> tuple[Floats, int]:
    """Return a Ballast kick as tools/balance/ holds it, and its rate; refuse one not at KICK_DB."""
    path = KICKS / f"{name}.wav"
    output = ballast_patch(path)["output"]
    if not isinstance(output, dict) or output.get("db") != KICK_DB:
        msg = f"{path}: its kick is not at output.db = {KICK_DB:g}; render it again with Ballast"
        raise ValueError(msg)
    w = wav.read(path)
    return w.as_float(), w.rate


def rows() -> list[Row]:
    """Measure every preset's balance, in the order the presets ship."""
    if set(AGAINST) != set(cli.PRESETS):
        msg = f"AGAINST must name every preset and no other: {sorted(set(AGAINST) ^ set(cli.PRESETS))}"
        raise ValueError(msg)
    loudest: dict[str, float] = {}
    for name, bpm in TEMPOS_BPM.items():
        audio, rate = kick(name)
        loudest[name] = momentary_max(loop(audio, rate, bpm, range(BARS * 4)), rate)
    out = []
    for preset in cli.PRESETS:
        name = AGAINST[preset]
        rendered, report = render(patch.loads(cli.preset_text(preset)))
        x, rate = rendered.astype(float), report.sample_rate
        on = [bar * 4 + b - 1 for bar in range(BARS) for b in BEATS]
        need = loudest[name] - UNDER_DB - momentary_max(loop(x, rate, TEMPOS_BPM[name], on), rate)
        room = PEAK_DBFS - db(float(np.max(np.abs(x))))
        out.append(Row(preset, name, min(need, room), max(0.0, need - room)))
    return out


def signed(x: float) -> str:
    """Write a level change with its sign, and a true minus."""
    return f"{x:+.1f}".replace("-", "−")


def table(measured: Sequence[Row]) -> str:
    """Write the guide's table."""
    lines = list(HEADER)
    for r in measured:
        note = f" {r.short_db:.1f} dB short, held at −1 dBFS by its peak " if r.short_db > 0 else " "
        lines.append(f"| `{r.preset}` | `{r.kick}` | {signed(r.raise_db)} dB |{note}|")
    return markdown("\n".join(lines) + "\n")  # as mdformat writes it into the guide


def in_guide(text: str) -> tuple[int, int]:
    """Return where the guide's table starts and ends, in characters."""
    start = text.index(HEADER[0])
    end = start
    for line in text[start:].splitlines(keepends=True):
        if not line.startswith("|"):
            break
        end += len(line)
    return start, end


def main(argv: Sequence[str] | None = None) -> int:
    """Print the table, or write it into the guide."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the table into docs/guide.md")
    args = ap.parse_args(argv)
    text = table(rows())
    if not args.write:
        sys.stdout.write(text)
        return 0
    guide = GUIDE.read_text(encoding="utf-8")
    start, end = in_guide(guide)
    GUIDE.write_text(guide[:start] + text + guide[end:], encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
