# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check a pack as two SFZ players play it: sfizz and liquidsfz (plan §7.3).

Three steps, so the players run where they are built, not from Python:

    uv run python -m tools.sfz_players prepare DIR   # a pack of three layers × three takes, and the notes to play
    tools/sfz_players.sh DIR                         # build both players in a throwaway container, render the notes
    uv run python -m tools.sfz_players check DIR     # what each player played, against the pack

Every note is identified by the file it played, the one it matches best piece
by piece, and measured by its gain against that file below 2 kHz. A player passes when every
note plays the layer its velocity selects, the round robins follow in order,
and every note's gain is the group's volume times its amplitude curve at that
velocity, up to one gain for the whole player, within 0.05 dB: velocity then
follows the square law across the layers' boundaries.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import soundfile

from halyard_synth.cli import reference_text
from halyard_synth.pack import Options, pack
from halyard_synth.patch import dumps, loads, softer
from tools import say

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]

RATE = 48000
SPACING_S = 0.5  # from one note to the next
LENGTH_S = 0.01  # each note's length: as short as a programmed hit, so a player that cut it would fail
VELOCITIES = (127, 110, 96, 80, 64, 48, 32, 20)  # every layer, each layer's top and bottom nearby
REPEATS = 3  # each velocity played as often as there are takes, so the round robins cycle
TAKES = 3
LAYERS = (0, 12, 24)  # the crack reference struck this much softer
TOLERANCE_DB = 0.05
PLAYERS = ("sfizz", "liquidsfz")


class Note(NamedTuple):
    """A note: when it starts, in frames, and its velocity."""

    frame: int
    velocity: int


def notes() -> list[Note]:
    """Return the notes to play: every velocity, each as often as there are takes."""
    out = []
    for i, v in enumerate(v for v in VELOCITIES for _ in range(REPEATS)):
        out.append(Note(round(i * SPACING_S * RATE), v))
    return out


def vlq(n: int) -> bytes:
    """Return a MIDI variable-length quantity."""
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def midi_file(played: Sequence[Note], key: int) -> bytes:
    """Return a standard MIDI file of the notes at 120 bpm, 480 ticks a beat: 960 ticks a second."""
    ticks = 960
    events: list[tuple[int, bytes]] = []
    for n in played:
        start = round(n.frame / RATE * ticks)
        events.append((start, bytes((0x90, key, n.velocity))))
        events.append((start + round(LENGTH_S * ticks), bytes((0x80, key, 0))))
    events.sort(key=lambda e: e[0])
    track, now = bytearray(), 0
    for t, data in events:
        track += vlq(t - now) + data
        now = t
    track += vlq(0) + b"\xff\x2f\x00"
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)


def prepare(folder: Path) -> None:
    """Write the pack, its notes as a MIDI file for sfizz and as events for liquidsfz, and what to expect."""
    folder.mkdir(parents=True, exist_ok=True)
    crack = loads(reference_text("crack"))
    crack = dataclasses.replace(crack, output=dataclasses.replace(crack.output, db=-6.0))  # loud, never over
    paths = []
    for db in LAYERS:
        path = folder / f"crack-{db}.toml"
        path.write_text(dumps(softer(crack, db)), encoding="utf-8")
        paths.append(str(path))
    with (folder / "pack.txt").open("w", encoding="utf-8") as report:
        code = pack(paths, folder / "kit", Options(takes=TAKES), out=report)
    if code:
        msg = f"the pack failed: {(folder / 'pack.txt').read_text(encoding='utf-8')}"
        raise RuntimeError(msg)
    key = read_sfz(sfz_of(folder)).key
    played = notes()
    (folder / "notes.mid").write_bytes(midi_file(played, key))
    lines = []
    for n in played:
        lines += [f"{n.frame} {key} {n.velocity}", f"{n.frame + round(LENGTH_S * RATE)} {key} 0"]
    (folder / "notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frames = played[-1].frame + round(SPACING_S * RATE)
    (folder / "frames.txt").write_text(f"{frames}\n", encoding="utf-8")
    say(f"wrote a pack of {len(LAYERS)} layers × {TAKES} takes and {len(played)} notes in {folder}")


@dataclasses.dataclass(frozen=True)
class Group:
    """A velocity layer as the SFZ maps it: its velocities, volume, amplitude curve and samples by position."""

    lovel: int
    hivel: int
    volume_db: float
    curve: dict[int, float]
    samples: dict[int, str]  # by seq_position

    def gain(self, velocity: int) -> float:
        """Return the amplitude the mapping asks for at a velocity: the volume times the curve."""
        return 10 ** (self.volume_db / 20) * self.curve[velocity]


class Mapping(NamedTuple):
    """The SFZ file's key and groups."""

    key: int
    groups: list[Group]

    def group(self, velocity: int) -> Group:
        """Return the group a velocity selects."""
        return next(g for g in self.groups if g.lovel <= velocity <= g.hivel)


def sfz_of(folder: Path) -> Path:
    """Return the pack's SFZ file."""
    return next((folder / "kit").glob("*.sfz"))


def read_sfz(path: Path) -> Mapping:
    """Read the mapping halyard pack writes: a global key, then groups of velocities, volume, curve and regions."""
    key = 0
    opcodes: list[dict[str, str]] = []
    curves: list[dict[int, float]] = []
    samples: list[dict[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("//")[0].strip()
        if line == "<group>":
            opcodes.append({})
            curves.append({})
            samples.append({})
        elif line.startswith("key="):
            key = int(line.split("=")[1])
        elif opcodes and line.startswith("<region>"):
            fields = dict(part.split("=") for part in line.removeprefix("<region>").split())
            samples[-1][int(fields["seq_position"])] = fields["sample"]
        elif opcodes and "=" in line:
            name, value = line.split("=")
            if name.startswith("amp_velcurve_"):
                curves[-1][int(name.removeprefix("amp_velcurve_"))] = float(value)
            else:
                opcodes[-1][name] = value
    groups = [
        Group(int(o["lovel"]), int(o["hivel"]), float(o["volume"]), c, r)
        for o, c, r in zip(opcodes, curves, samples, strict=True)
    ]
    return Mapping(key, groups)


class Heard(NamedTuple):
    """What a player played for one note: the file it matches best, how well, and its gain against it."""

    sample: str
    match: float  # the normalized correlation, 1 for a perfect match
    gain: float


LAG = 256  # how late a player may start a note, in frames
WINDOW = 960  # a note's delay is found from its first 20 ms
LOW_HZ = 2000.0  # and its gain measured below this, where both players' interpolators are flat


def lowpassed(x: Floats, frames: int) -> Floats:
    """Return the first frames of x with everything from LOW_HZ up removed."""
    n = 1 << math.ceil(math.log2(4 * frames))
    spectrum = np.fft.rfft(x[:frames], n)
    spectrum[np.fft.rfftfreq(n, 1 / RATE) >= LOW_HZ] = 0
    out: Floats = np.fft.irfft(spectrum, n)
    return out


PIECE = 64  # frames: a note is compared piece by piece, each allowed a frame either way


def matched(y: Floats, start: int, x: Floats) -> float:
    """Return how well a note matches a file: the median over pieces of each piece's best correlation.

    sfizz streams a sample's remainder after a preloaded start and, at the seam,
    plays one frame twice or skips one (VALIDATION.md), so the note matches the
    file at one delay before the seam and at the next after it. Each piece may
    therefore be a frame early or late, and the median ignores the piece the
    seam falls in.
    """
    scores = []
    for a in range(0, x.size - PIECE, PIECE):
        piece = x[a : a + PIECE]
        norm = np.linalg.norm(piece)
        if norm < SILENT * np.linalg.norm(x) / math.sqrt(x.size / PIECE):
            continue
        best = -1.0
        for shift in (-1, 0, 1):
            b = start + a + shift
            if b < 0:
                continue
            seg = y[b : b + PIECE]
            if seg.size == PIECE:
                best = max(best, float(np.dot(seg, piece) / max(np.linalg.norm(seg) * norm, 1e-30)))
        scores.append(best)
    return float(np.median(scores))


SILENT = 0.01  # a piece this far below the file's average is too quiet to judge


def onset(y: Floats, start: int, x: Floats) -> int:
    """Return the delay, in frames up to LAG, at which a note's first 20 ms best match a file's."""
    head = x[:WINDOW]
    seg = y[start : start + WINDOW + LAG]
    corr = np.correlate(seg, head, mode="valid")
    energy = np.sqrt(np.convolve(seg**2, np.ones(WINDOW), mode="valid")) * np.linalg.norm(head)
    return int(np.argmax(corr / np.maximum(energy, 1e-30)))


def identify(y: Floats, start: int, files: dict[str, Floats]) -> Heard:
    """Return the file a note played, the one it matches best piece by piece, and its gain against it.

    The gain is measured over the whole file below LOW_HZ, where both players
    are flat and a frame's slip changes nothing measurable, so that a player's
    16-bit output averages out even on the softest notes.
    """
    best, at = Heard("", -1.0, 0.0), 0
    for name, x in files.items():
        lag = onset(y, start, x)
        match = matched(y, start + lag, x)
        if match > best.match:
            best, at = Heard(name, match, 0.0), lag
    size = files[best.sample].size
    x, played = lowpassed(files[best.sample], size), lowpassed(y[start + at :], size)
    return best._replace(gain=float(np.dot(played, x) / np.dot(x, x)))


def check_player(name: str, y: Floats, mapping: Mapping, files: dict[str, Floats]) -> bool:
    """Say what a player played, and return whether it played the pack as mapped."""
    played = notes()
    heard = [identify(y, n.frame, files) for n in played]
    ok = True
    ratios = []
    last: dict[int, int] = {}
    for n, h in zip(played, heard, strict=True):
        group = mapping.group(n.velocity)
        index = mapping.groups.index(group)
        position = next((p for p, s in group.samples.items() if s == h.sample), None)
        if position is None:
            say(f"  {name}: velocity {n.velocity} played {h.sample}, not one of its layer's samples")
            ok = False
            continue
        if index in last and position != last[index] % len(group.samples) + 1:
            say(f"  {name}: velocity {n.velocity} played round robin {position} after {last[index]}")
            ok = False
        last[index] = position
        ratios.append(20 * math.log10(h.gain / group.gain(n.velocity)))
    worst = min(h.match for h in heard)
    spread = max(ratios) - min(ratios) if ratios else math.inf
    say(
        f"  {name}: {len(played)} notes, the worst waveform match {worst:.6f}; "
        f"gain against the mapping {np.mean(ratios):+.3f} dB, spread {spread:.4f} dB",
    )
    return ok and spread <= TOLERANCE_DB


def mono(path: Path) -> Floats:
    """Read a pack's file or a player's output, its first channel; players write stereo, in their own formats."""
    x: Floats = np.asarray(soundfile.read(str(path), dtype="float64", always_2d=True)[0][:, 0])
    return x


def check(folder: Path) -> int:
    """Check every player's rendering of the notes; return 1 if any played the pack otherwise than mapped."""
    mapping = read_sfz(sfz_of(folder))
    files = {name: mono(folder / "kit" / name) for g in mapping.groups for name in g.samples.values()}
    found = [p for p in PLAYERS if (folder / f"{p}.wav").exists()]
    if not found:
        say(f"no player's rendering in {folder}: run tools/sfz_players.sh {folder}")
        return 1
    results = {p: check_player(p, mono(folder / f"{p}.wav"), mapping, files) for p in found}
    say(json.dumps(results))
    return 0 if all(results.values()) and len(found) == len(PLAYERS) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare a pack and its notes, or check the players' renderings of them."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prepare", "check"))
    ap.add_argument("folder", type=Path)
    args = ap.parse_args(argv)
    if args.step == "prepare":
        prepare(args.folder)
        return 0
    return check(args.folder)


if __name__ == "__main__":
    sys.exit(main())
