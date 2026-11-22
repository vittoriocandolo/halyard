# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command line (docs/guide.md): halyard init | render | verify | extract | audition | pack.

The editing commands, tune | stretch | bursts | softer | flam | roll | stack, print a complete, canonical patch.
"""

from __future__ import annotations

import argparse
import math
import sys
import unicodedata
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, wav
from .audition import Settings, audition
from .pack import Options, pack
from .patch import Patch, PatchError, Roll, Train, bursts, dumps, flam, load, loads, roll, softer, stack, stretch, tune
from .reference import verify
from .render import FORMATS, RenderError, db, export, render_takes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PRESETS = (
    "tone",
    "crack",
    "fat",
    "driven",
    "clap",
    "snap",
    "machine",
    "house",
    "short-clap",
    "rim-click",
    "big-clap",
    "long-snare",
    "rimshot",
    "cross-stick",
    "brush",
    "zap",
    "reverse",
    "closing",
)
BEATS_PER_BAR = 4


def preset_text(name: str) -> str:
    """Return a preset's text, as it ships."""
    return resources.files("halyard_synth").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")


class CliError(Exception):
    """A message for stderr, and the exit status: 2 for bad input, 1 for a refusal or a failure."""

    def __init__(self, message: str, status: int = 2) -> None:
        """Keep the message, and the status to exit with."""
        super().__init__(message)
        self.status = status


def _listing(head: str, problems: Sequence[str]) -> CliError:
    """Make an error of `head`, then every problem, indented."""
    return CliError("\n".join([head, *(f"  {m}" for m in problems)]))


def _patch(path: str) -> Patch:
    """Read a patch file."""
    try:
        return load(path)
    except OSError as e:
        msg = f"{path}: {e.strerror}"
        raise CliError(msg) from None
    except PatchError as e:
        msg = f"{path}: invalid patch"
        raise _listing(msg, e.problems) from None


def _edit(path: str, what: str, edit: Callable[..., Patch], *args: object) -> int:
    """Apply an editing command and print the patch, or say why it cannot be done."""
    p = _patch(path)
    try:
        q = edit(p, *args)
    except (ValueError, PatchError) as e:
        problems = e.problems if isinstance(e, PatchError) else [str(e)]
        msg = f"{path}: cannot {what}"
        raise _listing(msg, problems) from None
    sys.stdout.write(dumps(q))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Print a preset, retuned when asked."""
    p = loads(preset_text(args.preset))
    if args.tune is not None:
        try:
            p = tune(p, args.tune)
        except (ValueError, PatchError) as e:
            problems = e.problems if isinstance(e, PatchError) else [str(e)]
            msg = f"--tune {args.tune}: cannot tune {args.preset}"
            raise _listing(msg, problems) from None
    sys.stdout.write(dumps(p))
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Print the patch retuned."""
    return _edit(args.patch, f"tune to {args.note}", tune, args.note)


def cmd_stretch(args: argparse.Namespace) -> int:
    """Print the patch with its curves stretched."""
    what = f"stretch {' '.join(args.curves)} by {args.factor:g}"
    return _edit(args.patch, what, stretch, args.curves, args.factor, args.after)


def cmd_bursts(args: argparse.Namespace) -> int:
    """Print the patch with a layer's shape rewritten as bursts."""
    train = Train(args.count, args.spacing, args.start, args.growth, args.rise, args.dip, args.taper, args.tail)
    return _edit(args.patch, f"write bursts on {args.layer}", bursts, args.layer, train)


def cmd_softer(args: argparse.Namespace) -> int:
    """Print the same drum struck softer."""
    return _edit(args.patch, f"make it {args.db:g} dB softer", softer, args.db, args.glide, args.contact)


def cmd_stack(args: argparse.Namespace) -> int:
    """Print one patch that sounds two: a snare and a clap, say."""
    return _edit(args.patch, f"stack {args.other} on it", stack, _patch(args.other), args.db)


def cmd_roll(args: argparse.Namespace) -> int:
    """Print the drum struck several times: a roll or a build."""
    r = Roll(args.count, args.spacing, args.growth, args.crescendo)
    return _edit(args.patch, f"write a roll of {args.count} strokes", roll, r, args.glide, args.contact)


def cmd_flam(args: argparse.Namespace) -> int:
    """Print the drum struck twice, a grace note before the main stroke."""
    return _edit(args.patch, f"write a flam {args.gap:g} ms wide", flam, args.gap, args.grace, args.glide, args.contact)


def take_paths(out: str, takes: int) -> list[Path]:
    """Return `snare.wav` for one take; `snare_rr1.wav` and so on for several."""
    path = Path(out)
    if takes == 1:
        return [path]
    return [path.with_name(f"{path.stem}_rr{j}{path.suffix or '.wav'}") for j in range(1, takes + 1)]


def cmd_render(args: argparse.Namespace) -> int:
    """Render a patch's takes, and write them unless they would be over."""
    p = _patch(args.patch)
    if args.allow_over and FORMATS[p.output.format] is not None:
        msg = "--allow-over writes float32 only: PCM cannot hold samples beyond full scale"
        raise CliError(msg)
    try:
        audio, report = render_takes(p, args.takes, level_match=args.level_match)
    except RenderError as e:
        msg = f"{args.patch}: {e}"
        raise CliError(msg) from None
    if not report.over or (args.allow_over and FORMATS[p.output.format] is None):
        # Every file first, then the report: a closed output never leaves a set half written.
        paths = take_paths(args.out, args.takes)
        for path, a, take in zip(paths, audio, report.takes, strict=True):
            try:
                export(path, a, p, report, take)
            except OSError as e:
                msg = f"{e.filename or path}: {e.strerror or e}"
                raise CliError(msg) from None
        sys.stdout.write(f"{report}\n" + "".join(f"wrote {path}\n" for path in paths))
        return 0
    sys.stdout.write(f"{report}\n")
    if FORMATS[p.output.format] is not None:
        msg = (
            f"not written: the peak is {db(report.peak):+.4f} dBFS, above {p.output.format}'s ceiling of "
            f"{db(report.ceiling):.4f} dBFS, which keeps 2 LSB for dither. Lower output.db by at least "
            f"{db(report.peak / report.ceiling):.4f} dB."
        )
        raise CliError(msg, 1)
    msg = (
        f"not written: the peak is {db(report.peak):+.2f} dBFS. Lower output.db by at least "
        f"{db(report.peak):.2f} dB, or pass --allow-over to write a float file with overs."
    )
    raise CliError(msg, 1)


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a patch's takes against the reference render."""
    try:
        v = verify(_patch(args.patch), takes=args.takes)
    except RenderError as e:
        msg = f"{args.patch}: {e}"
        raise CliError(msg) from None
    sys.stdout.write(f"{v}\n")
    return 0 if v.passed else 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Print the patch a WAV file embeds."""
    try:
        found = wav.chunks(args.wav)
    except OSError as e:
        msg = f"{args.wav}: {e.strerror or e}"
        raise CliError(msg) from None
    except ValueError as e:  # its message names the file
        raise CliError(str(e)) from None
    if wav.PRIVATE not in found:
        msg = f"{args.wav}: no embedded snare patch"
        raise CliError(msg)
    meta = wav.embedded(found[wav.PRIVATE])
    text = meta.get("patch") if meta is not None else None
    if not isinstance(text, str):
        msg = f"{args.wav}: the embedded patch is damaged"
        raise CliError(msg)
    if controls := _controls(text):
        msg = f"{args.wav}: the embedded patch holds control characters ({controls}), which no patch does; not printed"
        raise CliError(msg)
    sys.stdout.write(text)
    return 0


def _controls(text: str) -> str:
    """Return the control characters in `text` other than tab and line ends, as code points, or "" for none.

    TOML allows none of C0 in a patch, and a terminal acts on them and on C1
    rather than showing them, so text from a file holding them is never printed.
    """
    found = sorted({c for c in text if unicodedata.category(c) == "Cc" and c not in "\t\n\r"})
    return ", ".join(f"U+{ord(c):04X}" for c in found)


def cmd_audition(args: argparse.Namespace) -> int:
    """Play patches until interrupted."""
    settings = Settings(args.watch, args.bpm, args.beats, args.bed, args.against, args.blind, args.takes)
    return audition(args.patches, settings)


def cmd_pack(args: argparse.Namespace) -> int:
    """Render velocity layers with takes, and write them with their SFZ mapping."""
    return pack(args.patches, args.out, Options(args.takes, args.key, args.level_match))


def _beats(text: str) -> tuple[int, ...]:
    """Parse the beats of a 4/4 bar, such as 2,4."""
    try:
        beats = tuple(sorted({int(b) for b in text.split(",")}))
    except ValueError:
        msg = f"expected beats such as 2,4, found {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not beats or beats[0] < 1 or beats[-1] > BEATS_PER_BAR:
        msg = "beats of a 4/4 bar are 1 to 4"
        raise argparse.ArgumentTypeError(msg)
    return beats


def _finite(text: str) -> float:
    """Parse a finite number."""
    try:
        x = float(text)
    except ValueError:
        msg = f"expected a number, found {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not math.isfinite(x):
        msg = f"expected a finite number, found {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return x


def _render_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the commands that start, render, check and read patches."""
    p = sub.add_parser("init", help="print a complete starting patch")
    p.add_argument("preset", choices=PRESETS)
    p.add_argument("--tune", help="a note name such as D3 or F#3, or a frequency in Hz")
    p.set_defaults(run=cmd_init)

    p = sub.add_parser("render", help="render a patch, or a set of round-robin takes, to WAV")
    p.add_argument("patch")
    p.add_argument("--out", required=True, help="the file; several takes are written as <stem>_rr<take>.wav")
    p.add_argument("--takes", type=int, default=1, help="round-robin takes: seeds seed, seed + 1, … (default 1)")
    p.add_argument("--level-match", action="store_true", help="scale each take to the loudness of the first")
    p.add_argument(
        "--allow-over",
        action="store_true",
        help="write a gain-mode render that exceeds full scale (float32)",
    )
    p.set_defaults(run=cmd_render)

    p = sub.add_parser("verify", help="compare a patch, or its takes, with the reference render")
    p.add_argument("patch")
    p.add_argument("--takes", type=int, default=1)
    p.set_defaults(run=cmd_verify)

    p = sub.add_parser("extract", help="print the patch embedded in an exported WAV")
    p.add_argument("wav")
    p.set_defaults(run=cmd_extract)


def _edit_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the editing commands, which print a complete patch."""
    p = sub.add_parser("tune", help="print the patch retuned to a note or frequency")
    p.add_argument("patch")
    p.add_argument("note", help="a note name such as D3 or F#3, or a frequency in Hz")
    p.set_defaults(run=cmd_tune)

    p = sub.add_parser("stretch", help="print the patch with curves' knot times rescaled after their peaks")
    p.add_argument("patch")
    p.add_argument(
        "curves",
        nargs="+",
        metavar="CURVE",
        help="body.amp, body.pitch, noise[N].amp, or all amplitude shapes",
    )
    p.add_argument("factor", type=_finite, help="below 1 shortens, above 1 lengthens")
    p.add_argument(
        "--after",
        type=_finite,
        metavar="MS",
        help="keep knots up to this time (default: each shape's peak, and 0 ms for pitch)",
    )
    p.set_defaults(run=cmd_stretch)

    p = sub.add_parser("bursts", help="print the patch with a layer's shape rewritten as a clap train")
    p.add_argument("patch")
    p.add_argument("layer", help="the layer, such as noise[1]")
    p.add_argument("--count", type=int, required=True, help="bursts, 1 to 10")
    p.add_argument("--spacing", type=_finite, required=True, metavar="MS", help="from the first burst to the second")
    p.add_argument("--start", type=_finite, default=0.0, metavar="MS", help="when the first burst starts (default 0)")
    p.add_argument("--growth", type=_finite, default=1.0, help="each gap over the previous one (default 1)")
    p.add_argument("--rise", type=_finite, default=0.1, metavar="MS", help="each burst's rise (default 0.1)")
    p.add_argument(
        "--dip",
        type=_finite,
        default=0.1,
        metavar="GAIN",
        help="the gain between bursts, over the lower of the two peaks (default 0.1)",
    )
    p.add_argument("--taper", type=_finite, default=1.0, help="each peak over the previous one (default 1)")
    p.add_argument("--tail", type=_finite, default=80.0, metavar="MS", help="the last burst's decay to 0 (default 80)")
    p.set_defaults(run=cmd_bursts)

    p = sub.add_parser("softer", help="print the same drum struck DB softer")
    p.add_argument("patch")
    p.add_argument("db", type=_finite, metavar="DB")
    p.add_argument("--glide", type=_finite, default=2.0, help="the glide's exponent (default 2; 0 turns it off)")
    p.add_argument("--contact", type=_finite, default=0.2, help="the contact exponent (default 0.2; 0 turns it off)")
    p.set_defaults(run=cmd_softer)

    p = sub.add_parser("flam", help="print the drum struck twice: a softer grace note, then the main stroke")
    p.add_argument("patch")
    p.add_argument("--gap", type=_finite, required=True, metavar="MS", help="from the grace note to the main stroke")
    p.add_argument("--grace", type=_finite, default=12.0, metavar="DB", help="how much softer the grace (default 12)")
    p.add_argument("--glide", type=_finite, default=2.0, help="the grace's glide exponent, as softer's (default 2)")
    p.add_argument(
        "--contact",
        type=_finite,
        default=0.2,
        help="the grace's contact exponent, as softer's (default 0.2)",
    )
    p.set_defaults(run=cmd_flam)

    p = sub.add_parser("roll", help="print the drum struck several times: a roll or a build, as ordinary knots")
    p.add_argument("patch")
    p.add_argument("--count", type=int, required=True, help="strokes, 2 to 16, as the 32 knots of a shape allow")
    p.add_argument("--spacing", type=_finite, required=True, metavar="MS", help="from the first stroke to the second")
    p.add_argument(
        "--growth",
        type=_finite,
        default=1.0,
        help="each gap over the one before; below 1 speeds up (default 1)",
    )
    p.add_argument(
        "--crescendo",
        type=_finite,
        default=0.0,
        metavar="DB",
        help="how much softer the first stroke (default 0)",
    )
    p.add_argument(
        "--glide",
        type=_finite,
        default=2.0,
        help="the softer strokes' glide exponent, as softer's (default 2)",
    )
    p.add_argument("--contact", type=_finite, default=0.2, help="their contact exponent, as softer's (default 0.2)")
    p.set_defaults(run=cmd_roll)

    p = sub.add_parser("stack", help="print one patch that sounds two, such as a snare and a clap, through one drive")
    p.add_argument("patch", help="the patch whose body, drive, output and seed the stack keeps")
    p.add_argument("other", help="the patch whose sources join it")
    p.add_argument("--db", type=_finite, default=0.0, help="how much louder the other's sources (default 0)")
    p.set_defaults(run=cmd_stack)


def _play_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the commands that play patches and pack them for a sampler."""
    p = sub.add_parser("audition", help="play patches in a loop, on the backbeat, over a bed or against a sample")
    p.add_argument("patches", nargs="+", metavar="PATCH")
    p.add_argument("--watch", action="store_true", help="re-render whenever a file changes")
    p.add_argument("--bpm", type=_finite, help="hit on the beats of 4/4 bars at this tempo")
    p.add_argument("--beats", type=_beats, default=(2, 4), help="the beats to hit, such as 2,4 (default) or 1,2,3,4")
    p.add_argument(
        "--with",
        dest="bed",
        metavar="WAV",
        help="loop a bed under the hits: a whole number of bars at --bpm",
    )
    p.add_argument("--against", metavar="WAV", help="alternate with a sample, hit by hit, matched in loudness")
    p.add_argument("--blind", action="store_true", help="with --against, play each pair in random order")
    p.add_argument("--takes", type=int, default=1, help="cycle this many takes, hit by hit")
    p.set_defaults(run=cmd_audition)

    p = sub.add_parser("pack", help="render velocity layers with takes and an SFZ mapping")
    p.add_argument("patches", nargs="+", metavar="PATCH", help="one per velocity layer, loudest first")
    p.add_argument("--out", required=True, metavar="DIR")
    p.add_argument("--takes", type=int, default=1)
    p.add_argument("--key", type=int, default=None, help="the MIDI key (default 38, the GM snare; 39 for a clap)")
    p.add_argument("--level-match", action="store_true")
    p.set_defaults(run=cmd_pack)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line, and return the exit status."""
    ap = argparse.ArgumentParser(prog="halyard", description="A one-voice snare and clap synthesizer.")
    ap.add_argument("--version", action="version", version=f"halyard {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    _render_commands(sub)
    _edit_commands(sub)
    _play_commands(sub)
    args = ap.parse_args(argv)
    if getattr(args, "bpm", None) is not None and not args.bpm > 0:
        ap.error("--bpm must be positive")
    run: Callable[[argparse.Namespace], int] = args.run
    try:
        return run(args)
    except CliError as e:
        sys.stderr.write(f"{e}\n")
        return e.status
    except BrokenPipeError:  # the reader went away, as `halyard render … | head` does: the files are written
        sys.stderr.close()
        return 0
    except KeyboardInterrupt:  # Ctrl+C: stop quietly, with the status a shell expects
        return 130
