"""Command line (plan §7.6): halyard init | render | verify | extract | audition | pack, and the editing commands
tune | stretch | bursts | softer, which print a complete, canonical patch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import resources
from pathlib import Path

from . import patch as P
from .patch import PatchError, dumps, loads

REFERENCES = ("tone", "crack", "fat", "driven", "clap", "snap")


def reference_text(name: str) -> str:
    return resources.files("halyard_synth").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise SystemExit(_fail(f"{path}: {e.strerror}"))


def _fail(message: str, code: int = 2) -> int:
    print(message, file=sys.stderr)
    return code


def _patch(path: str, text: str | None = None) -> P.Patch:
    try:
        return loads(_read(path) if text is None else text)
    except PatchError as e:
        raise SystemExit(_fail(f"{path}: invalid patch\n" + "\n".join(f"  {m}" for m in e.problems)))


def _edit(path: str, what: str, fn, *args, **kwargs) -> int:
    """Apply an editing command and print the patch, or say why it cannot be done."""
    p = _patch(path)
    try:
        q = fn(p, *args, **kwargs)
    except (ValueError, PatchError) as e:
        problems = e.problems if isinstance(e, PatchError) else [str(e)]
        return _fail(f"{path}: cannot {what}\n" + "\n".join(f"  {m}" for m in problems))
    sys.stdout.write(dumps(q))
    return 0


def cmd_init(args) -> int:
    p = loads(reference_text(args.reference))
    if args.tune is not None:
        try:
            p = P.tune(p, args.tune)
        except (ValueError, PatchError) as e:
            problems = e.problems if isinstance(e, PatchError) else [str(e)]
            return _fail(f"--tune {args.tune}: cannot tune {args.reference}\n" + "\n".join(f"  {m}" for m in problems))
    sys.stdout.write(dumps(p))
    return 0


def cmd_tune(args) -> int:
    return _edit(args.patch, f"tune to {args.note}", P.tune, args.note)


def cmd_stretch(args) -> int:
    return _edit(
        args.patch,
        f"stretch {' '.join(args.curves)} by {args.factor:g}",
        P.stretch,
        args.curves,
        args.factor,
        args.after,
    )


def cmd_bursts(args) -> int:
    return _edit(
        args.patch,
        f"write bursts on {args.layer}",
        P.bursts,
        args.layer,
        args.count,
        args.spacing,
        start=args.start,
        growth=args.growth,
        rise=args.rise,
        dip=args.dip,
        taper=args.taper,
        tail=args.tail,
    )


def cmd_softer(args) -> int:
    return _edit(args.patch, f"make it {args.db:g} dB softer", P.softer, args.db, args.glide, args.contact)


def take_paths(out: str, takes: int) -> list[Path]:
    """`snare.wav` for one take; `snare_rr1.wav` … for several."""
    path = Path(out)
    if takes == 1:
        return [path]
    return [path.with_name(f"{path.stem}_rr{j}{path.suffix or '.wav'}") for j in range(1, takes + 1)]


def cmd_render(args) -> int:
    from .render import FORMATS, RenderError, db, export, render_takes

    p = _patch(args.patch)
    if args.allow_over and FORMATS[p.output.format] is not None:
        return _fail("--allow-over writes float32 only: PCM cannot hold samples beyond full scale")
    try:
        audio, report = render_takes(p, args.takes, args.level_match)
    except RenderError as e:
        return _fail(f"{args.patch}: {e}")
    if not report.over or (args.allow_over and FORMATS[p.output.format] is None):
        # Every file first, then the report: a closed output never leaves a set half written.
        paths = take_paths(args.out, args.takes)
        for path, a, take in zip(paths, audio, report.takes):
            export(path, a, p, report, take)
        print(report)
        for path in paths:
            print(f"wrote {path}")
        return 0
    print(report)
    if FORMATS[p.output.format] is not None:
        return _fail(
            f"not written: the peak is {db(report.peak):+.4f} dBFS, above {p.output.format}'s ceiling of "
            f"{db(report.ceiling):.4f} dBFS, which keeps 2 LSB for dither. Lower output.db by at least "
            f"{db(report.peak / report.ceiling):.4f} dB.",
            1,
        )
    return _fail(
        f"not written: the peak is {db(report.peak):+.2f} dBFS. Lower output.db by at least "
        f"{db(report.peak):.2f} dB, or pass --allow-over to write a float file with overs.",
        1,
    )


def cmd_verify(args) -> int:
    from .reference import verify

    p = _patch(args.patch)
    v = verify(p, takes=args.takes)
    print(v)
    return 0 if v.passed else 1


def cmd_extract(args) -> int:
    from . import wav

    try:
        found = wav.chunks(args.wav)
    except (OSError, ValueError) as e:
        return _fail(f"{args.wav}: {e}")
    if wav.PRIVATE not in found:
        return _fail(f"{args.wav}: no embedded snare patch")
    sys.stdout.write(json.loads(found[wav.PRIVATE].decode("utf-8"))["patch"])
    return 0


def cmd_audition(args) -> int:
    from .audition import audition

    return audition(
        args.patches,
        watch=args.watch,
        bpm=args.bpm,
        beats=args.beats,
        bed=args.bed,
        against=args.against,
        blind=args.blind,
        takes=args.takes,
    )


def cmd_pack(args) -> int:
    from .pack import pack

    return pack(args.patches, args.out, takes=args.takes, key=args.key, level_match=args.level_match)


def _beats(text: str) -> tuple[int, ...]:
    try:
        beats = tuple(sorted({int(b) for b in text.split(",")}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected beats such as 2,4, found {text!r}") from None
    if not beats or beats[0] < 1 or beats[-1] > 4:
        raise argparse.ArgumentTypeError("beats of a 4/4 bar are 1 to 4")
    return beats


def _finite(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, found {text!r}") from None
    if not math.isfinite(x):
        raise argparse.ArgumentTypeError(f"expected a finite number, found {text!r}")
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="halyard", description="A one-voice snare and clap synthesizer.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="print a complete starting patch")
    p.add_argument("reference", choices=REFERENCES)
    p.add_argument("--tune", help="a note name such as D3 or F#3, or a frequency in Hz")
    p.set_defaults(run=cmd_init)

    p = sub.add_parser("render", help="render a patch, or a set of round-robin takes, to WAV")
    p.add_argument("patch")
    p.add_argument("--out", required=True, help="the file; several takes are written as <stem>_rr<take>.wav")
    p.add_argument("--takes", type=int, default=1, help="round-robin takes: seeds seed, seed + 1, … (default 1)")
    p.add_argument("--level-match", action="store_true", help="scale each take to the loudness of the first")
    p.add_argument(
        "--allow-over", action="store_true", help="write a gain-mode render that exceeds full scale (float32)"
    )
    p.set_defaults(run=cmd_render)

    p = sub.add_parser("verify", help="compare a patch, or its takes, with the reference render")
    p.add_argument("patch")
    p.add_argument("--takes", type=int, default=1)
    p.set_defaults(run=cmd_verify)

    p = sub.add_parser("extract", help="print the patch embedded in an exported WAV")
    p.add_argument("wav")
    p.set_defaults(run=cmd_extract)

    p = sub.add_parser("tune", help="print the patch retuned to a note or frequency")
    p.add_argument("patch")
    p.add_argument("note", help="a note name such as D3 or F#3, or a frequency in Hz")
    p.set_defaults(run=cmd_tune)

    p = sub.add_parser("stretch", help="print the patch with curves' knot times rescaled after their peaks")
    p.add_argument("patch")
    p.add_argument(
        "curves", nargs="+", metavar="CURVE", help="body.amp, body.pitch, noise[N].amp, or all amplitude shapes"
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

    p = sub.add_parser("audition", help="play patches in a loop, on the backbeat, over a bed or against a sample")
    p.add_argument("patches", nargs="+", metavar="PATCH")
    p.add_argument("--watch", action="store_true", help="re-render whenever a file changes")
    p.add_argument("--bpm", type=_finite, help="hit on the beats of 4/4 bars at this tempo")
    p.add_argument("--beats", type=_beats, default=(2, 4), help="the beats to hit, such as 2,4 (default) or 1,2,3,4")
    p.add_argument(
        "--with", dest="bed", metavar="WAV", help="loop a bed under the hits: a whole number of bars at --bpm"
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

    args = ap.parse_args(argv)
    if getattr(args, "bpm", None) is not None and not args.bpm > 0:
        ap.error("--bpm must be positive")
    try:
        return args.run(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except BrokenPipeError:  # the reader went away, as `halyard render … | head` does: the files are written
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
