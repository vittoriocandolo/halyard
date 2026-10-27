"""The six references by measurement (plan §3, §8.4): seeds 1–16 at a candidate K.

For each reference it reports the loudness and sample peak of every take,
whether each take is inside the validated domain and its drive headroom, and
the loudness spread of takes 1–8. The acceptance criteria are:

  * inside the validated domain with at least 3 dB of drive headroom, seeds 1–16;
  * loudness matched within 1 dB across the references, peaks below −1 dBFS;
  * a loudness spread of at most 1 dB over takes 1–8, with level matching where flagged.

    python tools/references.py [--k 32] [--seeds 16]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from halyard_synth.cli import REFERENCES, reference_text  # noqa: E402
from halyard_synth.loudness import momentary_max  # noqa: E402
from halyard_synth.patch import loads  # noqa: E402
from halyard_synth.render import K, cause_of, db, domain, drive_headroom, output_stage, synthesize  # noqa: E402


def measure(name: str, k: int, seeds: int) -> dict:
    p = loads(reference_text(name))
    gain = 10 ** (p.output.db / 20)
    rows = []
    for seed in range(1, seeds + 1):
        q = dataclasses.replace(p, seed=seed)
        s = synthesize(q, k=k)
        d = domain(s)
        h = drive_headroom(s, q.drive.db, d)
        a = output_stage(s.out, gain, 1)
        rows.append(
            {
                "seed": seed,
                "lufs": momentary_max(a, q.output.sample_rate),
                "peak": db(float(max(abs(a.min()), a.max()))),
                "inside": d.inside,
                "margin": d.margin_db,
                "headroom": h.db if h else math.inf,
                "cause": cause_of(q, s, d).dominant,
            }
        )
    eight = [r["lufs"] for r in rows[:8]]
    return {
        "name": name,
        "rows": rows,
        "spread": max(eight) - min(eight),
        "lufs": rows[0]["lufs"],
        "peak": max(r["peak"] for r in rows),
        "headroom": min(r["headroom"] for r in rows),
        "inside": all(r["inside"] for r in rows),
        "margin": min(r["margin"] for r in rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args(argv)
    results = [measure(name, args.k, args.seeds) for name in REFERENCES]
    print(f"K = {args.k}, seeds 1–{args.seeds}")
    print(
        f"{'reference':<9s} {'LUFS':>6s} {'spread 1–8':>10s} {'max peak':>9s} {'inside':>7s} {'min margin':>11s} {'min headroom':>13s}"
    )
    for r in results:
        print(
            f"{r['name']:<9s} {r['lufs']:6.1f} {r['spread']:8.2f}dB {r['peak']:7.2f}dB {str(r['inside']):>7s} "
            f"{r['margin']:9.1f}dB {r['headroom']:11.1f}dB"
        )
    lufs = [r["lufs"] for r in results]
    ok = (
        max(lufs) - min(lufs) <= 1.0
        and all(r["peak"] < -1.0 for r in results)
        and all(r["inside"] and r["headroom"] >= 3.0 for r in results)
        and all(r["spread"] <= 1.0 for r in results)
    )
    print(f"loudness range {max(lufs) - min(lufs):.2f} dB; {'all criteria met' if ok else 'criteria NOT met'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
