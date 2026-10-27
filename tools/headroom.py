# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The report's drive headroom against brute force, on the adversarial grid (plan §5, §8.1).

For every grid fixture, the headroom that the report computes from one
render is compared with the first drive at which full renders change the
validated-domain status. The brute force renders every candidate the report
considers (every 0.5 dB, and both sides of every jump of the knee slew),
from the fixture's own drive until the status changes, then bisects that
bracket with full renders. With --dense it also scans every 0.05 dB up to
the edge on fixtures whose edge is within 6 dB, to show that no status
change hides between the candidates.

    uv run python -m tools.headroom run [--out docs/grid/headroom.csv] [--dense]
    uv run python -m tools.headroom analyze docs/grid/headroom.csv

A run appends to its CSV and skips fixtures already there, so it can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from halyard_synth.patch import from_dict
from halyard_synth.render import DRIVE_MAX, HEADROOM_STEP_DB, _drives, domain, render, synthesize
from tools import grid

FIELDS = ["family", "name", "drive", "inside", "estimate", "reference", "error", "dense", "renders", "seconds", "edits"]
BISECT = 16  # halvings of a candidate bracket: 0.5 dB / 2¹⁶ ≈ 10⁻⁵ dB
DENSE_STEP_DB, DENSE_RANGE_DB = 0.05, 6.0


def status(edits: dict, drive: float) -> bool:
    """Inside the validated domain, from a full render at that drive."""
    return domain(synthesize(from_dict(grid.patch_dict({**edits, "drive.db": float(drive)})))).inside


def candidates(knees, d0: float, up: bool) -> list[float]:
    """The drives the report's search visits, in search order."""
    drops = knees.drops()
    sides = np.concatenate((drops * (1.0 - 1e-12), drops * (1.0 + 1e-12)))
    jumps = [float(d) for d in _drives(sides)]
    if up:
        steps = list(np.arange(d0 + HEADROOM_STEP_DB, DRIVE_MAX, HEADROOM_STEP_DB)) + [DRIVE_MAX]
        return sorted({float(d) for d in steps + jumps if d0 < d <= DRIVE_MAX})
    steps = list(np.arange(d0 - HEADROOM_STEP_DB, 0.0, -HEADROOM_STEP_DB)) + [0.0]
    return sorted({float(d) for d in steps + jumps if 0.0 <= d < d0}, reverse=True)


def run_one(args) -> dict:
    family, name, edits, dense = args
    p = from_dict(grid.patch_dict(edits))
    t0 = time.perf_counter()
    take = render(p)[1].takes[0]
    d0, here = p.drive.db, take.inside
    h = take.headroom
    estimate = (
        math.nan if h is None else (h.edge_db if h.edge_db is not None else (math.inf if h.db > 0 else -math.inf))
    )
    reference, renders = (math.inf if here else -math.inf), 0
    if h is not None:
        knees = synthesize(p).slew.knees
        prev = d0
        for d in candidates(knees, d0, up=here):
            renders += 1
            if status(edits, d) != here:
                inside, outside = (prev, d) if here else (d, prev)
                for _ in range(BISECT):
                    mid = 0.5 * (inside + outside)
                    renders += 1
                    if status(edits, mid):
                        inside = mid
                    else:
                        outside = mid
                reference = inside
                break
            prev = d
    error = estimate - reference if math.isfinite(estimate) and math.isfinite(reference) else math.nan
    agree = math.isnan(error) and (estimate == reference or h is None)
    # Optionally: every 0.05 dB between the fixture's drive and the edge, where the edge is near.
    first_dense = math.nan
    if dense and h is not None and math.isfinite(reference) and abs(reference - d0) <= DENSE_RANGE_DB:
        step = DENSE_STEP_DB if here else -DENSE_STEP_DB
        d = d0 + step
        while (d < reference) if here else (d > reference):
            renders += 1
            if status(edits, d) != here:
                first_dense = d
                break
            d += step
        else:
            first_dense = reference
    return {
        "family": family,
        "name": name,
        "drive": d0,
        "inside": here,
        "estimate": estimate,
        "reference": reference,
        "error": 0.0 if agree else error,
        "dense": first_dense,
        "renders": renders,
        "seconds": time.perf_counter() - t0,
        "edits": json.dumps(edits),
    }


def run(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["family"], r["name"]) for r in csv.DictReader(Path(out).read_text(encoding="utf-8").splitlines())} if out.exists() else set()
    jobs = [(fam, name, edits, args.dense) for fam, name, edits in grid.fixtures() if (fam, name) not in done]
    print(f"{len(jobs)} fixtures to run ({len(done)} already in {out})", flush=True)
    start = time.perf_counter()
    fresh = not out.exists()
    with Path(out).open("a", newline="", encoding="utf-8") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.DictWriter(f, FIELDS)
        if fresh:
            w.writeheader()
        for i, row in enumerate(pool.map(run_one, jobs, chunksize=1)):
            w.writerow(row)
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} after {time.perf_counter() - start:.0f} s", flush=True)
    print(f"done in {time.perf_counter() - start:.0f} s")


def load(path) -> list[dict]:
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
    for r in rows:
        for key in ("drive", "estimate", "reference", "error", "dense", "seconds"):
            r[key] = float(r[key])
        r["inside"] = r["inside"] == "True"
        r["renders"] = int(r["renders"])
    return rows


def split_infinite(rows) -> tuple[list, list, list]:
    """Of the fixtures whose status full renders never change within 0–48 dB: the silent ones (no headroom is
    reported for silence), the ones where the report is cautious (a finite edge where the status stays inside),
    and the ones where it is wrong (anything else that disagrees)."""
    silent, cautious, wrong = [], [], []
    for r in rows:
        if r["estimate"] == r["reference"]:
            continue
        if math.isnan(r["estimate"]) and r["reference"] == math.inf:
            silent.append(r)
        elif r["reference"] == math.inf and math.isfinite(r["estimate"]) and r["estimate"] > r["drive"]:
            cautious.append(r)
        else:
            wrong.append(r)
    return silent, cautious, wrong


def analyze(args) -> None:
    rows = load(args.csv)
    finite = [r for r in rows if not math.isnan(r["error"]) and math.isfinite(r["reference"])]
    infinite = [r for r in rows if not math.isfinite(r["reference"])]
    silent, cautious, wrong = split_infinite(infinite)
    print(f"{args.csv}: {len(rows)} fixtures, {sum(r['renders'] for r in rows)} full renders")
    print(
        f"  no change of status within 0–48 dB: {len(infinite)}; the report agrees on "
        f"{len(infinite) - len(silent) - len(cautious) - len(wrong)}, reports nothing for {len(silent)} silent ones, "
        f"is cautious on {len(cautious)} and wrong on {len(wrong)}"
    )
    for r in cautious + wrong:
        print(
            f"    {'cautious' if r in cautious else 'WRONG'}: {r['family']}/{r['name']} at {r['drive']:.1f} dB: "
            f"report {r['estimate']:.3f} dB, full renders {r['reference']}"
        )
    if finite:
        errors = np.array([r["error"] for r in finite])
        print(
            f"  an edge within 0–48 dB: {len(finite)}; error of the report (+ is optimistic): "
            f"max {errors.max():+.5f} dB, min {errors.min():+.4f} dB, median {np.median(errors):+.5f} dB"
        )
        print(f"    exact to 0.001 dB: {int(np.sum(np.abs(errors) < 0.001))}")
        for lo, hi in ((0.001, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, math.inf)):
            n = sum(1 for e in errors if lo <= -e < hi)
            print(f"    conservative by {lo:g}–{hi:g} dB: {n}")
        near = [r for r in finite if abs(r["reference"] - r["drive"]) <= 3.0]
        if near:
            print(
                f"  within 3 dB of the edge ({len(near)}): at most {-min(r['error'] for r in near):.3f} dB conservative"
            )
        for side, rs in (
            ("inside", [r for r in finite if r["inside"]]),
            ("outside", [r for r in finite if not r["inside"]]),
        ):
            if rs:
                e = np.array([r["error"] for r in rs])
                print(f"  {side} ({len(rs)}): max {e.max():+.5f} dB, min {e.min():+.4f} dB")
        for r in sorted(finite, key=lambda r: r["error"])[:5]:
            print(
                f"    most conservative: {r['family']}/{r['name']} at {r['drive']:.1f} dB: "
                f"report {r['estimate']:.3f}, full renders {r['reference']:.3f} dB"
            )
    dense = [r for r in rows if not math.isnan(r["dense"])]
    if dense:
        missed = [r for r in dense if abs(r["dense"] - r["reference"]) > DENSE_STEP_DB + 1e-9]
        print(f"  dense 0.05 dB scans: {len(dense)}; a change of status before the edge in {len(missed)}")
        for r in missed:
            print(f"    {r['family']}/{r['name']}: dense {r['dense']:.2f} dB, edge {r['reference']:.3f} dB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=str(grid.ROOT / "docs/grid/headroom.csv"))
    r.add_argument("--dense", action="store_true")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    a = sub.add_parser("analyze")
    a.add_argument("csv")
    args = ap.parse_args(argv)
    (run if args.cmd == "run" else analyze)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
