# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The adversarial grid (plan §8.3): combined corners of the valid domain.

Every fixture is rendered by the shipping path and compared with its reference
at 128 × base (6.144 MHz for the 48 kHz family), whose convergence is checked
against 256 × base, and against 512 × base on a subset. The results fix the
internal rate and the constant of the validated-domain rule.

    python tools/grid.py run --k 32 [--family NAME ...] [--out docs/grid/k32.csv]
    python tools/grid.py analyze docs/grid/k16.csv docs/grid/k32.csv
    python tools/grid.py list [--family NAME]

A run appends to its CSV and skips fixtures already there, so it can resume.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from halyard_synth import cli  # noqa: E402
from halyard_synth import patch as P  # noqa: E402
from halyard_synth.reference import compare, reference  # noqa: E402
from halyard_synth.render import LIMIT, QUIET, SATURATION, synthesize  # noqa: E402

INF = math.inf
FLOOR = 1e-7


def amp(*pts):
    return [{"ms": float(m), "gain": float(g)} for m, g in pts]


def pitch(*pts):
    return [{"ms": float(m), "ratio": float(r)} for m, r in pts]


def modes(*pairs):
    return [{"ratio": float(r), "level_db": float(v)} for r, v in pairs]


WIRES = amp((0, 0), (1.5, 1), (40, 0.4), (140, 0.08), (260, 0))
STICK = amp((0, 0), (0.05, 1), (0.15, 0.7), (3, 0.1), (8, 0))
MEMBRANE = ((1, 0), (1.593, -5), (2.136, -11), (2.295, -14))


def layer(level, band, shape, rattle=0.0):
    return {
        "level_db": float(level),
        "band_hz": [float(band[0]), float(band[1])],
        "rattle": float(rattle),
        "amp": shape,
    }


BASE = {
    "instrument": "snare",
    "schema": 1,
    "seed": 1,
    "body": {
        "tune_hz": 185.0,
        "level_db": -4.0,
        "damping": 1.0,
        "modes": modes(*MEMBRANE[:3]),
        "pitch": pitch((0, 1.08), (15, 1)),
        "amp": amp((0, 0), (0.4, 1), (25, 0.45), (90, 0.1), (180, 0)),
    },
    "noise": [layer(-10, (1500, 16000), STICK), layer(-6, (2500, 9000), WIRES, 0.5)],
    "drive": {"db": 0.0},
    "output": {"sample_rate": 48000, "format": "float32", "mode": "gain", "db": 0.0, "polarity": 1},
}

BANDS = {
    "lowest": (20, 200),
    "narrowest": (5000, 5001),
    "wires": (2000, 9000),
    "widest": (20, 20000),
    "top": (19000, 20000),
    "stick": (1000, 16000),
    "clap": (800, 2500),
}


def set_path(d: dict, path: str, value) -> None:
    """Set a field by its patch path, counting from 1: set_path(d, "noise[2].amp", [...])."""
    node = d
    parts = path.split(".")
    for i, part in enumerate(parts):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        name, index = m.group(1), m.group(2)
        last = i == len(parts) - 1
        if index is None:
            if last:
                node[name] = value
            else:
                node = node[name]
        elif last:
            node[name][int(index) - 1] = value
        else:
            node = node[name][int(index) - 1]


def patch_dict(edits: dict) -> dict:
    d = copy.deepcopy(BASE)
    for path, value in edits.items():
        set_path(d, path, copy.deepcopy(value))
    return d


def valid(edits: dict) -> bool:
    try:
        P.from_dict(patch_dict(edits))
        return True
    except P.PatchError:
        return False


BODY_ONLY = {"noise[1].level_db": -INF, "noise[2].level_db": -INF}
NOISE_ONLY = {"body.level_db": -INF, "noise[1].level_db": -INF, "noise[2].rattle": 0.0}


def fixtures():
    """(family, name, edits): combined corners and a random sample of the valid domain."""
    out = []

    def add(family, name, **edits):
        if valid(edits):
            out.append((family, name, edits))

    drives5 = (0, 12, 24, 36, 48)
    # Body: tuning × modes × glide × damping × level × drive.
    mode_sets = {"one": ((1, 0),), "membrane": MEMBRANE, "wide": ((1, -6), (2.5, 0), (5, -6), (8, -12))}
    for tune in (40, 100, 185, 400, 1000, 2000, 4000):
        for mname, ms in mode_sets.items():
            top = max(r for r, _ in ms)
            deep = min(2.0, 20000 / (tune * top))
            for drive in drives5:
                add(
                    "body",
                    f"t{tune}-{mname}-deep-b3-d{drive}",
                    **BODY_ONLY,
                    **{
                        "body.tune_hz": float(tune),
                        "body.modes": modes(*ms),
                        "body.damping": 3.0,
                        "body.pitch": pitch((0, deep), (3, 1 + (deep - 1) / 4), (20, 1)),
                        "body.level_db": 0.0,
                        "drive.db": float(drive),
                    },
                )
            for drive in (0, 24, 48):
                add(
                    "body",
                    f"t{tune}-{mname}-flat-b0-L24-d{drive}",
                    **BODY_ONLY,
                    **{
                        "body.tune_hz": float(tune),
                        "body.modes": modes(*ms),
                        "body.damping": 0.0,
                        "body.pitch": pitch((0, 1), (20, 1)),
                        "body.level_db": 24.0,
                        "drive.db": float(drive),
                    },
                )
    # Noise: bands × level × drive, one layer.
    for bname, band in BANDS.items():
        for level in (0, 12, 24):
            for drive in (0, 12, 18, 24, 30, 36, 48):
                add(
                    "noise",
                    f"{bname}-L{level}-d{drive}",
                    **NOISE_ONLY,
                    **{
                        "noise[2].band_hz": list(map(float, band)),
                        "noise[2].level_db": float(level),
                        "drive.db": float(drive),
                    },
                )
    # Layers: four coherent overlapping layers at +24 dB, and crossfades.
    four = [
        layer(24, (1000, 16000), STICK),
        layer(24, (2000, 9000), WIRES),
        layer(24, (2500, 9000), WIRES),
        layer(24, (800, 2500), amp((0, 0), (3, 1), (60, 0.5), (200, 0.1), (420, 0))),
    ]
    for drive in drives5:
        add("layers", f"four-L24-d{drive}", **{"body.level_db": -INF, "noise": four, "drive.db": float(drive)})
    fade_out = amp((0, 0), (1, 1), (40, 0.5), (120, 0))
    fade_in = amp((0, 0), (20, 0.1), (80, 1), (240, 0))
    for level in (0, 12):
        for drive in (0, 12, 24, 36):
            add(
                "layers",
                f"crossfade-L{level}-d{drive}",
                **{
                    "body.level_db": -INF,
                    "drive.db": float(drive),
                    "noise": [layer(level, (3000, 12000), fade_out), layer(level, (1000, 5000), fade_in)],
                },
            )
    # Rattle: depth × fundamental × glide × drive, with the body sounding or silent.
    for depth in (0.25, 0.5, 1.0):
        for tune in (40, 185, 1000, 4000):
            for glide in ("flat", "deep"):
                r0 = 1.0 if glide == "flat" else min(2.0, 20000 / (tune * 2.136))
                for drive in (0, 12, 24):
                    for body in ("on", "off"):
                        if body == "off" and drive == 12:
                            continue
                        add(
                            "rattle",
                            f"d{depth}-t{tune}-{glide}-{body}-d{drive}",
                            **{
                                "body.tune_hz": float(tune),
                                "body.pitch": pitch((0, r0), (2, 1 + (r0 - 1) / 3), (15, 1)),
                                "body.level_db": 0.0 if body == "on" else -INF,
                                "noise[2].rattle": depth,
                                "noise[2].level_db": 0.0,
                                "drive.db": float(drive),
                            },
                        )
    # Clicks: 10–50 µs rises under full-scale noise, and high, strongly damped modes.
    for h in (0.01, 0.02, 0.05):
        for bname in ("widest", "stick", "top"):
            for level in (0, 12):
                for drive in (0, 12, 24):
                    shape = amp((0, 0), (h, 1), (h + 0.1, 0.7), (2, 0.1), (6, 0))
                    add(
                        "clicks",
                        f"h{h * 1000:g}us-{bname}-L{level}-d{drive}",
                        **{
                            "body.level_db": -INF,
                            "noise[2].level_db": -INF,
                            "noise[1].level_db": float(level),
                            "noise[1].band_hz": list(map(float, BANDS[bname])),
                            "noise[1].amp": shape,
                            "drive.db": float(drive),
                        },
                    )
    for tune in (2000, 4000):
        for drive in (0, 12, 24):
            for rise in (0.01, 0.05):
                add(
                    "clicks",
                    f"ping-t{tune}-r{rise * 1000:g}us-d{drive}",
                    **BODY_ONLY,
                    **{
                        "body.tune_hz": float(tune),
                        "body.level_db": 0.0,
                        "body.damping": 3.0,
                        "body.modes": modes((1, 0), (min(5.0, 20000 / tune), -3)),
                        "body.pitch": pitch((0, 1), (5, 1)),
                        "body.amp": amp((0, 0), (rise, 1), (3, 0.3), (12, 0.05), (30, 0)),
                        "drive.db": float(drive),
                    },
                )
    # Bursts: count × spacing × rise × drive, and stacked trains.
    base = P.from_dict(patch_dict({**NOISE_ONLY, "noise[2].band_hz": [800.0, 2500.0], "noise[2].level_db": 0.0}))
    for count in (2, 4, 10):
        for spacing in (1, 5, 30):
            for rise in (0.01, 0.1, 1.0):
                try:
                    shape = P.bursts(base, "noise[2]", count, float(spacing), rise=rise, tail=40.0).noise[1].amp
                except P.PatchError:
                    continue
                for drive in (0, 18):
                    add(
                        "bursts",
                        f"n{count}-s{spacing}-r{rise * 1000:g}us-d{drive}",
                        **NOISE_ONLY,
                        **{
                            "noise[2].band_hz": [800.0, 2500.0],
                            "noise[2].level_db": 0.0,
                            "noise[2].amp": amp(*shape),
                            "drive.db": float(drive),
                        },
                    )
    stacked = P.bursts(
        P.bursts(
            P.from_dict(
                patch_dict(
                    {
                        "body.level_db": -INF,
                        "noise[1].level_db": 0.0,
                        "noise[2].level_db": -3.0,
                        "noise[2].rattle": 0.0,
                        "noise[1].band_hz": [800.0, 2500.0],
                        "noise[2].band_hz": [1200.0, 5000.0],
                    }
                )
            ),
            "noise[1]",
            4,
            10.0,
        ),
        "noise[2]",
        3,
        12.0,
        start=4.0,
    )
    for drive in (0, 12, 24):
        add(
            "bursts",
            f"stacked-d{drive}",
            **{"body.level_db": -INF, "noise": P.to_dict(stacked)["noise"], "drive.db": float(drive)},
        )
    # Snaps: a single mode at 1–4 kHz under bursts of 1–30 ms.
    for tune in (1000, 2000, 4000):
        for length in (1, 10, 30):
            for drive in (0, 12, 24):
                burst = amp((0, 0), (0.05, 1), (0.15, 0.7), (length * 0.3, 0.25), (length, 0))
                add(
                    "snaps",
                    f"t{tune}-b{length}-d{drive}",
                    **{
                        "body.tune_hz": float(tune),
                        "body.modes": modes((1, 0)),
                        "body.level_db": -2.0,
                        "body.pitch": pitch((0, 1.02), (3, 1)),
                        "body.amp": amp((0, 0), (0.1, 1), (4, 0.4), (15, 0.08), (40, 0)),
                        "noise": [layer(-6, (2000, 14000), burst)],
                        "drive.db": float(drive),
                    },
                )
    # Dynamics: softer ladders from every reference.
    for name in cli.REFERENCES:
        ref = P.loads(cli.reference_text(name))
        for db in (0, 6, 12, 18, 24, 30, 36):
            d = P.to_dict(P.softer(ref, db))
            d["output"]["db"] = 0.0
            add("dynamics", f"{name}-softer{db}", **{k: d[k] for k in ("seed", "body", "noise", "drive", "output")})
    # Seeds: one fixture per family across 16 seeds.
    picks = [
        ("body", "t185-membrane-deep-b3-d24"),
        ("noise", "wires-L0-d18"),
        ("rattle", "d0.5-t185-deep-on-d12"),
        ("clicks", "h20us-stick-L0-d12"),
        ("bursts", "n4-s5-r100us-d18"),
        ("snaps", "t2000-b10-d12"),
    ]
    named = {(f, n): e for f, n, e in out}
    for fam, name in picks:
        for seed in range(2, 17):
            add("seeds", f"{fam}-{name}-s{seed}", **named[(fam, name)], seed=seed)
    # Export rates share their family's internal rate.
    for rate in (44100, 88200, 96000):
        for drive in (0, 24, 36):
            add("rates", f"sr{rate}-d{drive}", **{"output.sample_rate": rate, "drive.db": float(drive)})
            add(
                "rates",
                f"sr{rate}-hot-d{drive}",
                **NOISE_ONLY,
                **{"output.sample_rate": rate, "drive.db": float(drive), "noise[2].level_db": 6.0},
            )
    # A random sample of the whole valid domain.
    rng = np.random.default_rng(2026)
    made = 0
    while made < 220:
        e = random_edits(rng)
        if valid(e):
            out.append(("random", f"r{made:03d}", e))
            made += 1
    # The longest sounds: 2 s of body and 1 s of noise, at 96 kHz.
    for drive in (0, 12, 24):
        add(
            "longest",
            f"2s-96k-d{drive}",
            **{
                "output.sample_rate": 96000,
                "drive.db": float(drive),
                "body.amp": amp((0, 0), (0.5, 1), (100, 0.6), (800, 0.15), (2000, 0)),
                "noise[2].amp": amp((0, 0), (1.5, 1), (100, 0.4), (500, 0.1), (1000, 0)),
            },
        )
    return out


def random_edits(rng) -> dict:
    tune = float(10 ** rng.uniform(math.log10(40), math.log10(4000)))
    n_modes = int(rng.integers(1, 5))
    ratios = [1.0] + sorted(float(x) for x in rng.uniform(1.05, 8.0, n_modes - 1))
    levels = [float(x) for x in rng.uniform(-40, 0, n_modes)]
    levels[int(rng.integers(n_modes))] = 0.0
    top = min(2.0, 20000 / (tune * max(ratios)))
    r0 = float(rng.uniform(max(10 / tune, 0.5), max(top, 0.51))) if top > 0.5 else 1.0

    def shape(span):
        rise = float(10 ** rng.uniform(-2, 0.5))
        mid = rise + float(rng.uniform(0.2, 0.6)) * span
        return amp(
            (0, 0),
            (rise, 1),
            (rise + 0.1 + 0.2 * span * rng.uniform(), float(rng.uniform(0.2, 1))),
            (mid + 0.2 + 0.3 * span * rng.uniform(), float(rng.uniform(0, 0.3))),
            (mid + 0.5 + span, 0),
        )

    layers = []
    for _ in range(int(rng.integers(1, 5))):
        lo = float(10 ** rng.uniform(math.log10(20), math.log10(15000)))
        hi = float(min(20000.0, lo * 10 ** rng.uniform(0.01, 2)))
        layers.append(
            layer(
                float(rng.choice([-INF, rng.uniform(-40, 24)])),
                (lo, hi),
                shape(float(rng.uniform(5, 400))),
                float(rng.choice([0.0, rng.uniform(0, 1)])),
            )
        )
    e = {
        "seed": int(rng.integers(0, 2**31)),
        "body.tune_hz": tune,
        "body.level_db": float(rng.choice([-INF, rng.uniform(-24, 24)])),
        "body.damping": float(rng.uniform(0, 3)),
        "body.modes": modes(*zip(ratios, levels)),
        "body.pitch": pitch((0, r0), (float(rng.uniform(1, 30)), 1)),
        "body.amp": shape(float(rng.uniform(10, 800))),
        "noise": layers,
        "drive.db": float(rng.choice([0.0, rng.uniform(0, 48)])),
        "output.sample_rate": int(rng.choice([44100, 48000, 88200, 96000])),
    }
    if e["body.level_db"] == -INF and all(n["level_db"] == -INF for n in layers):
        layers[0]["level_db"] = 0.0  # a silent fixture would test nothing; no draw is spent, so the others stay
    return e


FIELDS = [
    "family",
    "name",
    "k",
    "seed",
    "drive",
    "alpha",
    "e_rel",
    "e_max",
    "max_abs",
    "peak",
    "c_rel",
    "c_max",
    "c_abs",
    "c3_rel",
    "c3_max",
    "c3_abs",
    "knee",
    "kink",
    "ship_s",
    "total_s",
    "edits",
]


def run_one(args) -> dict:
    family, name, edits, k, deep = args
    p = P.from_dict(patch_dict(edits))
    t0 = time.perf_counter()
    ship = synthesize(p, k=k)
    t_ship = time.perf_counter() - t0
    ref = reference(p, 128, k=k)
    check = reference(p, 256, k=k)
    e, c = compare(ship.out, ref.out), compare(ref.out, check.out)
    row = {
        "family": family,
        "name": name,
        "k": k,
        "seed": p.seed,
        "drive": p.drive.db,
        "alpha": ship.alpha,
        "e_rel": e.rel_db,
        "e_max": e.max_db,
        "max_abs": e.max_abs,
        "peak": e.peak,
        "c_rel": c.rel_db,
        "c_max": c.max_db,
        "c_abs": c.max_abs,
        "knee": ship.slew.knee(ship.alpha),
        "kink": ship.kink,
        "ship_s": t_ship,
        "edits": json.dumps(edits),
    }
    del ref
    if deep:
        c3 = compare(check.out, reference(p, 512, k=k).out)
        row.update(c3_rel=c3.rel_db, c3_max=c3.max_db, c3_abs=c3.max_abs)
    row["total_s"] = time.perf_counter() - t0
    return row


def run(args) -> None:
    todo = [f for f in fixtures() if not args.family or f[0] in args.family]
    out = Path(args.out or ROOT / f"docs/grid/k{args.k}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["family"], r["name"]) for r in csv.DictReader(open(out))}
    jobs = [
        (fam, name, edits, args.k, i % 10 == 0 and fam != "longest")
        for i, (fam, name, edits) in enumerate(todo)
        if (fam, name) not in done
    ]
    # The longest fixtures first, so the pool never waits on a straggler at the end.
    jobs.sort(key=lambda j: -P.from_dict(patch_dict(j[2])).length_ms)
    print(f"{len(jobs)} fixtures to run at K = {args.k} ({len(done)} already in {out})", flush=True)
    start = time.perf_counter()
    fresh = not out.exists()
    with open(out, "a", newline="") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.DictWriter(f, FIELDS)
        if fresh:
            w.writeheader()
        for i, row in enumerate(pool.map(run_one, jobs, chunksize=1)):
            w.writerow(row)
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} after {time.perf_counter() - start:.0f} s", flush=True)
    print(f"done in {time.perf_counter() - start:.0f} s")


def load(path, saturation: float = SATURATION) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for key in FIELDS[2:-1]:
            r[key] = float(r[key]) if r.get(key) not in (None, "") else math.nan
        near = r["peak"] < QUIET
        r["pass"] = r["max_abs"] <= FLOOR if near else (r["e_rel"] <= -100 and r["e_max"] <= -100)
        r["converged"] = r["c_abs"] <= FLOOR / 10 if near else (r["c_rel"] <= -120 and r["c_max"] <= -120)
        q = r["knee"]
        r["saturation"] = math.exp(-(math.pi**2) / q) * r["peak"] if q > 0 else 0.0  # per unit coefficient
        r["allowed"] = LIMIT * r["peak"] if not near else FLOOR
        predicted = saturation * r["saturation"] + r["kink"]
        r["predicted"] = predicted
        r["inside"] = predicted <= r["allowed"]
        r["margin_db"] = 20 * math.log10(r["allowed"] / predicted) if predicted > 0 else math.inf
    return rows


def fit_saturation(rows) -> tuple[float, dict | None]:
    """The smallest coefficient that covers every converged fixture with at least 6 dB of margin (plan §5).

    Where the measured error exceeds the kink bound, the saturation term must
    cover the rest twice over: C · e^(−π²/q) · P ≥ 2 · (measured − kink bound).
    """
    need, worst = 0.0, None
    for r in rows:
        if r["converged"] and r["saturation"] > 0 and r["max_abs"] > r["kink"]:
            c = 2 * (r["max_abs"] - r["kink"]) / r["saturation"]
            if c > need:
                need, worst = c, r
    return need, worst


def analyze(args) -> None:
    for path in args.csv:
        rows = load(path)
        k = int(rows[0]["k"])
        conv = [r for r in rows if r["converged"]]
        need, worst = fit_saturation(rows)
        linear = [r for r in rows if r["alpha"] == 0.0 and r["converged"]]
        misses = [r for r in linear if r["max_abs"] > r["kink"] and r["max_abs"] > 1e-15 * r["peak"]]
        thin = [r for r in linear if 2 * r["max_abs"] > r["kink"] and r["max_abs"] > 1e-15 * r["peak"]]
        print(f"\n{path}: K = {k}, {len(rows)} fixtures")
        print(f"  pass {sum(r['pass'] for r in rows)}, converged references {len(conv)}")
        print(
            f"  undriven fixtures {len(linear)}: pass {sum(r['pass'] for r in linear)}; the kink bound is exceeded by "
            f"the measured error in {len(misses)}, and within 6 dB of it in {len(thin)}"
        )
        for r in misses + thin:
            print(f"    {r['family']}/{r['name']}: measured {r['max_abs']:.3g}, bound {r['kink']:.3g}")
        text = f"  saturation coefficient for 6 dB of margin on every converged fixture: {need:.3g}"
        if worst:
            text += f" ({worst['family']}/{worst['name']})"
        print(text + f"; the rule uses {SATURATION:g}")
        rows = load(path, max(need, SATURATION))
        inside = [r for r in rows if r["inside"]]
        bad = [r for r in inside if not (r["pass"] and r["converged"])]
        print(f"  inside the validated domain: {len(inside)}; of those failing or unconverged: {len(bad)}")
        for r in bad:
            print(f"    {r['family']}/{r['name']}: E_max {r['e_max']:.1f} dB, convergence {r['c_max']:.1f} dB")
        loud = [r for r in inside if r["peak"] >= QUIET]
        if loud:
            w = max(loud, key=lambda r: r["e_max"])
            print(f"  worst error inside: E_max {w['e_max']:.1f} dB ({w['family']}/{w['name']})")
            m = min(loud, key=lambda r: 20 * math.log10(r["predicted"] / r["max_abs"]) if r["max_abs"] > 0 else INF)
            if m["max_abs"] > 0:
                print(
                    f"  least conservative inside: predicted {20 * math.log10(m['predicted'] / m['max_abs']):.1f} dB "
                    f"above measured ({m['family']}/{m['name']})"
                )
        print(
            f"  outside but passing (the rule's caution): {sum(1 for r in rows if not r['inside'] and r['pass'] and r['converged'])}"
        )
        checks = [r for r in rows if not math.isnan(r["c3_max"])]
        if checks:
            better = sum(1 for r in checks if r["c3_max"] <= r["c_max"] + 1e-9 or r["c3_max"] <= -200)
            print(f"  512× check on {len(checks)} fixtures: 256×/512× agree at least as well as 128×/256× in {better}")
        print(f"  {'family':10s} {'n':>4s} {'inside':>7s} {'pass':>5s} {'unconverged':>12s} {'worst inside':>13s}")
        families: dict[str, list] = {}
        for r in rows:
            f = families.setdefault(r["family"], [0, 0, 0, 0, -INF])
            f[0] += 1
            f[1] += r["inside"]
            f[2] += r["pass"] and r["converged"]
            f[3] += not r["converged"]
            if r["inside"] and r["peak"] >= QUIET:
                f[4] = max(f[4], r["e_max"])
        for name, (n, i, ok, unconv, w) in families.items():
            print(f"  {name:10s} {n:4d} {i:7d} {ok:5d} {unconv:12d} {w:13.1f}")


def list_fixtures(args) -> None:
    rows = [f for f in fixtures() if not args.family or f[0] in args.family]
    counts: dict[str, int] = {}
    for fam, name, _ in rows:
        counts[fam] = counts.get(fam, 0) + 1
        if args.verbose:
            print(f"{fam}/{name}")
    print(json.dumps(counts), sum(counts.values()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--k", type=int, default=32)
    r.add_argument("--family", action="append")
    r.add_argument("--out")
    r.add_argument("--workers", type=int, default=os.cpu_count())
    a = sub.add_parser("analyze")
    a.add_argument("csv", nargs="+")
    lst = sub.add_parser("list")
    lst.add_argument("--family", action="append")
    lst.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    {"run": run, "analyze": analyze, "list": list_fixtures}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
