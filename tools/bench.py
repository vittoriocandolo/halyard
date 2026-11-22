# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Performance check (docs/design.md, docs/validation.md): render time and memory on this machine.

Each case runs in a fresh interpreter, a spawned process, after numpy, scipy
and halyard_synth are imported and a first render has paid the libraries'
first-use costs, so memory is measured above that baseline. A render with a
new seed computes its noise tables, as every take does; a repeated render
finds them cached, as audition's re-renders do. Both are reported, and the
budget applies to the first.

    uv run python -m tools.bench
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy

from halyard_synth import cli
from halyard_synth.pack import Options, build
from halyard_synth.patch import loads, softer, stretch
from halyard_synth.render import decimator, render
from tools import say

if TYPE_CHECKING:
    from collections.abc import Callable

HIT_S, LONG_S, PACK_S, MEMORY_MIB = 0.3, 1.5, 15.0, 300
RATES = (44100, 48000, 88200, 96000, 176400, 192000)
REPEAT = 3  # seeds per case; the fastest counts
KIB = 1024  # ru_maxrss is in KiB on Linux


class Case(NamedTuple):
    """One hit to time: its export rate, its length (0 keeps the preset's), whether it rings long, its preset."""

    rate: int
    ms: float
    long: bool
    preset: str = "fat"


def cpu() -> str:
    """Name this machine's processor."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _memory() -> int:
    """Return this process's peak resident memory, in KiB."""
    if sys.platform != "win32":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sys.exit("the benchmark reads peak memory from the resource module, which Windows lacks")


def hit(case: Case) -> dict[str, float]:
    """Time a new seed's render and a repeated one of a preset, and its memory above the baseline."""
    p = loads(cli.preset_text(case.preset).replace("sample_rate = 48000", f"sample_rate = {case.rate}"))
    if case.ms:
        p = stretch(p, "all", case.ms / p.length_ms, after_ms=0.0)
    if case.long:
        body = dataclasses.replace(p.body, amp=((0.0, 0.0), (0.5, 1.0), (100.0, 0.6), (800.0, 0.15), (2000.0, 0.0)))
        wires_amp = ((0.0, 0.0), (3.0, 1.0), (100.0, 0.5), (500.0, 0.12), (1000.0, 0.0))
        wires = dataclasses.replace(p.noise[2], amp=wires_amp)
        p = dataclasses.replace(p, body=body, noise=(p.noise[0], p.noise[1], wires))
    decimator(44100 if case.rate in {44100, 88200} else 48000)  # loading the shipped coefficients is not a render
    render(dataclasses.replace(p, seed=999))  # first-use costs of the libraries, not of this render
    base = _memory()
    fresh, cached = float("inf"), float("inf")
    for seed in range(1, 1 + REPEAT):
        q = dataclasses.replace(p, seed=seed)
        t0 = time.perf_counter()
        render(q)
        fresh = min(fresh, time.perf_counter() - t0)
        t0 = time.perf_counter()
        render(q)
        cached = min(cached, time.perf_counter() - t0)
    return {"fresh": fresh, "cached": cached, "mib": (_memory() - base) / KIB, "ms": p.length_ms}


def pack(_: None = None) -> dict[str, float]:
    """Time a pack of four layers of the crack preset, eight takes each, and its memory."""
    p = loads(cli.preset_text("crack"))
    decimator(48000)
    base = _memory()
    t0 = time.perf_counter()
    pk = build([(f"v{db}", softer(p, db)) for db in (0, 8, 16, 24)], Options(8))
    seconds = time.perf_counter() - t0
    return {"seconds": seconds, "mib": (_memory() - base) / KIB, "files": sum(len(r) for r in pk.audio)}


def fresh[T, R](work: Callable[[T], R], arg: T) -> R:
    """Run work(arg) in a newly spawned interpreter."""
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(work, arg).result()


def main() -> int:
    """Time every case, and return 1 if any is over its budget."""
    say(f"machine: {cpu()}, Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}")
    say(f"{'sound':<28s} {'rate':>6s} {'new seed':>9s} {'cached':>8s} {'limit':>7s} {'memory':>8s}")
    ok = True
    for label, ms, long, preset, limit in (
        ("0.5 s hit, three layers", 500.0, False, "fat", HIT_S),
        ("2 s body, 1 s of noise", 0.0, True, "fat", LONG_S),
        ("brush, swept over 380 ms", 0.0, False, "brush", HIT_S),
        ("rimshot, a hoop of 4 modes", 0.0, False, "rimshot", HIT_S),
    ):
        for rate in RATES:
            r = fresh(hit, Case(rate, ms, long, preset))
            fine = r["fresh"] <= limit and r["mib"] <= MEMORY_MIB
            ok &= fine
            say(
                f"{label:<28s} {rate:6d} {1000 * r['fresh']:7.0f}ms {1000 * r['cached']:6.0f}ms {1000 * limit:5.0f}ms "
                f"{r['mib']:5.0f}MiB {'' if fine else 'OVER BUDGET'}",
            )
    r = fresh(pack, None)
    fine = r["seconds"] <= PACK_S and r["mib"] <= MEMORY_MIB
    ok &= fine
    say(
        f"{'pack: 4 layers × 8 takes':<28s} {48000:6d} {r['seconds']:8.1f}s {'':>8s} {PACK_S:6.0f}s "
        f"{r['mib']:5.0f}MiB {'' if fine else 'OVER BUDGET'}  ({r['files']:.0f} files)",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
