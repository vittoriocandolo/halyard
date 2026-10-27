# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Performance check (plan §3, §8.1): render time and memory on this machine.

Each case runs in a fresh interpreter after numpy, scipy and halyard_synth are
imported, so memory is measured above that baseline. A render with a new
seed computes its noise tables, as every take does; a repeated render finds
them cached, as audition's re-renders do. Both are reported, and the budget
applies to the first.

    python tools/bench.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys

HIT = r"""
import dataclasses, json, resource, time
from halyard_synth import cli
from halyard_synth.patch import loads, stretch
from halyard_synth.render import decimator, render
p = loads(cli.reference_text("fat").replace("sample_rate = 48000", "sample_rate = %(rate)d"))
p = stretch(p, "all", %(ms)f / p.length_ms, after_ms=0.0) if %(ms)f else p
if %(long)d:
    body = dataclasses.replace(p.body, amp=((0.0, 0.0), (0.5, 1.0), (100.0, 0.6), (800.0, 0.15), (2000.0, 0.0)))
    wires = dataclasses.replace(p.noise[2], amp=((0.0, 0.0), (3.0, 1.0), (100.0, 0.5), (500.0, 0.12), (1000.0, 0.0)))
    p = dataclasses.replace(p, body=body, noise=(p.noise[0], p.noise[1], wires))
decimator(%(base)d)  # loading the shipped coefficients is not part of a render
render(dataclasses.replace(p, seed=999))  # first-use costs of the libraries, not of this render
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
fresh, cached = 1e9, 1e9
for seed in range(1, 1 + %(repeat)d):
    q = dataclasses.replace(p, seed=seed)
    t0 = time.perf_counter(); audio, report = render(q); fresh = min(fresh, time.perf_counter() - t0)
    t0 = time.perf_counter(); render(q); cached = min(cached, time.perf_counter() - t0)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"fresh": fresh, "cached": cached, "mib": (peak - base) / 1024, "ms": p.length_ms,
                  "layers": len(p.noise)}))
"""

PACK = r"""
import json, resource, time
from halyard_synth import cli
from halyard_synth.patch import loads, softer
from halyard_synth.pack import build
from halyard_synth.render import decimator
p = loads(cli.reference_text("crack"))
decimator(48000)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
pk = build([(f"v{db}", softer(p, db)) for db in (0, 8, 16, 24)], takes=8)
seconds = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"seconds": seconds, "mib": (peak - base) / 1024, "files": sum(len(r) for r in pk.audio)}))
"""

HIT_S, LONG_S, PACK_S, MEMORY_MIB = 0.3, 1.5, 15.0, 300


def cpu() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(code: str) -> dict:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
    import numpy
    import scipy

    print(f"machine: {cpu()}, Python {platform.python_version()}, NumPy {numpy.__version__}, SciPy {scipy.__version__}")
    print(f"{'sound':<28s} {'rate':>6s} {'new seed':>9s} {'cached':>8s} {'limit':>7s} {'memory':>8s}")
    ok = True
    for label, ms, long, limit in (
        ("0.5 s hit, three layers", 500.0, 0, HIT_S),
        ("2 s body, 1 s of noise", 0.0, 1, LONG_S),
    ):
        for rate in (44100, 48000, 88200, 96000):
            base = 44100 if rate in (44100, 88200) else 48000
            r = run(HIT % {"rate": rate, "ms": ms, "long": long, "base": base, "repeat": 3})
            fine = r["fresh"] <= limit and r["mib"] <= MEMORY_MIB
            ok &= fine
            print(
                f"{label:<28s} {rate:6d} {1000 * r['fresh']:7.0f}ms {1000 * r['cached']:6.0f}ms {1000 * limit:5.0f}ms "
                f"{r['mib']:5.0f}MiB {'' if fine else 'OVER BUDGET'}"
            )
    r = run(PACK)
    fine = r["seconds"] <= PACK_S and r["mib"] <= MEMORY_MIB
    ok &= fine
    print(
        f"{'pack: 4 layers × 8 takes':<28s} {48000:6d} {r['seconds']:8.1f}s {'':>8s} {PACK_S:6.0f}s "
        f"{r['mib']:5.0f}MiB {'' if fine else 'OVER BUDGET'}  ({r['files']} files)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
