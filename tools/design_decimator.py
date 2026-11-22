# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design the frozen decimators (docs/design.md: Rates and the decimator) and write them to src/halyard_synth/data.

The renderer never designs filters: it loads these coefficients and checks
their SHA-256. Run this only to change the rendering policy.

    uv run python -m tools.design_decimator            # write the shipped designs
    uv run python -m tools.design_decimator --check    # compare a fresh design with the shipped data

The homomorphic method takes the logarithm of a spectrum that reaches 116 dB
down, so the rounding of its FFTs, which differs between platforms and library
versions, moves the taps by up to about 10⁻⁹: a fresh design is the same
filter, not the same bits. The check compares what a filter is: its length,
and its passband and stopband to 10⁻⁴ dB, and reports how far apart the two
frequency responses lie.

Design: a Kaiser-window linear-phase prototype at 116 dB, converted to minimum
phase by SciPy's full-length, magnitude-matching homomorphic method with a
2^20-point FFT, then renormalized so that the taps sum to exactly 1. The
renderer ships K = 32; 16× and 64× are shipped too, for the adversarial grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy.signal import firwin, kaiserord, minimum_phase

from tools import say

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]

PASS_HZ = 20000.0
ATTENUATION_DB = 116.0
N_FFT = 2**20
BASES = (44100, 48000)
SHIPPED_K = (16, 32, 64)
SPEC_DB = (
    1e-4  # a fresh design's passband and stopband agree with the shipped one's to this, ten times finer than stated
)
DATA = Path(__file__).resolve().parent.parent / "src" / "halyard_synth" / "data"


def design(base: int, k: int) -> Floats:
    """Design the decimator for a base rate at factor k."""
    rate = base * k
    numtaps, beta = kaiserord(ATTENUATION_DB, (base / 2 - PASS_HZ) / (rate / 2))
    numtaps += 1 - numtaps % 2  # odd length: a type I prototype
    prototype = firwin(numtaps, (PASS_HZ + base / 2) / 2, window=("kaiser", beta), fs=rate)
    return normalize(minimum_phase(prototype, method="homomorphic", n_fft=N_FFT, half=False))


def normalize(taps: Floats) -> Floats:
    """Give the taps unit DC gain: nudge the largest tap until the exactly rounded sum is 1."""
    h = taps / math.fsum(taps)
    k = int(np.argmax(np.abs(h)))
    for _ in range(64):
        residual = 1.0 - math.fsum(h)
        if residual == 0.0:
            return h
        h[k] += residual
    msg = "could not normalize the taps"
    raise RuntimeError(msg)


def measure(h: Floats, base: int, k: int) -> dict[str, float]:
    """Measure a design: its passband deviation, stopband, length and where 99% of its energy lies."""
    rate = base * k
    n = 2**23
    response = np.abs(np.fft.rfft(h, n))
    f = np.arange(response.size) * rate / n
    energy = np.cumsum(h**2) / np.sum(h**2)
    return {
        "passband_db": float(np.max(np.abs(20 * np.log10(response[f <= PASS_HZ])))),
        "stopband_db": float(-20 * np.log10(np.max(response[f >= base / 2]))),
        "length_ms": 1000 * h.size / rate,
        "energy99_ms": 1000 * int(np.searchsorted(energy, 0.99)) / rate,
    }


def checksum(h: Floats) -> str:
    """Return the SHA-256 of the taps as little-endian float64."""
    return hashlib.sha256(np.asarray(h, dtype="<f8").tobytes()).hexdigest()


def check(fresh: Floats, shipped: Floats, base: int, k: int) -> int:
    """Compare a fresh design with the shipped one, and return 1 unless they are the same filter."""
    if fresh.size != shipped.size:
        say(f"  shipped vs fresh design: {shipped.size} taps against {fresh.size}")
        return 1
    a, b = measure(fresh, base, k), measure(shipped, base, k)
    spec = max(abs(a["passband_db"] - b["passband_db"]), abs(a["stopband_db"] - b["stopband_db"]))
    apart = 20 * math.log10(max(float(np.max(np.abs(np.fft.rfft(fresh - shipped, 2**22)))), 1e-300))
    taps = float(np.max(np.abs(fresh - shipped)))
    say(f"  shipped vs fresh design: specifications within {spec:.1e} dB, responses {apart:.1f} dB apart")
    say(f"  taps up to {taps:.1e} apart")
    return int(spec > SPEC_DB)


def main(argv: Sequence[str] | None = None) -> int:
    """Design every shipped decimator, and write it, or with --check compare it with the shipped one."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare a fresh design with the shipped data")
    args = ap.parse_args(argv)
    filters: dict[str, dict[str, Any]] = {}
    manifest = {
        "design": {
            "method": "Kaiser linear-phase prototype, minimum_phase(method='homomorphic', half=False), "
            "taps renormalized to sum to 1",
            "passband_hz": PASS_HZ,
            "attenuation_db": ATTENUATION_DB,
            "n_fft": N_FFT,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "filters": filters,
    }
    status = 0
    for k in SHIPPED_K:
        for base in BASES:
            h = design(base, k)
            name = f"{base}x{k}"
            entry: dict[str, Any] = {
                "base": base,
                "K": k,
                "rate": base * k,
                "taps": int(h.size),
                "file": f"decimator-{name}.f64",
                "sha256": checksum(h),
                **measure(h, base, k),
            }
            filters[name] = entry
            say(
                f"{name}: {h.size} taps, {entry['length_ms']:.2f} ms, 99% energy in {entry['energy99_ms']:.2f} ms, "
                f"passband {entry['passband_db']:.1e} dB, stopband {entry['stopband_db']:.1f} dB",
            )
            if args.check:
                status |= check(h, np.fromfile(DATA / entry["file"], dtype="<f8"), base, k)
            else:
                h.astype("<f8").tofile(DATA / entry["file"])
    if not args.check:
        (DATA / "decimators.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
