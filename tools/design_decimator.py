# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design the frozen decimators (plan §3) and write them to src/halyard_synth/data.

The renderer never designs filters: it loads these coefficients and checks
their SHA-256. Run this only to change the rendering policy.

    uv run python -m tools.design_decimator            # write the shipped designs
    uv run python -m tools.design_decimator --check    # compare a fresh design with the shipped data

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

import numpy as np
import scipy
from scipy.signal import firwin, kaiserord, minimum_phase

PASS_HZ = 20000.0
ATTENUATION_DB = 116.0
N_FFT = 2**20
BASES = (44100, 48000)
SHIPPED_K = (16, 32, 64)
DATA = Path(__file__).resolve().parent.parent / "src" / "halyard_synth" / "data"


def design(base: int, K: int) -> np.ndarray:
    rate = base * K
    numtaps, beta = kaiserord(ATTENUATION_DB, (base / 2 - PASS_HZ) / (rate / 2))
    numtaps += 1 - numtaps % 2  # odd length: a type I prototype
    prototype = firwin(numtaps, (PASS_HZ + base / 2) / 2, window=("kaiser", beta), fs=rate)
    return normalize(minimum_phase(prototype, method="homomorphic", n_fft=N_FFT, half=False))


def normalize(h: np.ndarray) -> np.ndarray:
    """Unit DC gain: nudge the largest tap until the exactly rounded sum is 1."""
    h = h / math.fsum(h)
    k = int(np.argmax(np.abs(h)))
    for _ in range(64):
        residual = 1.0 - math.fsum(h)
        if residual == 0.0:
            return h
        h[k] += residual
    raise RuntimeError("could not normalize the taps")


def measure(h: np.ndarray, base: int, K: int) -> dict:
    rate = base * K
    n = 2**23
    H = np.abs(np.fft.rfft(h, n))
    f = np.arange(H.size) * rate / n
    energy = np.cumsum(h**2) / np.sum(h**2)
    return {
        "passband_db": float(np.max(np.abs(20 * np.log10(H[f <= PASS_HZ])))),
        "stopband_db": float(-20 * np.log10(np.max(H[f >= base / 2]))),
        "length_ms": 1000 * h.size / rate,
        "energy99_ms": 1000 * int(np.searchsorted(energy, 0.99)) / rate,
    }


def checksum(h: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(h, dtype="<f8").tobytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare a fresh design with the shipped data")
    args = ap.parse_args(argv)
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
        "filters": {},
    }
    status = 0
    for K in SHIPPED_K:
        for base in BASES:
            h = design(base, K)
            name = f"{base}x{K}"
            entry = {
                "base": base,
                "K": K,
                "rate": base * K,
                "taps": int(h.size),
                "file": f"decimator-{name}.f64",
                "sha256": checksum(h),
                **measure(h, base, K),
            }
            manifest["filters"][name] = entry
            print(
                f"{name}: {h.size} taps, {entry['length_ms']:.2f} ms, 99% energy in {entry['energy99_ms']:.2f} ms, "
                f"passband {entry['passband_db']:.1e} dB, stopband {entry['stopband_db']:.1f} dB"
            )
            if args.check:
                shipped = np.fromfile(DATA / entry["file"], dtype="<f8")
                diff = np.max(np.abs(shipped - h)) if shipped.size == h.size else math.inf
                print(f"  shipped vs fresh design: max difference {diff:.1e}")
                status |= int(diff > 1e-12)
            else:
                h.astype("<f8").tofile(DATA / entry["file"])
    if not args.check:
        (DATA / "decimators.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
