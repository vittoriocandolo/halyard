# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seeded continuous-time Gaussian noise for the layers (plan §2.4, §6).

    n_j(t) = γ_j Σ_{k=1}^{20000} |H_j(f_k)| ρ_k cos(2π f_k t + θ_k),    f_k = k Hz

Every layer uses the same components, a pure function of the seed. SHAKE-256
over a domain-separated key yields 16 bytes per component, which become two
53-bit uniforms and then, by the polar form of Box–Muller, a Rayleigh
amplitude ρ_k (mean square 2) and a uniform phase θ_k. Each component is a
circular Gaussian, so n_j(t) is exactly Gaussian at every t.

A layer weights the components by its band |H_j|: second-order Butterworth
high- and low-pass skirts, each with its half-power point at a band edge.
γ_j sets the expected mean square to ½, that of a unit sine. The layers are
therefore coherent: identical bands give identical noise, a crossfade between
two layers is a spectral morph of one noise, and editing a band reweights the
same components instead of rerolling them.

The noise repeats every second. Its samples at an even rate above 40 kHz
come exactly from one inverse real FFT of length rate × 1 s. The tables of a
render's layers are computed in parallel, one transform each, so a table is
the same array however it was computed, and they are cached while the seed
and band stay the same. Values and slopes at a few arbitrary times, which the
kink bound needs at knots, come from those tables by cubic interpolation.

The same construction, under its own key, gives the TPDF dither of PCM
exports, so a PCM file is as reproducible as a float one.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import scipy.fft

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike, NDArray

    type Floats = NDArray[np.float64]
    type Words = NDArray[np.uint64]

COMPONENTS = 20000
KEY = b"snare/noise/v1"
DITHER_KEY = b"snare/dither/v1"


def _check_seed(seed: int) -> None:
    """Refuse a seed outside 0 … 2**63 − 1."""
    if not 0 <= seed < 2**63:
        msg = "seed must be an integer from 0 to 2**63 - 1"
        raise ValueError(msg)


def uniforms(seed: int) -> tuple[Words, Words]:
    """Return the two 53-bit integers per component, from SHAKE-256 of the key and seed."""
    _check_seed(seed)
    stream = hashlib.shake_256(KEY + seed.to_bytes(8, "little")).digest(16 * COMPONENTS)
    words = np.frombuffer(stream, dtype="<u8").reshape(COMPONENTS, 2) >> np.uint64(11)
    return words[:, 0], words[:, 1]


@functools.lru_cache(maxsize=16)
def coefficients(seed: int) -> tuple[Floats, Floats]:
    """Return the amplitudes ρ_k and phases θ_k for f_k = 1 … 20000 Hz (read-only)."""
    a, b = uniforms(seed)
    u = (a + np.uint64(1)).astype(float) * 2.0**-53  # (0, 1]
    v = b.astype(float) * 2.0**-53  # [0, 1)
    rho, theta = np.sqrt(-2.0 * np.log(u)), 2.0 * np.pi * v
    rho.flags.writeable = theta.flags.writeable = False
    return rho, theta


def band_power(lo: float, hi: float) -> Floats:
    """Return |H(f_k)|² for the band with half-power points lo and hi."""
    f = np.arange(1, COMPONENTS + 1, dtype=float)
    return 1.0 / ((1.0 + (lo / f) ** 4) * (1.0 + (f / hi) ** 4))


@functools.lru_cache(maxsize=32)
def amplitudes(seed: int, lo: float, hi: float) -> tuple[Floats, Floats]:
    """Return the calibrated component amplitudes γ|H(f_k)|ρ_k and phases θ_k (read-only)."""
    rho, theta = coefficients(seed)
    power = band_power(lo, hi)
    gamma = math.sqrt(0.5 / math.fsum(power))
    amp = gamma * np.sqrt(power) * rho
    amp.flags.writeable = False
    return amp, theta


def at(seed: int, lo: float, hi: float, t: ArrayLike, *, derivative: bool = False) -> Floats:
    """Return n(t), or n′(t), at arbitrary times by summing the components (for a few points, and for tests)."""
    amp, theta = amplitudes(seed, lo, hi)
    k = np.arange(1, COMPONENTS + 1, dtype=float)
    phase = 2 * np.pi * np.mod(np.outer(np.atleast_1d(t), k), 1.0) + theta
    out: Floats = -(np.sin(phase) @ (2 * np.pi * k * amp)) if derivative else np.cos(phase) @ amp
    return out


def _components(seed: int, lo: float, hi: float, rate: int) -> NDArray[np.complex128]:
    """Return the nonzero bins of the table's spectrum, 1 … 20000 Hz."""
    amp, theta = amplitudes(seed, lo, hi)
    phasor: NDArray[np.complex128] = np.exp(1j * theta)
    out: NDArray[np.complex128] = (rate / 2) * amp * phasor
    return out


def _check_rate(rate: int) -> None:
    """Refuse a table rate that is odd or not above 40 kHz."""
    if rate % 2 or rate <= 2 * COMPONENTS:
        msg = "a noise table needs an even rate above 40 kHz"
        raise ValueError(msg)


def table(seed: int, lo: float, hi: float, rate: int) -> Floats:
    """Return n(j / rate) for j = 0 … rate − 1, uncached (the reference renderer's large tables)."""
    _check_rate(rate)
    spectrum = np.zeros(rate // 2 + 1, dtype=complex)
    spectrum[1 : COMPONENTS + 1] = _components(seed, lo, hi, rate)
    out: Floats = scipy.fft.irfft(spectrum, n=rate)
    return out


_CACHE: dict[tuple[int, float, float, int], Floats] = {}  # insertion order is recency: the most recently used last
_LOCK = threading.Lock()
CACHE_TABLES = 8  # about 100 MB at the shipping rates


def tables(seed: int, bands: Iterable[tuple[float, float]], rate: int) -> list[Floats]:
    """Return one read-only table per band, computed in parallel and cached (plan §3)."""
    _check_rate(rate)
    keys = [(seed, float(lo), float(hi), rate) for lo, hi in bands]
    with _LOCK:
        found = {k: _CACHE[k] for k in keys if k in _CACHE}
    missing = sorted(set(keys) - found.keys())
    if missing:
        # Computed without the lock, so a render whose tables are cached never waits on another's. One
        # transform per table, in threads (the FFT releases the GIL): faster than a batched transform, and
        # a table never depends on which others it was computed with.
        with ThreadPoolExecutor(max_workers=min(len(missing), os.cpu_count() or 1)) as pool:
            for k, row in zip(missing, pool.map(lambda k: table(*k), missing), strict=True):
                row.flags.writeable = False
                found[k] = row
    with _LOCK:
        for k in keys:
            # A table another render cached meanwhile is kept, so one table is always one array; one
            # evicted meanwhile comes back.
            found[k] = _CACHE.pop(k, found[k])
            _CACHE[k] = found[k]
        while len(_CACHE) > max(CACHE_TABLES, len(set(keys))):
            del _CACHE[next(iter(_CACHE))]
    return [found[k] for k in keys]


def cached() -> list[tuple[int, float, float, int]]:
    """Return the cached tables' keys, (seed, low, high, rate), the least recently used first."""
    with _LOCK:
        return list(_CACHE)


def clear_cache() -> None:
    """Forget every cached table."""
    with _LOCK:
        _CACHE.clear()


def interpolate(table_: Floats, rate: int, t: ArrayLike) -> tuple[Floats, Floats]:
    """Return n(t) and n′(t) at arbitrary times, by the cubic through the four nearest table samples.

    The table is periodic with period 1 s. At the internal rates of 32× the
    error is at most about 2·10⁻⁶ of the noise's RMS value and 1.5·10⁻⁴ of its
    RMS slope, for the widest band (measured); the kink bound's 1% inflation
    covers it with room to spare.
    """
    times = np.atleast_1d(np.asarray(t, dtype=float))
    pos = np.mod(times, 1.0) * rate
    whole = np.floor(pos)
    i = whole.astype(np.int64)
    u = pos - whole
    y = [table_[(i + k) % rate] for k in (-1, 0, 1, 2)]
    # Lagrange weights on nodes −1, 0, 1, 2, and their derivatives with respect to u.
    w = (
        -u * (u - 1) * (u - 2) / 6,
        (u + 1) * (u - 1) * (u - 2) / 2,
        -(u + 1) * u * (u - 2) / 2,
        (u + 1) * u * (u - 1) / 6,
    )
    dw = (-(3 * u * u - 6 * u + 2) / 6, (3 * u * u - 4 * u - 1) / 2, -(3 * u * u - 2 * u - 2) / 2, (3 * u * u - 1) / 6)
    value: Floats = sum((wk * yk for wk, yk in zip(w, y, strict=True)), np.zeros_like(u))
    slope: Floats = sum((dk * yk for dk, yk in zip(dw, y, strict=True)), np.zeros_like(u)) * rate
    return value, slope


def dither(seed: int, count: int) -> Floats:
    """Return triangular (TPDF) dither in LSB for `count` samples, from SHAKE-256 of its own key and the seed.

    Each value is the difference of two 32-bit uniforms on [0, 1), so it lies in
    (−1, 1) with a triangular density: the dither that makes the first two
    moments of the quantization error independent of the signal. The sequence
    is prefix-stable, like the noise.
    """
    _check_seed(seed)
    stream = hashlib.shake_256(DITHER_KEY + seed.to_bytes(8, "little")).digest(8 * count)
    u = np.frombuffer(stream, dtype="<u4").reshape(count, 2).astype(float) * 2.0**-32
    return u[:, 0] - u[:, 1]
