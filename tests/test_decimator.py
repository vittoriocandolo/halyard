# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import hashlib
import math
from importlib import resources

import numpy as np
import pytest
from conftest import make
from scipy.signal import upfirdn

from halyard_synth import render as R

# (base, K, taps, length in ms, 99% of the energy within, in ms), as plan §3 records them.
FILTERS = [
    (44100, 16, 2593, 3.67, 0.85),
    (48000, 16, 1447, 1.88, 0.52),
    (44100, 32, 5183, 3.67, 0.85),
    (48000, 32, 2891, 1.88, 0.52),
    (44100, 64, 10363, 3.67, 0.85),
    (48000, 64, 5781, 1.88, 0.52),
]


@pytest.mark.parametrize("base, k, taps, length_ms, energy_ms", FILTERS)
def test_frozen_designs(base, k, taps, length_ms, energy_ms):
    d = R.decimator(base, k)
    h = d.taps
    rate = base * k
    assert h.size == taps
    assert 1000 * h.size / rate == pytest.approx(length_ms, abs=0.01)
    energy = np.cumsum(h**2) / np.sum(h**2)
    assert 1000 * np.searchsorted(energy, 0.99) / rate == pytest.approx(energy_ms, abs=0.01)
    n = 2**23
    H = np.abs(np.fft.rfft(h, n))
    f = np.arange(H.size) * rate / n
    assert np.max(np.abs(20 * np.log10(H[f <= 20000]))) <= 0.001
    assert -20 * np.log10(np.max(H[f >= base / 2])) >= 110
    assert d.stopband_db >= 110
    # DC gain exactly 1: the exactly rounded sum of the taps is 1.
    assert math.fsum(h) == 1.0
    # Minimum phase: the energy arrives first, so transients never pre-ring.
    assert np.argmax(np.abs(h)) < h.size // 20


@pytest.mark.parametrize("base, k", [(44100, 32), (48000, 32), (48000, 16)])
def test_checksum_is_enforced(base, k):
    entry = R.manifest()["filters"][f"{base}x{k}"]
    raw = resources.files("halyard_synth").joinpath("data", entry["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert np.array_equal(R.load_taps(raw, entry), R.decimator(base, k).taps)
    tampered = bytearray(raw)
    tampered[100] ^= 1
    with pytest.raises(R.RenderError):
        R.load_taps(bytes(tampered), entry)
    with pytest.raises(R.RenderError):
        R.load_taps(raw[:-8], entry)
    assert not R.decimator(base, k).taps.flags.writeable


def test_timing_contract_on_synthetic_input():
    h = R.decimator(48000).taps
    L, K = h.size, R.K
    rng = np.random.default_rng(2)
    for N in (1, 2, 31, 32, 33, 1000, 12345):
        y = rng.normal(size=N)
        out = upfirdn(h, y, down=K)
        assert out.size == (N + L - 2) // K + 1
        assert out[0] == h[0] * y[0]
        for m in {0, 1, out.size // 2, out.size - 1}:
            k = np.arange(L)
            n = m * K - k
            ok = (n >= 0) & (n < N)
            assert out[m] == pytest.approx(math.fsum(h[k[ok]] * y[n[ok]]), abs=1e-15)
        # The last output still sees the last input sample.
        assert (out.size - 1) * K - (N - 1) <= L - 1


@pytest.mark.parametrize("low, high", [(48000, 96000), (44100, 88200)])
def test_double_rate_is_the_base_rate_with_samples_filled_in(low, high):
    edits = {"drive.db": 12.0, "noise[2].rattle": 0.7}
    a, _ = R.render(make(**edits, **{"output.sample_rate": low}))
    b, _ = R.render(make(**edits, **{"output.sample_rate": high}))
    assert a.size == (b.size + 1) // 2
    assert np.array_equal(b[::2], a)


def test_rates():
    assert R.K == 32
    assert R.rates(44100) == (1411200, 32)
    assert R.rates(88200) == (1411200, 16)
    assert R.rates(48000) == (1536000, 32)
    assert R.rates(96000) == (1536000, 16)
    assert R.rates(48000, 16) == (768000, 16)


def test_parallel_decimation_is_bit_identical_to_upfirdn():
    rng = np.random.default_rng(3)
    for base in (44100, 48000):
        h = R.decimator(base).taps
        for n in (1, 33, 8191, 140001, 600000):
            y = rng.normal(size=n)
            for factor in (32, 16):
                reference = upfirdn(h, y, down=factor)
                for workers in (2, 3, 5):
                    assert np.array_equal(R.decimate(h, y, factor, workers=workers), reference)
