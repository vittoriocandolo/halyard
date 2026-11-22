# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The frozen decimators: their designs, checksums, timing, the 2× rates and parallel decimation."""

import hashlib
import math
from importlib import resources

import numpy as np
import pytest
from scipy.signal import upfirdn

from halyard_synth import render

from .conftest import make

# (base, K, taps, length in ms, 99% of the energy within, in ms), as docs/validation.md records them.
FILTERS = [
    (44100, 16, 2593, 3.67, 0.85),
    (48000, 16, 1447, 1.88, 0.52),
    (44100, 32, 5183, 3.67, 0.85),
    (48000, 32, 2891, 1.88, 0.52),
    (44100, 64, 10363, 3.67, 0.85),
    (48000, 64, 5781, 1.88, 0.52),
]
PASS_HZ = 20000  # the passband's edge
PASSBAND_DB = 0.001  # flat within this up to it
STOPBAND_DB = 110  # and at least this far down from half the base rate
MS = 0.01  # the lengths as docs/validation.md records them, rounded
EARLY = 20  # minimum phase: the largest tap lies in the first twentieth
ROUNDING = 1e-15  # a sum of a few hundred products, against its exactly rounded value


@pytest.mark.parametrize(("base", "k", "taps", "length_ms", "energy_ms"), FILTERS)
def test_frozen_designs(base: int, k: int, taps: int, length_ms: float, energy_ms: float) -> None:
    """Find each shipped design as docs/validation.md records it: flat, deep, of unit DC gain and minimum phase."""
    d = render.decimator(base, k)
    h = d.taps
    rate = base * k
    assert h.size == taps
    assert 1000 * h.size / rate == pytest.approx(length_ms, abs=MS)
    energy = np.cumsum(h**2) / np.sum(h**2)
    assert 1000 * np.searchsorted(energy, 0.99) / rate == pytest.approx(energy_ms, abs=MS)
    n = 2**23
    response = np.abs(np.fft.rfft(h, n))
    f = np.arange(response.size) * rate / n
    assert np.max(np.abs(20 * np.log10(response[f <= PASS_HZ]))) <= PASSBAND_DB
    assert -20 * np.log10(np.max(response[f >= base / 2])) >= STOPBAND_DB
    assert d.stopband_db >= STOPBAND_DB
    # DC gain exactly 1: the exactly rounded sum of the taps is 1.
    assert math.fsum(h) == 1.0
    # Minimum phase: the energy arrives first, so transients never pre-ring.
    assert np.argmax(np.abs(h)) < h.size // EARLY


@pytest.mark.parametrize(("base", "k"), [(44100, 32), (48000, 32), (48000, 16)])
def test_checksum_is_enforced(base: int, k: int) -> None:
    """Load the shipped taps only when they match their checksum and length, and read-only."""
    entry = render.manifest()["filters"][f"{base}x{k}"]
    raw = resources.files("halyard_synth").joinpath("data", entry["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert np.array_equal(render.load_taps(raw, entry), render.decimator(base, k).taps)
    tampered = bytearray(raw)
    tampered[100] ^= 1
    with pytest.raises(render.RenderError):
        render.load_taps(bytes(tampered), entry)
    with pytest.raises(render.RenderError):
        render.load_taps(raw[:-8], entry)
    assert not render.decimator(base, k).taps.flags.writeable


def test_timing_contract_on_synthetic_input() -> None:
    """Find out[m] = Σ h[k] y[mK − k] with no delay or trimming, the whole filter tail kept, at any length."""
    h = render.decimator(48000).taps
    length, factor = h.size, render.K
    rng = np.random.default_rng(2)
    for count in (1, 2, 31, 32, 33, 1000, 12345):
        y = rng.normal(size=count)
        out = upfirdn(h, y, down=factor)
        assert out.size == (count + length - 2) // factor + 1
        assert out[0] == h[0] * y[0]
        for m in {0, 1, out.size // 2, out.size - 1}:
            k = np.arange(length)
            n = m * factor - k
            ok = (n >= 0) & (n < count)
            assert out[m] == pytest.approx(math.fsum(h[k[ok]] * y[n[ok]]), abs=ROUNDING)
        # The last output still sees the last input sample.
        assert (out.size - 1) * factor - (count - 1) <= length - 1


@pytest.mark.parametrize(
    ("low", "high"),
    [(48000, 96000), (44100, 88200), (48000, 192000), (44100, 176400)],
)
def test_higher_rates_are_the_base_rate_with_samples_filled_in(low: int, high: int) -> None:
    """Render at 2× and 4× the base rate and find the base rate's render as every 2nd or 4th sample, bit for bit."""
    edits = {"drive.db": 12.0, "noise[2].rattle": 0.7}
    a, _ = render.render(make(**edits, **{"output.sample_rate": low}))
    b, _ = render.render(make(**edits, **{"output.sample_rate": high}))
    step = high // low
    assert a.size == (b.size + step - 1) // step
    assert np.array_equal(b[::step], a)


def test_rates() -> None:
    """Find each export rate's internal rate and factor: K = 32 × the base, a half and a quarter of it at 2× and 4×."""
    assert render.rates(48000)[0] == render.K * 48000
    assert render.rates(44100) == (1411200, 32)
    assert render.rates(88200) == (1411200, 16)
    assert render.rates(48000) == (1536000, 32)
    assert render.rates(96000) == (1536000, 16)
    assert render.rates(176400) == (1411200, 8)
    assert render.rates(192000) == (1536000, 8)
    assert render.rates(48000, 16) == (768000, 16)


def test_parallel_decimation_is_bit_identical_to_upfirdn() -> None:
    """Decimate in 2, 3 and 5 parallel chunks and find one upfirdn call's output, bit for bit, at any length."""
    rng = np.random.default_rng(3)
    for base in (44100, 48000):
        h = render.decimator(base).taps
        for n in (1, 33, 8191, 140001, 600000):
            y = rng.normal(size=n)
            for factor in (32, 16):
                reference = upfirdn(h, y, down=factor)
                for workers in (2, 3, 5):
                    assert np.array_equal(render.decimate(h, y, factor, workers=workers), reference)
