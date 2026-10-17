import hashlib
import math
import threading

import numpy as np
import pytest
from scipy import stats

from snare import noise

WIDEST = (20.0, 20000.0)
WIRES = (2500.0, 9000.0)


def direct(seed, lo, hi, j, rate):
    """n(j / rate) by summing the components, with the phase reduced exactly in integers."""
    amp, theta = noise.amplitudes(seed, lo, hi)
    k = np.arange(1, noise.COMPONENTS + 1, dtype=np.int64)
    return np.array([math.fsum(amp * np.cos(2 * np.pi * ((k * int(jj)) % rate / rate) + theta)) for jj in j])


def direct_slope(seed, lo, hi, j, rate):
    """n′(j / rate), likewise."""
    amp, theta = noise.amplitudes(seed, lo, hi)
    k = np.arange(1, noise.COMPONENTS + 1, dtype=np.int64)
    return np.array(
        [-math.fsum(2 * np.pi * k * amp * np.sin(2 * np.pi * ((k * int(jj)) % rate / rate) + theta)) for jj in j]
    )


def rms_slope(seed, lo, hi):
    amp, _ = noise.amplitudes(seed, lo, hi)
    k = np.arange(1, noise.COMPONENTS + 1)
    return math.sqrt(np.sum((2 * np.pi * k * amp) ** 2) / 2)


@pytest.fixture
def fresh_cache():
    noise._CACHE.clear()
    yield
    noise._CACHE.clear()


# The integer stage and the coefficients.


def test_integer_stage_is_pinned():
    # It depends only on FIPS 202 and our key, so it is exact everywhere, and it may never change.
    a, b = noise.uniforms(1)
    assert [int(x) for x in a[:3]] == [7722249428135383, 2233965327905589, 1986645927032010]
    assert [int(x) for x in b[:3]] == [7559645094285810, 779071378434091, 8356828306956370]
    a, b = noise.uniforms(2**63 - 1)
    assert [int(x) for x in a[:2]] == [8483661322644481, 5277765733941658]
    assert [int(x) for x in b[:2]] == [1925678445488676, 3621218641772745]


def test_coefficients_follow_from_the_integers():
    a, b = noise.uniforms(7)
    rho, theta = noise.coefficients(7)
    for i in range(0, noise.COMPONENTS, 997):
        assert rho[i] == pytest.approx(math.sqrt(-2 * math.log((int(a[i]) + 1) / 2**53)), rel=1e-15)
        assert theta[i] == pytest.approx(2 * math.pi * int(b[i]) / 2**53, rel=1e-15, abs=1e-300)
    assert np.all(np.isfinite(rho)) and np.all(rho >= 0)
    assert np.all((theta >= 0) & (theta < 2 * np.pi))
    assert not rho.flags.writeable and not theta.flags.writeable


def test_stream_is_prefix_stable_and_repeatable():
    key = noise.KEY + (123).to_bytes(8, "little")
    short = hashlib.shake_256(key).digest(16 * 100)
    long = hashlib.shake_256(key).digest(16 * noise.COMPONENTS)
    assert long[: len(short)] == short
    assert np.array_equal(noise.table(123, *WIRES, 48000), noise.table(123, *WIRES, 48000))


def test_seed_range():
    for bad in (-1, 2**63):
        with pytest.raises(ValueError):
            noise.uniforms(bad)
        with pytest.raises(ValueError):
            noise.dither(bad, 4)


def test_calibration_is_on_the_expected_power():
    # γ² Σ|H|² = ½ exactly, whatever the realization: realizations are never renormalized.
    for lo, hi in (WIDEST, WIRES, (20.0, 21.0), (19999.0, 20000.0)):
        amp, _ = noise.amplitudes(5, lo, hi)
        rho, _ = noise.coefficients(5)
        assert math.fsum((amp / rho) ** 2) == pytest.approx(0.5, rel=1e-13)


def test_band_edges_are_half_power_points():
    power = noise.band_power(100.0, 10000.0)
    assert power[100 - 1] == pytest.approx(0.5, rel=1e-7)
    assert power[10000 - 1] == pytest.approx(0.5, rel=1e-7)


# Tables sample the continuous definition exactly.


def test_table_samples_the_continuous_definition():
    rate = 48000
    j = np.array([0, 1, 17, 4801, 23999, 47999])
    x = noise.table(5, 300.0, 12000.0, rate)
    np.testing.assert_allclose(x[j], direct(5, 300.0, 12000.0, j, rate), rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    "r1, r2, step1, step2",
    [
        (1411200, 1536000, 147, 160),  # the two families at 32×: common instants every 1/9600 s
        (768000, 1536000, 1, 2),
        (1536000, 6144000, 1, 4),  # an internal rate and the reference rate
    ],
)
def test_common_instants_agree_across_rates(r1, r2, step1, step2):
    x1 = noise.table(9, *WIRES, r1)[::step1]
    x2 = noise.table(9, *WIRES, r2)[::step2]
    assert x1.size == x2.size
    assert np.max(np.abs(x1 - x2)) <= 1e-12


def test_rate_requirements():
    for bad in (40000, 48001):
        with pytest.raises(ValueError):
            noise.table(1, *WIRES, bad)
        with pytest.raises(ValueError):
            noise.tables(1, [WIRES], bad)


@pytest.mark.parametrize("lo, hi", [(20.0, 21.0), WIDEST, (19999.0, 20000.0), (5000.0, 5001.0)])
def test_extreme_bands_are_well_behaved(lo, hi):
    x = noise.table(1, lo, hi, 48000)
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) < 10


# Statistics.


def test_ensemble_mean_square_is_calibrated():
    ms = np.array([np.mean(noise.table(seed, *WIRES, 48000) ** 2) for seed in range(200)])
    assert abs(ms.mean() / 0.5 - 1) <= 0.005
    # The spread of a realization's power is that of Σ|H|²ρ² with E ρ⁴ = 8.
    power = noise.band_power(*WIRES)
    predicted = math.sqrt(np.sum(power**2)) / np.sum(power)
    assert ms.std(ddof=1) / 0.5 == pytest.approx(predicted, rel=0.2)


def test_ensemble_spectrum_follows_the_band_shape():
    rate, (lo, hi) = 48000, WIRES
    power = noise.band_power(lo, hi)
    gamma2 = 0.5 / np.sum(power)
    ratio = np.zeros(noise.COMPONENTS)
    for seed in range(200):
        X = np.fft.rfft(noise.table(seed, lo, hi, rate))[1 : noise.COMPONENTS + 1] / (rate / 2)
        ratio += np.abs(X) ** 2 / (gamma2 * power) / 2  # E ρ² = 2
    ratio /= 200
    # Pooling 100 components per band gives each band mean a standard error of 1/√20000.
    bands = ratio.reshape(200, 100).mean(axis=1)
    assert np.max(np.abs(bands - 1)) <= 6 / math.sqrt(20000)
    # Nothing at DC or above 20 kHz.
    full = np.fft.rfft(noise.table(0, lo, hi, rate))
    assert np.max(np.abs(full[noise.COMPONENTS + 1 :])) <= 1e-9 * np.max(np.abs(full))
    assert abs(full[0]) <= 1e-9 * np.max(np.abs(full))


LOW, HIGH = (100.0, 400.0), (6400.0, 20000.0)


@pytest.fixture(scope="module")
def ensemble():
    """Each layer's value at one instant, for seeds 0 … 1999."""
    rate, j = 48000, 12345
    k = np.arange(1, noise.COMPONENTS + 1, dtype=np.int64)
    phase = 2 * np.pi * ((k * j) % rate / rate)
    weights = {}
    for band in (WIRES, LOW, HIGH):
        power = noise.band_power(*band)
        weights[band] = math.sqrt(0.5 / math.fsum(power)) * np.sqrt(power)
    values = {band: [] for band in weights}
    for seed in range(2000):
        rho, theta = noise.coefficients(seed)
        c = rho * np.cos(phase + theta)
        for band, w in weights.items():
            values[band].append(w @ c)
    return {band: np.array(v) for band, v in values.items()}


def test_marginal_is_gaussian(ensemble):
    values = ensemble[WIRES]
    assert stats.kstest(values, stats.norm(scale=math.sqrt(0.5)).cdf).pvalue > 1e-3
    assert abs(values.mean()) <= 5 * math.sqrt(0.5 / values.size)


# Coherent layers.


def test_identical_bands_give_bit_identical_layers(fresh_cache):
    a, b, c = noise.tables(3, [WIRES, (1500.0, 16000.0), WIRES], 48000)
    assert a is c
    assert np.array_equal(a, noise.table(3, *WIRES, 48000))
    assert not np.array_equal(a, b)


def test_a_table_never_depends_on_its_companions(fresh_cache):
    bands = [WIDEST, WIRES, (1500.0, 16000.0), (800.0, 2500.0)]
    together = noise.tables(11, bands, 1536000)
    for band, t in zip(bands, together):
        noise._CACHE.clear()
        (alone,) = noise.tables(11, [band], 1536000)
        assert np.array_equal(alone, t)


def test_band_edits_reweight_the_same_noise():
    a = noise.table(3, 2500.0, 9000.0, 48000)
    b = noise.table(3, 2500.0, 9500.0, 48000)
    assert np.corrcoef(a, b)[0, 1] >= 0.999
    # Same phases, and amplitudes in the ratio of the bands alone, whatever the realization.
    amp_a, theta_a = noise.amplitudes(3, *WIRES)
    amp_b, theta_b = noise.amplitudes(3, 100.0, 15000.0)
    assert np.array_equal(theta_a, theta_b)
    p_a, p_b = noise.band_power(*WIRES), noise.band_power(100.0, 15000.0)
    expected = np.sqrt(p_a / p_b * (math.fsum(p_b) / math.fsum(p_a)))
    np.testing.assert_allclose(amp_a / amp_b, expected, rtol=1e-12)


def test_layers_correlate_exactly_by_their_spectral_overlap():
    # Over one period the cross terms of distinct components vanish, so the correlation of two
    # layers is exactly the ρ²-weighted overlap of their bands, for every realization.
    rate = 48000
    for seed in (1, 2):
        for b1, b2 in ((WIRES, (1500.0, 16000.0)), (WIRES, (5000.0, 20000.0)), ((100.0, 400.0), (6400.0, 20000.0))):
            x1, x2 = noise.table(seed, *b1, rate), noise.table(seed, *b2, rate)
            a1, _ = noise.amplitudes(seed, *b1)
            a2, _ = noise.amplitudes(seed, *b2)
            assert np.mean(x1 * x2) == pytest.approx(math.fsum(a1 * a2) / 2, rel=1e-11)


def test_disjoint_bands_are_uncorrelated_within_their_skirts(ensemble):
    # Across the ensemble, at any instant, two layers correlate by their bands' overlap only.
    p1, p2 = noise.band_power(*LOW), noise.band_power(*HIGH)
    overlap = np.sum(np.sqrt(p1 * p2)) / math.sqrt(np.sum(p1) * np.sum(p2))
    assert overlap < 0.025  # second-order skirts: bands four octaves apart share about 2% of their power
    r = np.corrcoef(ensemble[LOW], ensemble[HIGH])[0, 1]
    assert abs(r - overlap) <= 5 / math.sqrt(ensemble[LOW].size)


# Values and slopes between samples.


def test_direct_evaluation_and_derivative():
    rate = 48000
    x = noise.table(4, *WIRES, rate)
    j = np.array([0, 11, 2400, 47999])
    np.testing.assert_allclose(noise.at(4, *WIRES, j / rate), x[j], atol=1e-11)
    np.testing.assert_allclose(
        noise.at(4, *WIRES, j / rate, derivative=True),
        direct_slope(4, *WIRES, j, rate),
        atol=1e-11 * rms_slope(4, *WIRES),
    )


@pytest.mark.parametrize("rate", [1411200, 1536000])
def test_interpolation_is_accurate_at_the_internal_rates(rate):
    # The kink bound takes carriers at knots from the tables and is inflated by 1% to cover this.
    rng = np.random.default_rng(0)
    t = np.concatenate((rng.uniform(0, 1, 400), [0.0, 1e-9, 0.99999999, 0.5]))
    for lo, hi in (WIDEST, WIRES):
        (tab,) = noise.tables(2, [(lo, hi)], rate)
        value, slope = noise.interpolate(tab, rate, t)
        assert np.max(np.abs(value - noise.at(2, lo, hi, t))) <= 2e-6 * math.sqrt(0.5)
        assert np.max(np.abs(slope - noise.at(2, lo, hi, t, derivative=True))) <= 1.5e-4 * rms_slope(2, lo, hi)


def test_interpolation_is_exact_at_samples_and_periodic():
    rate = 48000
    tab = noise.table(6, *WIRES, rate)
    j = np.array([0, 5, 47999])
    value, _ = noise.interpolate(tab, rate, j / rate)
    assert np.array_equal(value, tab[j])
    t = np.array([0.2, 0.7777])
    np.testing.assert_allclose(noise.interpolate(tab, rate, t + 1)[0], noise.interpolate(tab, rate, t)[0], atol=1e-12)


# Dither.


def test_dither_is_triangular_and_reproducible():
    d = noise.dither(3, 200000)
    assert np.array_equal(d, noise.dither(3, 200000))
    assert np.array_equal(d[:1000], noise.dither(3, 1000))
    assert not np.array_equal(d[:1000], noise.dither(4, 1000))
    assert np.all(np.abs(d) < 1)
    assert abs(d.mean()) <= 5 * math.sqrt(1 / 6 / d.size)
    assert d.var() == pytest.approx(1 / 6, rel=0.01)
    assert stats.kstest(d, stats.triang(c=0.5, loc=-1, scale=2).cdf).pvalue > 1e-3
    assert noise.dither(3, 0).size == 0


def test_dither_has_its_own_stream():
    stream = hashlib.shake_256(noise.DITHER_KEY + (1).to_bytes(8, "little")).digest(8)
    u = np.frombuffer(stream, dtype="<u4")
    assert noise.dither(1, 1)[0] == (int(u[0]) - int(u[1])) / 2**32
    assert noise.DITHER_KEY != noise.KEY


# The cache.


def test_cache_returns_the_same_read_only_tables(fresh_cache):
    first = noise.tables(1, [WIRES, WIDEST], 48000)
    again = noise.tables(1, [WIDEST, WIRES], 48000)
    assert again[0] is first[1] and again[1] is first[0]
    assert all(not t.flags.writeable for t in first)
    with pytest.raises(ValueError):
        first[0][0] = 1.0


def test_cache_evicts_the_least_recently_used(fresh_cache):
    n = noise.CACHE_TABLES
    keep = noise.tables(1, [WIRES], 48000)[0]
    for i in range(n + 2):
        noise.tables(1, [(100.0 + i, 5000.0)], 48000)
        assert noise.tables(1, [WIRES], 48000)[0] is keep  # used every time, so never evicted
    assert len(noise._CACHE) == n
    assert (1, 100.0, 5000.0, 48000) not in noise._CACHE
    # A render with more layers than the cache holds still gets all of them.
    bands = [(200.0 + i, 5000.0) for i in range(n + 3)]
    out = noise.tables(1, bands, 48000)
    assert len(out) == n + 3 and len(noise._CACHE) == n + 3
    noise.tables(1, [WIRES], 48000)
    assert len(noise._CACHE) == n


def test_cache_is_safe_across_threads(fresh_cache):
    results = []

    def work():
        results.append(noise.tables(8, [WIRES, WIDEST], 48000))

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 6
    assert all(r[0] is results[0][0] and r[1] is results[0][1] for r in results)
