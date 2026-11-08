# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Monotone cubic Hermite curves: slopes, evaluation, integrals, derivatives, jumps and decay compression (§2.2)."""

import math
from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from halyard_synth.curve import Curve, amplitude_curve, compressed, pchip_slopes, peak_index, pitch_curve

type Floats = NDArray[np.float64]

FEW = 2  # a curve of two knots has one segment
NORMAL, MONOTONE, FLAT = 0, 1, 2  # kinds of random values; the fourth kind is an amplitude shape
RELATIVE = 1e-12  # agreement with an independent implementation, relative
SLICED = 1e-15  # the sliced integral against the integral at arbitrary times, relative
ROUND_DB = 20  # deviations of tails spaced this far apart are read to 0.1 dB, closer ones to 0.01 dB
PEAK_DB = 0.05  # the first segment's bulge after a peak, as measured
LONG_SPAN = 0.3  # shapes of this span or less are the short ones


def random_knots(rng: np.random.Generator) -> tuple[Floats, Floats]:
    """Draw knots: 2 to 32, of four kinds, from random values to amplitude shapes."""
    n = int(rng.integers(2, 33))
    t = np.cumsum(10 ** rng.uniform(-5, 0, n)) - 10 ** rng.uniform(-6, 0)
    kind = rng.integers(4)
    if kind == NORMAL:
        v = rng.normal(size=n) * 10 ** rng.uniform(-3, 3)
    elif kind == MONOTONE:
        v = np.sort(rng.uniform(0, 20, n))[:: rng.choice([1, -1])]
    elif kind == FLAT:  # flat runs and repeated values
        v = np.asarray(rng.choice([0.0, 0.5, 1.0, 2.0], n), dtype=float)
    else:  # amplitude-like shapes
        v = rng.uniform(0, 1, n)
        v[0] = v[-1] = 0.0
        v[rng.integers(1, n - 1) if n > FEW else 0] = 1.0
    return t, v


def random_curves(seed: int, count: int) -> Iterator[tuple[Floats, Floats, Curve]]:
    """Draw curves over random knots, with no end slope overridden, both, or the last."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        t, v = random_knots(rng)
        override = rng.integers(3)
        if override == 0:
            yield t, v, Curve.pchip(t, v)
        elif override == 1:
            yield t, v, Curve.pchip(t, v, first=0.0, last=0.0)
        else:
            yield t, v, Curve.pchip(t, v, last=0.0)


def test_slopes_match_scipy_pchip() -> None:
    """Find the PCHIP slopes as SciPy's, to 10⁻¹² of the steepest secant, over 2,000 knot sets."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        t, v = random_knots(rng)
        ours = pchip_slopes(t, v)
        theirs = PchipInterpolator(t, v).derivative()(t)
        scale = np.max(np.abs(np.diff(v) / np.diff(t)))
        np.testing.assert_allclose(ours, theirs, rtol=RELATIVE, atol=RELATIVE * scale)


def test_evaluation_matches_independent_hermite() -> None:
    """Evaluate curves as an independent Hermite spline does, and reproduce their knots exactly."""
    for t, v, c in random_curves(2, 500):
        x = np.linspace(t[0], t[-1], 997)
        ref = CubicHermiteSpline(t, v, c.d)(x)
        np.testing.assert_allclose(c(x), ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(v)) + 1e-300)
        assert np.array_equal(c(t), v)  # knots are reproduced exactly


def test_sliced_evaluation_is_the_same_curve() -> None:
    """Find the segment-by-segment evaluation on sorted times the same values as the evaluation at arbitrary times."""
    # Synthesis evaluates segment by segment on sorted times; it must give the very same values.
    rng = np.random.default_rng(14)
    for t, v, c in random_curves(15, 400):
        span = t[-1] - t[0]
        x = np.sort(np.concatenate((rng.uniform(t[0] - 0.3 * span, t[-1] + 0.3 * span, 3000), t)))
        assert np.array_equal(c.on(x), c(x))
        np.testing.assert_allclose(
            c.integral_on(x),
            c.integral(x),
            rtol=SLICED,
            atol=SLICED * (np.max(np.abs(v)) * 2 * span + abs(c.total)) + 1e-300,
        )
        assert np.array_equal(c.on(x[:0]), x[:0])


def test_constant_outside_knots() -> None:
    """Find a curve holding its end values outside its knots."""
    for t, v, c in random_curves(3, 200):
        span = t[-1] - t[0]
        assert np.all(c(t[0] - span * np.array([1e-9, 0.5, 3])) == v[0])
        assert np.all(c(t[-1] + span * np.array([1e-9, 0.5, 3])) == v[-1])


def test_segments_stay_in_their_knot_interval() -> None:
    """Find every segment monotone between its two knot values, before clipping and after."""
    # Monotonicity is a property of the slopes, so check it with an independent
    # evaluator that does not clip, then check that clipping removes nothing else.
    worst = 0.0
    for t, v, c in random_curves(4, 1000):
        s = np.linspace(0, 1, 257)
        for i in range(t.size - 1):
            x = np.clip(t[i] + s * (t[i + 1] - t[i]), t[i], t[i + 1])
            y = CubicHermiteSpline(t, c.v, c.d)(x)
            lo, hi = min(v[i], v[i + 1]), max(v[i], v[i + 1])
            scale = max(abs(v[i]), abs(v[i + 1]), 1e-300)
            worst = max(worst, (lo - y.min()) / scale, (y.max() - hi) / scale)
            ours = c(x)
            assert ours.min() >= lo
            assert ours.max() <= hi
    assert worst <= 16 * np.finfo(float).eps  # the independent evaluator's own rounding


def test_amplitude_shape_confined_to_unit_interval() -> None:
    """Find an amplitude shape flat at both ends and never outside 0 … 1."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(3, 33))
        t = np.cumsum(10 ** rng.uniform(-5, -1, n))
        v = rng.uniform(0, 1, n)
        v[0] = v[-1] = 0.0
        v[rng.integers(1, n - 1)] = 1.0
        c = amplitude_curve(t, v)
        assert c.d[0] == 0.0
        assert c.d[-1] == 0.0
        y = c.on(np.linspace(t[0] - 0.01, t[-1] + 0.01, 20001))
        assert y.min() >= 0.0
        assert y.max() <= 1.0


def test_pitch_extremes_only_at_knots() -> None:
    """Find a pitch curve flat at its end, its first slope kept, and its extremes at its knots."""
    rng = np.random.default_rng(6)
    for _ in range(300):
        n = int(rng.integers(2, 33))
        t = np.concatenate(([0.0], np.cumsum(10 ** rng.uniform(-5, -1, n - 1))))
        r = 10 ** rng.uniform(-0.5, 1.3, n)
        r[-1] = 1.0
        c = pitch_curve(t, r)
        assert c.d[-1] == 0.0
        assert c.d[0] == pchip_slopes(t, r)[0]  # the first slope is kept
        y = c(np.linspace(0, t[-1] * 1.5, 20001))
        assert y.max() <= r.max()
        assert y.min() >= r.min()


def _quad(c: Curve, a: float, b: float) -> float:
    """Integrate a curve from a to b by adaptive quadrature, splitting at its knots."""
    if a == b:
        return 0.0
    lo, hi = min(a, b), max(a, b)
    inside = [p for p in c.t if lo < p < hi]
    epsabs = 1e-14 * np.max(np.abs(c.v)) * (hi - lo)
    val: float = quad(lambda x: float(c(x)), lo, hi, points=inside or None, limit=400, epsabs=epsabs, epsrel=1e-13)[0]
    return val if b >= a else -val


def test_integrals_match_adaptive_quadrature() -> None:
    """Find the integrals from the first knot and to the last as adaptive quadrature does, outside the knots too."""
    rng = np.random.default_rng(7)
    for t, v, c in random_curves(8, 150):
        span = t[-1] - t[0]
        xs = np.concatenate((rng.uniform(t[0] - 0.5 * span, t[-1] + 0.5 * span, 3), [t[0], t[-1]]))
        scale = np.max(np.abs(v)) * (span * 2) + 1e-300
        for x in xs:
            assert abs(c.integral(x) - _quad(c, t[0], x)) <= 1e-12 * scale
            assert abs(c.remaining(x) - _quad(c, x, t[-1])) <= 1e-12 * scale


def test_integral_derivative_is_the_curve() -> None:
    """Find a segment's partial integral as Simpson's rule, exact for cubics; the two integrals sum to the total."""
    # Within a segment the integral equals Simpson's rule on the curve, which is exact for cubics.
    rng = np.random.default_rng(10)
    for t, v, c in random_curves(11, 300):
        i = rng.integers(t.size - 1, size=50)
        a = t[i] + rng.uniform(0, 1, 50) * (t[i + 1] - t[i])
        b = np.minimum(a + rng.uniform(0, 1, 50) * (t[i + 1] - a), t[i + 1])
        mid = (a + b) / 2
        simpson = (b - a) / 6 * (c(a) + 4 * c(mid) + c(b))
        slope = 3 * np.max(np.abs(np.diff(v) / np.diff(t))) + np.max(np.abs(c.d))
        mid_err = (b - a) * slope * np.spacing(np.abs(mid))
        got = c.integral(b) - c.integral(a)
        tol = (
            1e-13
            * (np.abs(c.integral(a)) + np.abs(c.integral(b)) + np.abs(simpson) + np.max(np.abs(v)) * (t[i + 1] - t[i]))
            + mid_err
            + 1e-300
        )
        assert np.all(np.abs(got - simpson) <= tol)
        x = np.linspace(t[0], t[-1], 101)
        np.testing.assert_allclose(
            c.integral(x) + c.remaining(x),
            c.total,
            rtol=1e-12,
            atol=1e-13 * abs(c.total) + 1e-300,
        )


def test_integral_grows_linearly_outside() -> None:
    """Find the integral growing by the end value outside the knots, at arbitrary and at sorted times."""
    c = pitch_curve([0.0, 0.01, 0.05], [8.0, 2.0, 1.0])
    for x in (0.05, 0.1, 1.0, 2.0):
        assert c.integral(x) == pytest.approx(c.total + (x - 0.05), rel=1e-15)
        assert c.integral_on(np.array([x]))[0] == pytest.approx(c.total + (x - 0.05), rel=1e-15)
    assert c.integral(-0.5) == pytest.approx(-4.0)


def test_rejects_bad_knots() -> None:
    """Refuse repeated knot times, and a single knot."""
    with pytest.raises(ValueError, match="knot times must increase"):
        Curve([0.0, 0.0], [1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="at least two knots"):
        Curve([0.0], [1.0], [0.0])


def test_derivatives_match_independent_hermite() -> None:
    """Find the first and second derivatives as an independent Hermite spline's, and 0 outside the knots."""
    for t, v, c in random_curves(12, 300):
        spline = CubicHermiteSpline(t, c.v, c.d)
        x = np.linspace(t[0], t[-1], 499)[:-1]
        scale = np.max(np.abs(np.diff(v) / np.diff(t))) * 3 + np.max(np.abs(c.d)) + 1e-300
        np.testing.assert_allclose(c.derivative(x), spline.derivative()(x), rtol=1e-9, atol=1e-9 * scale)
        assert np.all(c.derivative(t[-1] + np.array([1e-3, 1.0])) == 0)
        np.testing.assert_allclose(c.derivative(t), c.d, rtol=1e-9, atol=1e-9 * scale)
        scale2 = scale / np.min(np.diff(t)) * 10
        np.testing.assert_allclose(c.second(x), spline.derivative(2)(x), rtol=1e-8, atol=1e-9 * scale2)
        assert np.all(c.second(np.array([t[0] - 1.0, t[-1], t[-1] + 1.0])) == 0)


def test_jumps_are_the_one_sided_derivative_differences() -> None:
    """Find each knot's jumps in x″ and x‴ as the differences of the one-sided derivatives there."""
    for t, v, c in random_curves(13, 300):
        spline = CubicHermiteSpline(t, c.v, c.d)
        d2, d3 = spline.derivative(2), spline.derivative(3)
        j2, j3 = c.jumps()
        n = t.size
        for j in range(n):
            after2 = d2.c[-1, j] if j < n - 1 else 0.0
            before2 = d2.c[-1, j - 1] + d2.c[-2, j - 1] * (t[j] - t[j - 1]) if j > 0 else 0.0
            after3 = d3.c[-1, j] if j < n - 1 else 0.0
            before3 = d3.c[-1, j - 1] if j > 0 else 0.0
            scale = (np.max(np.abs(np.diff(v))) / np.min(np.diff(t)) ** 2) * 20 + 1e-300
            assert j2[j] == pytest.approx(after2 - before2, rel=1e-8, abs=1e-9 * scale)
            assert j3[j] == pytest.approx(after3 - before3, rel=1e-8, abs=1e-9 * scale / np.min(np.diff(t)))
            b2, a2, b3, a3 = c.sides(j)
            assert a2 - b2 == pytest.approx(j2[j], rel=1e-14, abs=1e-14 * scale)
            assert a3 - b3 == pytest.approx(j3[j], rel=1e-14, abs=1e-14 * scale / np.min(np.diff(t)))


@pytest.mark.parametrize(("spacing_db", "within_db"), [(6, 0.03), (10, 0.15), (20, 1.3)])
def test_points_every_few_db_reproduce_an_exponential_tail(spacing_db: float, within_db: float) -> None:
    """Find knots every 6, 10 and 20 dB following an exponential tail within the plan's 0.03, 0.15 and 1.3 dB."""
    # Plan §2.2 authoring guidance, on the interior of a tail.
    tau = 0.1
    step = spacing_db * math.log(10) / 20 * tau
    knots = 0.001 + step * np.arange(12)
    c = amplitude_curve(
        np.concatenate(([0.0], knots, [knots[-1] + step])),
        np.concatenate(([0.0], np.exp(-(knots - 0.001) / tau), [0.0])),
    )
    x = np.linspace(knots[1], knots[-3], 100001)
    deviation = 20 * np.log10(c(x) / np.exp(-(x - 0.001) / tau))
    assert round(float(np.max(np.abs(deviation))), 2 if spacing_db < ROUND_DB else 1) <= within_db


@pytest.mark.parametrize(("spacing_db", "above_db"), [(6, 1.1), (10, 2.0), (20, 5.1)])
def test_the_first_segment_after_a_peak_is_rounder(spacing_db: float, above_db: float) -> None:
    """Find the first segment after a peak above the exponential by 1.1, 2 or 5.1 dB: a crack needs a shoulder knot."""
    # The peak's slope is zero, so the decay starts flat (plan §2.2): a crack needs a shoulder knot.
    tau = 0.1
    step = spacing_db * math.log(10) / 20 * tau
    knots = 0.001 + step * np.arange(12)
    c = amplitude_curve(
        np.concatenate(([0.0], knots, [knots[-1] + step])),
        np.concatenate(([0.0], np.exp(-(knots - 0.001) / tau), [0.0])),
    )
    x = np.linspace(knots[0], knots[1], 100001)
    above = 20 * np.log10(c(x) / np.exp(-(x - 0.001) / tau))
    assert float(np.max(above)) == pytest.approx(above_db, abs=PEAK_DB)


# Decay compression.


def random_shape(rng: np.random.Generator, span: float = 2.0) -> Curve:
    """Draw an amplitude shape on microsecond knots, sometimes with a plateau at its peak."""
    n = int(rng.integers(3, 33))
    h = 10 ** rng.uniform(-5, 0, n - 1)
    t = np.concatenate(([0.0], np.cumsum(h)))
    t *= min(1.0, span / t[-1])
    t = np.round(t * 1e6) / 1e6  # microsecond knots, as a patch writes them
    t = np.maximum.accumulate(t + np.arange(n) * 1e-6)
    v = rng.uniform(0, 1, n)
    v[0] = v[-1] = 0.0
    v[rng.integers(1, n - 1)] = 1.0
    if rng.random() < LONG_SPAN:  # a plateau at the peak, three times in ten
        k = int(np.flatnonzero(v == 1.0)[0])
        v[min(k + 1, n - 2)] = 1.0
    return amplitude_curve(t, v)


@pytest.mark.parametrize(("span", "s_max"), [(0.3, 20.0), (2.0, 512.0)])
def test_compressed_decay_is_the_shape_at_compressed_time(span: float, s_max: float) -> None:
    """Find a compressed shape equal to the shape at the compressed time, to the rounding of that time."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        e = random_shape(rng, span)
        s = s_max ** rng.uniform(0, 1)
        c = compressed(e, s)
        tpk = e.t[peak_index(e)]
        x = np.sort(np.concatenate((rng.uniform(-0.01, e.t[-1] * 1.1, 500), c.t)))
        tau = np.where(x <= tpk, x, tpk + s * (x - tpk))
        # Any representation carries the rounding of the compressed time, a few ulps of the time
        # times s, through the shape's slope, which is at most 3 times its steepest secant.
        slope = 3 * np.max(np.abs(np.diff(e.v)) / e.h)
        tol = 1e-15 + 8 * np.finfo(float).eps * (np.abs(x) + tpk) * s * slope
        assert np.all(np.abs(c(x) - e(tau)) <= tol)
        np.testing.assert_allclose(c.on(x), c(x), rtol=0, atol=0)


def test_compressed_shapes_keep_their_peak_and_monotone_segments() -> None:
    """Find a compressed shape keeping its peak at exactly 1, its segments monotone and within 0 … 1."""
    rng = np.random.default_rng(6)
    for _ in range(300):
        e = random_shape(rng)
        c = compressed(e, 8 ** rng.uniform(0, 3))
        k = peak_index(c)
        assert k == peak_index(e)
        assert c.v[k] == 1.0
        assert c(c.t[k]) == 1.0
        assert np.all(np.diff(c.t) > 0)
        assert c.t[-1] <= e.t[-1]
        for i in range(c.t.size - 1):
            x = np.linspace(c.t[i], c.t[i + 1], 101)
            y = c(x)
            assert np.all(np.diff(y) * np.sign(c.v[i + 1] - c.v[i]) >= 0)
            assert np.all((y >= 0) & (y <= 1))


def test_compressed_jumps_scale_by_s_squared_and_cubed() -> None:
    """Find the jumps after the peak scaled by s² and s³, and at the peak the rise's kept."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        e = random_shape(rng, span=0.3)
        s = 8 ** rng.uniform(0, 3)
        c = compressed(e, s)
        k = peak_index(e)
        for j in range(e.t.size):
            b2, a2, b3, a3 = e.sides(j)
            if j > k:
                b2, a2, b3, a3 = b2 * s**2, a2 * s**2, b3 * s**3, a3 * s**3
            elif j == k:  # the rise keeps its own derivatives, and the decay's are scaled
                a2, a3 = a2 * s**2, a3 * s**3
            got = c.sides(j)
            scale = max(abs(b2), abs(a2), abs(b3) * np.min(c.h), abs(a3) * np.min(c.h), 1e-300)
            np.testing.assert_allclose(got[:2], (b2, a2), rtol=1e-7, atol=1e-9 * scale)
            np.testing.assert_allclose(got[2:], (b3, a3), rtol=1e-7, atol=1e-9 * scale / np.min(c.h))


def test_compression_by_one_is_the_shape_itself() -> None:
    """Return the shape itself for a compression of 1."""
    e = amplitude_curve([0.0, 0.0004, 0.025, 0.18], [0.0, 1.0, 0.45, 0.0])
    assert compressed(e, 1.0) is e


def test_compression_needs_a_flat_peak() -> None:
    """Refuse to compress a shape whose slope at its peak is not 0."""
    bent = Curve([0.0, 0.001, 0.01], [0.0, 1.0, 0.0], [0.0, 5.0, 0.0])
    with pytest.raises(ValueError, match="zero slope at the peak"):
        compressed(bent, 2.0)


@pytest.mark.parametrize("gain", [5e-324, 1e-320, 2.2250738585072014e-308])
def test_a_subnormal_knot_has_a_slope_within_rounding_of_zero(gain: float) -> None:
    """Find the slope at a knot a subnormal gain away from its neighbours' 0, where a reciprocal overflows."""
    d = pchip_slopes([0.0, 1.0, 2.0, 3.0], [0.0, gain, 2 * gain, 0.0])
    assert np.all(np.isfinite(d))
    assert np.all(np.abs(d) <= 3 * 2 * gain)  # the shape-preserving rules: at most three times the steepest secant
