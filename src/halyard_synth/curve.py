# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Monotone cubic Hermite curves with exact integrals (docs/design.md: Curves).

A curve passes through its knots with slopes from the PCHIP rule:

* interior knots take the weighted harmonic mean of the adjacent secants
  (Fritsch–Butland), or zero where the secants differ in sign or one is zero;
* end knots take the shape-preserving one-sided three-point rule.

End-slope overrides are applied before the curve is evaluated or integrated.
Every slope the rule or an override can produce keeps both normalized end
slopes of each segment in [0, 3], inside the Fritsch–Carlson monotonicity
region, so each segment is monotone and stays between its two knot values.
Evaluation clips to that interval, which removes only rounding excursions.

Outside its knots a curve holds its end values, and its integral grows
linearly there. Times are in seconds.

Synthesis evaluates curves on sorted time arrays, one segment at a time
(`on`), so each sample costs a few multiplications and no lookup.

A body mode's envelope is its shape with the decay compressed (`compressed`),
which is itself a curve of this kind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    type Floats = NDArray[np.float64]
    type Values = Floats | float  # arrays over many times, or one segment's numbers

MIN_KNOTS = 2  # a curve's fewest knots, for one segment; patches never give fewer


TWO_KNOTS = 2  # the fewest a curve has: one segment, whose ends share its secant


def pchip_slopes(times: ArrayLike, values: ArrayLike) -> Floats:
    """Return the knot slopes from the PCHIP rule, before any override."""
    t = np.asarray(times, dtype=float)
    if t.size < MIN_KNOTS:
        msg = f"a curve needs at least {MIN_KNOTS} knots, found {t.size}"
        raise ValueError(msg)
    v = np.asarray(values, dtype=float)
    h = np.diff(t)
    m = np.diff(v) / h
    if t.size == TWO_KNOTS:
        return np.array([m[0], m[0]])
    ml, mr, hl, hr = m[:-1], m[1:], h[:-1], h[1:]
    agree = (np.sign(ml) == np.sign(mr)) & (ml != 0) & (mr != 0)
    w1, w2 = 2 * hr + hl, hr + 2 * hl
    # Where a secant is 0 the mean is not used; where one is subnormal its reciprocal overflows to ∞ and the mean,
    # at most three times the smaller secant, comes out 0, within rounding of its exact value.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean = (w1 + w2) / (w1 / ml + w2 / mr)
    d = np.empty(t.size)
    d[1:-1] = np.where(agree, mean, 0.0)
    d[0] = _end_slope(h[0], h[1], m[0], m[1])
    d[-1] = _end_slope(h[-1], h[-2], m[-1], m[-2])
    return d


def _end_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    """Return an end knot's slope: the one-sided three-point rule, kept shape-preserving."""
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3 * abs(m0):
        return 3 * m0
    return float(d)


class Segment(NamedTuple):
    """The segments under some times: their lengths, and their end values and slopes."""

    h: Values
    va: Values
    vb: Values
    da: Values
    db: Values


class Curve:
    """Cubic Hermite curve through knots (t, v) with knot slopes d."""

    def __init__(self, times: ArrayLike, values: ArrayLike, slopes: ArrayLike) -> None:
        """Make the curve through the knots, and its integrals and derivative jumps once."""
        self.t = np.array(times, dtype=float)
        self.v = np.array(values, dtype=float)
        self.d = np.array(slopes, dtype=float)
        if not (self.t.ndim == 1 and self.t.size >= TWO_KNOTS and self.v.shape == self.t.shape == self.d.shape):
            msg = "a curve needs at least two knots with one value and one slope each"
            raise ValueError(msg)
        self.h = np.diff(self.t)
        if not np.all(self.h > 0):
            msg = "knot times must increase"
            raise ValueError(msg)
        v, d, h = self.v, self.d, self.h
        full = h * (v[:-1] + v[1:]) / 2 + h * h * (d[:-1] - d[1:]) / 12
        # Integral from the first knot to knot i, and from knot i to the last knot.
        self._before = np.concatenate(([0.0], np.cumsum(full)))
        self._after = np.concatenate((np.cumsum(full[::-1])[::-1], [0.0]))
        m = np.diff(v) / h
        self._second_start = (6 * m - 4 * d[:-1] - 2 * d[1:]) / h  # second derivative just after each segment starts
        self._second_end = (-6 * m + 2 * d[:-1] + 4 * d[1:]) / h  # and just before it ends
        self._third = 6 * (d[:-1] + d[1:] - 2 * m) / h**2  # constant on each segment
        for a in (
            self.t,
            self.v,
            self.d,
            self.h,
            self._before,
            self._after,
            self._second_start,
            self._second_end,
            self._third,
        ):
            a.flags.writeable = False

    @classmethod
    def pchip(cls, t: ArrayLike, v: ArrayLike, *, first: float | None = None, last: float | None = None) -> Curve:
        """Make the PCHIP curve; `first` and `last` override the end slopes."""
        d = pchip_slopes(t, v)
        if first is not None:
            d[0] = first
        if last is not None:
            d[-1] = last
        return cls(t, v, d)

    @property
    def total(self) -> float:
        """The integral over the knot span."""
        return float(self._before[-1])

    # Evaluation at arbitrary times.

    def _segment(self, x: Floats) -> tuple[NDArray[np.intp], Segment]:
        """Return the segment under each time, and its length, end values and end slopes."""
        i = np.clip(np.searchsorted(self.t, x, side="right") - 1, 0, self.t.size - 2)
        return i, Segment(self.h[i], self.v[i], self.v[i + 1], self.d[i], self.d[i + 1])

    def __call__(self, times: ArrayLike) -> Floats:
        """Return the curve at arbitrary times, its end values outside its knots."""
        x = np.asarray(times, dtype=float)
        i, seg = self._segment(x)
        s = (x - self.t[i]) / seg.h
        y = np.where(x < self.t[0], self.v[0], _hermite(s, seg))
        out: Floats = np.where(x > self.t[-1], self.v[-1], y)
        return out

    def integral(self, times: ArrayLike) -> Floats:
        """Return the integral from the first knot to each time."""
        x = np.asarray(times, dtype=float)
        i, (h, va, vb, da, db) = self._segment(x)
        s = (x - self.t[i]) / h
        y = self._before[i] + h * _partial(s, va, vb, h * da, h * db)
        y = np.where(x < self.t[0], self.v[0] * (x - self.t[0]), y)
        out: Floats = np.where(x > self.t[-1], self.total + self.v[-1] * (x - self.t[-1]), y)
        return out

    def derivative(self, times: ArrayLike) -> Floats:
        """Return the slope at arbitrary times; 0 outside the knots."""
        x = np.asarray(times, dtype=float)
        i, (h, va, vb, da, db) = self._segment(x)
        s = (x - self.t[i]) / h
        y = 6 * (vb - va) / h * s * (1 - s) + da * (1 - s) * (1 - 3 * s) - db * s * (2 - 3 * s)
        out: Floats = np.where((x < self.t[0]) | (x > self.t[-1]), 0.0, y)
        return out

    def second(self, times: ArrayLike) -> Floats:
        """Return the second derivative (one-sided from the right at knots; 0 outside)."""
        x = np.asarray(times, dtype=float)
        i = np.clip(np.searchsorted(self.t, x, side="right") - 1, 0, self.t.size - 2)
        s = (x - self.t[i]) / self.h[i]
        y = self._second_start[i] * (1 - s) + self._second_end[i] * s
        out: Floats = np.where((x < self.t[0]) | (x >= self.t[-1]), 0.0, y)
        return out

    # Evaluation on sorted times, one segment at a time.

    def on(self, times: ArrayLike) -> Floats:
        """Return the curve at ascending times: the same values as calling it, sliced by segment."""
        x = np.asarray(times, dtype=float)
        y = np.empty_like(x)
        cut = np.searchsorted(x, self.t, side="left")  # x[cut[i]:cut[i + 1]] lies in segment i
        y[: cut[0]] = self.v[0]
        for i in range(self.t.size - 1):
            a, b = cut[i], cut[i + 1]
            if a < b:
                s = (x[a:b] - self.t[i]) / self.h[i]
                y[a:b] = _hermite(s, Segment(self.h[i], self.v[i], self.v[i + 1], self.d[i], self.d[i + 1]))
        y[cut[-1] :] = self.v[-1]
        return y

    def integral_on(self, times: ArrayLike) -> Floats:
        """Return the integral from the first knot at ascending times, sliced by segment."""
        x = np.asarray(times, dtype=float)
        y = np.empty_like(x)
        cut = np.searchsorted(x, self.t, side="left")
        y[: cut[0]] = self.v[0] * (x[: cut[0]] - self.t[0])
        for i in range(self.t.size - 1):
            a, b = cut[i], cut[i + 1]
            if a < b:
                h = self.h[i]
                s = (x[a:b] - self.t[i]) / h
                y[a:b] = self._before[i] + h * _partial(s, self.v[i], self.v[i + 1], h * self.d[i], h * self.d[i + 1])
        y[cut[-1] :] = self.total + self.v[-1] * (x[cut[-1] :] - self.t[-1])
        return y

    # The knots' discontinuities.

    def jumps(self) -> tuple[Floats, Floats]:
        """Return the jumps in the second and third derivatives at each knot; the curve is constant outside.

        The curve is only C¹, so these jumps set how fast its spectrum falls.
        """
        return _jump(self._second_start, self._second_end), _jump(self._third, self._third)

    def sides(self, k: int) -> tuple[float, float, float, float]:
        """(x″ before, x″ after, x‴ before, x‴ after) at knot k; 0 outside the knots."""
        n = self.t.size
        before2 = float(self._second_end[k - 1]) if k > 0 else 0.0
        after2 = float(self._second_start[k]) if k < n - 1 else 0.0
        before3 = float(self._third[k - 1]) if k > 0 else 0.0
        after3 = float(self._third[k]) if k < n - 1 else 0.0
        return before2, after2, before3, after3

    def remaining(self, times: ArrayLike) -> Floats:
        """Return the integral from each time to the last knot."""
        x = np.asarray(times, dtype=float)
        i, (h, va, vb, da, db) = self._segment(x)
        s = (self.t[i + 1] - x) / h
        # The segment read backwards is the same Hermite form with the knots swapped.
        y = self._after[i + 1] + h * _partial(s, vb, va, -h * db, -h * da)
        y = np.where(x < self.t[0], self.total + self.v[0] * (self.t[0] - x), y)
        out: Floats = np.where(x > self.t[-1], self.v[-1] * (self.t[-1] - x), y)
        return out


def _jump(after: Floats, before: Floats) -> Floats:
    """Return what changes at each knot: the value just after it less the value just before."""
    return np.concatenate((after, [0.0])) - np.concatenate(([0.0], before))


def _hermite(s: Floats, seg: Segment) -> Floats:
    """Return the Hermite segment at normalized positions s, clipped to its knot values."""
    h, va, vb, da, db = seg
    p = s * s * (3 - 2 * s)
    y = va * (1 - p) + vb * p + h * s * (1 - s) * (da * (1 - s) - db * s)
    out: Floats = np.clip(y, np.minimum(va, vb), np.maximum(va, vb))
    return out


def _partial(s: Floats, va: Values, vb: Values, ma: Values, mb: Values) -> Floats:
    """Return the integral over [0, s] of the unit Hermite segment with scaled slopes ma, mb."""
    s2 = s * s
    p = s2 * s * (1 - s / 2)  # antiderivative of s²(3 - 2s)
    a = s2 * (6 - 8 * s + 3 * s2) / 12  # antiderivative of s(1 - s)²
    b = s2 * s * (4 - 3 * s) / 12  # antiderivative of s²(1 - s)
    out: Floats = va * (s - p) + vb * p + ma * a - mb * b
    return out


def amplitude_curve(t: ArrayLike, v: ArrayLike) -> Curve:
    """Make an amplitude shape: zero end slopes, so it joins silence smoothly."""
    return Curve.pchip(t, v, first=0.0, last=0.0)


def pitch_curve(t: ArrayLike, v: ArrayLike) -> Curve:
    """Make a pitch ratio: zero last slope, so it lands smoothly on the tuning."""
    return Curve.pchip(t, v, last=0.0)


def peak_index(shape: Curve) -> int:
    """Return the shape's first knot at gain 1, where its decay starts."""
    return int(np.flatnonzero(shape.v == 1.0)[0])


def compressed(shape: Curve, s: float) -> Curve:
    """Return the shape with its decay played s times faster from the peak on: e(t_pk + s(t − t_pk)) (docs/design.md).

    The result is again a Hermite curve. Knots after the peak move to
    t_pk + (t_k − t_pk)/s and their slopes scale by s, so each segment is the
    original segment read at the compressed time: its peak is still exactly 1,
    its segments stay monotone, and its jumps scale by s² and s³. At the peak
    the rise keeps its own second and third derivatives and the decay's are
    scaled, which `sides` reports. The shape's slope at its peak is zero, so
    the result is C¹. For s = 1 the shape itself is returned.
    """
    if s == 1:
        return shape
    k = peak_index(shape)
    if shape.d[k] != 0:
        msg = "decay compression needs a zero slope at the peak"
        raise ValueError(msg)
    t, d = shape.t.copy(), shape.d.copy()
    t[k + 1 :] = shape.t[k] + (shape.t[k + 1 :] - shape.t[k]) / s
    d[k + 1 :] *= s
    return Curve(t, shape.v, d)
