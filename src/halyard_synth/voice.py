# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The modal body, the noise layers and the drive as pure functions of time (plan §2.3–§2.5, §3, §5).

    x(t) = b(t) + Σ_j w_j(t) + Σ_k c_k(t),    y(t) = S_α(x(t))

    b(t)   = Σ_i g_b a_i E_i(t) sin φ_i(t),    φ_i = r_i φ_1,    φ_1 = 2π f0 R(t),    R(t) = ∫_0^t r
    w_j(t) = g_j e_j(t) q_j(t) n_j(t),    q_j = 1 − d_j e_b(t) (1 − cos φ_1(t)) / 2
    c_k(t) = Σ_i g_k a_ki E_ki(t) sin 2π f_ki t
    v_j(t) = g_j e_j(t) q_j(t) Σ_k B(u_j(t) − k) n_jk(t),    u_j = 3 log2 s_j(t)

E_i is the body shape e_b with its decay compressed by s_i = r_i^β, itself a
Hermite curve (`curve.compressed`), and q_j is layer j's rattle, gated at the
body's fundamental. The rim k rings at its own frequencies f_ki, unglided, its
shape compressed by (f_ki / f_k1)^β_k in the same way. A brush v_j is a layer
whose band moves with its sweep s_j: n_jk is the same noise in the brush's band
moved by k third-octaves, and the cubic B-spline B blends them as u_j moves.
Being one noise, the blend is a filter, not a crossfade: all along the sweep
the band keeps the written band's shape within 0.5 dB, down to 30 dB below its
peak, and its level within 0.1 dB. Nothing here keeps state: sample n at rate F is computed
from t_n = n / F alone, so a block boundary, a longer render or a different
rate never changes a sample's value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.optimize import brentq

from . import noise
from .curve import Curve, amplitude_curve, compressed, pitch_curve
from .patch import played

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import ArrayLike, NDArray

    from .patch import Patch

    type Floats = NDArray[np.float64]
    # Per layer, its noise table or None for a silent layer; then each audible brush's tables, in order.
    type Tables = Sequence[Floats | None]

BLOCK = 2**16
TWO_PI = 2 * math.pi
STEPS = 3  # a brush's tables per octave of its sweep


def drive_alpha(db: float) -> float:
    """Return α with α / tanh α = 10^(db/20): the saturator's small-signal gain. 0 dB is bypass."""
    if db == 0.0:
        return 0.0
    g = 10.0 ** (db / 20.0)
    # x / tanh x lies between x and x + 1, so [g − 1, g] brackets the root.
    root: float = brentq(
        lambda a: a / math.tanh(a) - g,
        g - 1.0,
        g,
        xtol=1e-300,
        rtol=4 * np.finfo(float).eps,
        maxiter=500,
    )
    return root


@dataclass(frozen=True)
class Mode:
    """A body mode: its frequency ratio, its gain and its envelope."""

    ratio: float
    gain: float  # g_b · a_i; 0 when the body is silent
    env: Curve  # E_i: the body shape, its decay compressed by ratio^damping


@dataclass(frozen=True)
class Ring:
    """A rim mode: its rim, its frequency, its gain and its envelope."""

    rim: int  # the rim's index, from 0
    hz: float
    gain: float  # g_k · a_ki; 0 when the rim is silent
    env: Curve  # E_ki: the rim's shape, its decay compressed by (hz / hz_1)^damping


@dataclass(frozen=True)
class Layer:
    """A noise layer: its gain, its band, its rattle depth and its shape."""

    gain: float  # g_j; 0 when the layer is silent
    band: tuple[float, float]
    rattle: float  # d_j
    amp: Curve  # e_j


@dataclass(frozen=True)
class Brush:
    """A brush: its gain, its rattle depth and shape, its sweep, and the bands of its tables."""

    gain: float  # g_j; 0 when the brush is silent
    rattle: float
    amp: Curve
    sweep: Curve  # s_j, the ratio its band's edges are multiplied by
    first: int  # the first table's k
    bands: tuple[tuple[float, float], ...]  # table k − first: the band moved by k third-octaves; none when silent

    def u(self, t: Floats, *, on: bool = False) -> Floats:
        """Return u = 3 log2 s at times, ascending ones `on` the sweep's segments."""
        out: Floats = STEPS * np.log2(self.sweep.on(t) if on else self.sweep(t))
        return out


@dataclass(frozen=True)
class Voice:
    """The patch reduced to what synthesis needs; times in seconds."""

    seed: int
    f0: float
    pitch: Curve
    body_amp: Curve  # e_b, which also gates the rattle
    modes: tuple[Mode, ...]
    layers: tuple[Layer, ...]  # in patch order, silent ones included, so paths keep their numbers
    rings: tuple[Ring, ...]  # every rim's modes, in patch order
    brushes: tuple[Brush, ...]
    alpha: float
    t_src: float

    @property
    def body_sounds(self) -> bool:
        """Some mode is audible."""
        return any(m.gain for m in self.modes)

    @property
    def carriers(self) -> tuple[Layer | Brush, ...]:
        """The layers, then the brushes: every source of noise, numbered as `State.parts` numbers them."""
        return (*self.layers, *self.brushes)

    @property
    def rattles(self) -> bool:
        """Some audible layer or brush reads the body's shape and phase."""
        return any(c.gain and c.rattle for c in self.carriers)


def _seconds(knots: Iterable[tuple[float, float]]) -> list[float]:
    """Return the knots' times in seconds."""
    return [ms / 1000 for ms, _ in knots]


def voice(p: Patch) -> Voice:
    """Reduce a patch to what synthesis needs: the take its seed names, as played when its takes differ."""
    p = played(p)
    b = p.body
    body_amp = amplitude_curve(_seconds(b.amp), [g for _, g in b.amp])
    g_b = 10.0 ** (b.level_db / 20.0)
    modes = tuple(
        Mode(ratio, g_b * 10.0 ** (level / 20.0), compressed(body_amp, ratio**b.damping)) for ratio, level in b.modes
    )
    layers = tuple(
        Layer(10.0 ** (n.level_db / 20.0), n.band_hz, n.rattle, amplitude_curve(_seconds(n.amp), [g for _, g in n.amp]))
        for n in p.noise
    )
    rings = []
    for k, r in enumerate(p.rim):
        rim_amp = amplitude_curve(_seconds(r.amp), [g for _, g in r.amp])
        g_k, first = 10.0 ** (r.level_db / 20.0), r.modes[0][0]
        rings += [
            Ring(k, hz, g_k * 10.0 ** (level / 20.0), compressed(rim_amp, (hz / first) ** r.damping))
            for hz, level in r.modes
        ]
    brushes = []
    for pb in p.brush:
        ratios = [r for _, r in pb.sweep]
        # Each segment of the sweep is monotone, so u is extreme at its knots: B(u − k) is 0 for k outside.
        first, last = reach(min(ratios), max(ratios))
        lo, hi = pb.band_hz
        audible = pb.level_db > -math.inf
        bands = tuple((lo * 2.0 ** (k / STEPS), hi * 2.0 ** (k / STEPS)) for k in range(first, last + 1))
        brushes.append(
            Brush(
                10.0 ** (pb.level_db / 20.0),
                pb.rattle,
                amplitude_curve(_seconds(pb.amp), [g for _, g in pb.amp]),
                Curve.pchip(_seconds(pb.sweep), ratios, first=0.0, last=0.0),
                first,
                bands if audible else (),
            ),
        )
    return Voice(
        seed=p.seed,
        f0=b.tune_hz,
        pitch=pitch_curve(_seconds(b.pitch), [r for _, r in b.pitch]),
        body_amp=body_amp,
        modes=modes,
        layers=layers,
        rings=tuple(rings),
        brushes=tuple(brushes),
        alpha=drive_alpha(p.drive.db),
        t_src=p.length_ms / 1000,
    )


def sample_count(t_src: float, rate: int) -> int:
    """N = ⌊T_src·F⌋ + 1: every internal sample with n / F ≤ T_src, decided in float64."""
    n = math.floor(t_src * rate)
    while (n + 1) / rate <= t_src:
        n += 1
    while n / rate > t_src:
        n -= 1
    return n + 1


def tables(v: Voice, rate: int) -> list[Floats | None]:
    """Return the noise tables at the internal rate, as `Tables` orders them, computed together and cached."""
    bands = [layer.band for layer in v.layers if layer.gain] + [band for b in v.brushes for band in b.bands]
    found = iter(noise.tables(v.seed, bands, rate)) if bands else iter(())
    out = [next(found) if layer.gain else None for layer in v.layers]
    return out + list(found)


def brush_tables(v: Voice, tabs: Tables) -> list[list[Floats]]:
    """Return each brush's tables, none for a silent brush."""
    out, i = [], len(v.layers)
    for b in v.brushes:
        out.append([t for t in tabs[i : i + len(b.bands)] if t is not None])
        i += len(b.bands)
    return out


B_REACH = 2  # B is 0 from here out


def reach(low: float, high: float) -> tuple[int, int]:
    """Return the first and last k whose table a sweep between these ratios can weigh, and one more either way.

    u = 3 log2 s runs between the ratios' u, and B(u − k) is 0 unless |u − k| < 2;
    the extra k either way only ever weighs 0, and absorbs rounding.
    """
    return math.floor(STEPS * math.log2(low)) - 1, math.ceil(STEPS * math.log2(high)) + 1


def bspline_value(x: Floats) -> Floats:
    """Return B(x) alone, by the same operations as `bspline`, for synthesis."""
    a = np.abs(x)
    c = np.where((a >= 1) & (a < B_REACH), 2 - a, 0.0)
    out: Floats = np.where(a < 1, 2 / 3 - a * a + a**3 / 2, c**3 / 6)
    return out


def bspline(x: Floats) -> tuple[Floats, Floats, Floats]:
    """Return the cubic B-spline B(x), on [−2, 2], with its first and second derivatives: C², B‴ steps."""
    a = np.abs(x)
    inner, outer = a < 1, (a >= 1) & (a < B_REACH)
    c = np.where(outer, 2 - a, 0.0)
    value = np.where(inner, 2 / 3 - a * a + a**3 / 2, c**3 / 6)
    d1 = np.where(inner, -2 * x + 1.5 * x * a, -np.sign(x) * c * c / 2)
    d2 = np.where(inner, -2 + 3 * a, c)
    return value, d1, d2


B3_JUMPS = {-2: 1.0, -1: -4.0, 0: 6.0, 1: -4.0, 2: 1.0}  # B‴'s jumps at its knots


def _span(t: Floats, lo: float, hi: float) -> slice:
    """Return the samples of ascending t within [lo, hi]."""
    return slice(int(np.searchsorted(t, lo, "left")), int(np.searchsorted(t, hi, "right")))


def _support(t: Floats, curve: Curve) -> slice:
    """Return the samples of ascending t within the curve's knots."""
    return _span(t, curve.t[0], curve.t[-1])


def sources(v: Voice, rate: int, n0: int, n1: int, tabs: Tables) -> Floats:
    """Return x[n] = x(n / rate) for n0 ≤ n < n1; each source only over its support."""
    t = np.arange(n0, n1) / rate
    x = np.zeros(n1 - n0)
    rattle = _body(v, t, x)
    for layer, table in zip(v.layers, tabs, strict=False):  # the brushes' tables follow
        s = _support(t, layer.amp)
        if table is not None and s.start < s.stop:  # None: a silent layer
            x[s] += _envelope(layer, t, s, rattle) * _read(table, rate, n0, s)
    for brush, basis in zip(v.brushes, brush_tables(v, tabs), strict=True):
        s = _support(t, brush.amp)
        if basis and s.start < s.stop:  # none: a silent brush
            u = brush.u(t[s], on=True)
            blend = np.zeros(s.stop - s.start)
            near = range(math.floor(u.min()) - 1, math.ceil(u.max()) + 2)  # B(u − k) is 0 for every other k here
            for k, table in enumerate(basis, brush.first):
                if k in near:
                    blend += bspline_value(u - k) * _read(table, rate, n0, s)
            x[s] += _envelope(brush, t, s, rattle) * blend
    for ring in v.rings:
        if ring.gain:
            s = _support(t, ring.env)
            x[s] += ring.gain * ring.env.on(t[s]) * np.sin((TWO_PI * ring.hz) * t[s])
    return x


def _envelope(c: Layer | Brush, t: Floats, s: slice, rattle: tuple[slice, Floats | None]) -> Floats:
    """Return g e q over s: a layer's or a brush's gain and shape, and its rattle from the body's gate."""
    env = c.gain * c.amp.on(t[s])
    shared, gate = rattle
    if c.rattle and gate is not None:
        # The gate is defined on the shared span; outside it the body shape is 0 and so is the gate.
        a, b = max(s.start, shared.start), min(s.stop, shared.stop)
        if a < b:
            env[a - s.start : b - s.start] *= 1.0 - c.rattle * gate[a - shared.start : b - shared.start]
    return env


def _read(table: Floats, rate: int, n0: int, s: slice) -> Floats:
    """Return a noise table over s, the block starting at sample n0."""
    a, b = n0 + s.start, n0 + s.stop
    # Noise shapes end by 1 s, the table's period, so the index wraps only for a knot at exactly 1 s.
    out: Floats = table[a:b] if b <= rate else table[np.arange(a, b) % rate]
    return out


def _body(v: Voice, t: Floats, x: Floats) -> tuple[slice, Floats | None]:
    """Add the modes to x, and return the span the body shape covers and the rattle's gate on it, if any."""
    # The fundamental's phase and the body shape serve the modes and the rattle alike.
    lo, hi = math.inf, -math.inf
    if v.body_sounds:
        lo, hi = v.body_amp.t[0], v.body_amp.t[-1]
    for c in v.carriers:
        if c.gain and c.rattle:
            lo = min(lo, max(c.amp.t[0], v.body_amp.t[0]))
            hi = max(hi, min(c.amp.t[-1], v.body_amp.t[-1]))
    shared = _span(t, lo, hi) if lo <= hi else slice(0, 0)
    tb = t[shared]
    gate = None
    if tb.size:
        phi1 = (TWO_PI * v.f0) * v.pitch.integral_on(tb)
        xb = x[shared]
        for m in v.modes:
            if m.gain:
                s = _support(tb, m.env)
                xb[s] += m.gain * m.env.on(tb[s]) * np.sin(m.ratio * phi1[s])
        if v.rattles:
            gate = 0.5 * v.body_amp.on(tb) * (1.0 - np.cos(phi1))
    return shared, gate


def saturate(alpha: float, x: Floats) -> Floats:
    """Return S_α(x) = tanh(αx) / tanh α, odd by construction; the identity when α = 0."""
    if not alpha:
        return x
    norm = np.tanh(np.array([alpha]))[0]  # the same tanh as below, so S(1) = 1 exactly
    out: Floats = np.copysign(np.tanh(alpha * np.abs(x)), x) / norm
    return out


def slope(alpha: float, values: ArrayLike) -> Floats:
    """Return S′_α(x), the saturator's local gain."""
    x = np.asarray(values, dtype=float)
    if not alpha:
        return np.ones_like(x)
    out: Floats = alpha * (1.0 - np.tanh(alpha * x) ** 2) / np.tanh(alpha)
    return out


def curvature(alpha: float, values: ArrayLike) -> Floats:
    """Return S″_α(x) = −2α tanh(αx) S′_α(x)."""
    x = np.asarray(values, dtype=float)
    if not alpha:
        return np.zeros_like(x)
    out: Floats = -2.0 * alpha * np.tanh(alpha * x) * slope(alpha, x)
    return out


# The knee slew at every drive.


@dataclass(frozen=True)
class Knees:
    """The knee slew at every drive, recorded in one pass (plan §5).

    Drive leaves the saturator's input x unchanged, so for any α

        q(α) = α · M(1/α),   M(m) = the largest step of x that crosses zero or has an end within |x| ≤ m.

    M is kept as its upper frontier: `crossing`, the largest step that crosses
    zero, which qualifies at every α, and the steps that beat it, each with its
    `reach`, the smaller |x| at its two ends. As α grows a step stops qualifying
    once α·reach > 1 and never qualifies again, so q grows at most in
    proportion to α and drops where α passes 1/reach.
    """

    crossing: float
    reach: Floats  # ascending
    step: Floats  # ascending: the largest step with at most that reach

    def largest(self, m: ArrayLike) -> Floats:
        """Return M(m) for m ≥ 0."""
        i = np.searchsorted(self.reach, np.asarray(m, dtype=float), side="right")
        best = self.step[np.maximum(i - 1, 0)] if self.step.size else np.zeros(np.shape(i))
        out: Floats = np.maximum(self.crossing, np.where(i > 0, best, 0.0))
        return out

    def at(self, alpha: ArrayLike) -> Floats:
        """Return q(α); 0 without drive."""
        a = np.asarray(alpha, dtype=float)
        driven = a > 0
        out: Floats = np.where(driven, a * self.largest(np.where(driven, 1.0 / np.where(driven, a, 1.0), 0.0)), 0.0)
        return out

    def drops(self) -> Floats:
        """Return the α at which q drops: the last α at which each frontier step still qualifies."""
        out: Floats = 1.0 / self.reach[self.reach > 0]
        return out


def _frontier(reach: Floats, step: Floats) -> tuple[Floats, Floats]:
    """Return the points that no other beats with a smaller or equal reach and a larger or equal step."""
    if not reach.size:
        return reach, step
    order = np.lexsort((-step, reach))
    r, s = reach[order], step[order]
    keep = np.empty(s.size, dtype=bool)
    keep[0], keep[1:] = True, s[1:] > np.maximum.accumulate(s)[:-1]
    return r[keep], s[keep]


@dataclass(frozen=True)
class Slew:
    """Steps of the saturator's input x between consecutive internal samples."""

    peak: float  # the largest step anywhere
    knees: Knees = field(compare=False, repr=False)
    crest: int = field(default=0, compare=False)  # the internal sample with the largest |x|
    steep: int = field(default=0, compare=False)  # the end of the largest step that crosses zero

    def knee(self, alpha: float) -> float:
        """q_knee at α: the largest step of αx that crosses zero or touches |αx| ≤ 1."""
        return float(self.knees.at(alpha))


def internal(v: Voice, rate: int, count: int, tabs: Tables) -> tuple[Floats, Slew]:
    """Return y[n] = y(n / rate) for 0 ≤ n < count, in blocks aligned to n = 0, and the slew of x."""
    y = np.empty(count)
    prev, peak = 0.0, 0.0  # silence before the trigger
    crossing, crest, crest_x, steep = 0.0, 0, 0.0, 0
    reach: list[Floats] = []
    steps: list[Floats] = []
    for n0 in range(0, count, BLOCK):
        n1 = min(n0 + BLOCK, count)
        x = sources(v, rate, n0, n1, tabs)
        y[n0:n1] = saturate(v.alpha, x)
        u = np.concatenate(([prev], x))
        step = np.abs(np.diff(u))
        a, b = u[:-1], u[1:]
        peak = max(peak, float(step.max()))
        crosses = np.signbit(a) != np.signbit(b)
        c = float(np.max(step, where=crosses, initial=0.0))
        if c > crossing:
            crossing, steep = c, n0 + int(np.argmax(np.where(crosses, step, -1.0)))
        beats = ~crosses & (step > crossing)
        if beats.any():
            r, s = _frontier(np.minimum(np.abs(a), np.abs(b))[beats], step[beats])
            reach.append(r)
            steps.append(s)
        i = int(np.argmax(np.abs(x)))
        if abs(x[i]) > crest_x:
            crest, crest_x = n0 + i, abs(float(x[i]))
        prev = x[-1]
    r = np.concatenate(reach) if reach else np.zeros(0)
    s = np.concatenate(steps) if steps else np.zeros(0)
    keep = s > crossing
    r, s = _frontier(r[keep], s[keep])
    return y, Slew(peak, Knees(crossing, r, s), crest, steep)


# The kink bound.

ZETA3, ZETA4 = 1.2020569031595942, math.pi**4 / 90
INTERPOLATION_MARGIN = 1.01  # covers the carriers' interpolation at knots (plan §5)


class Part(NamedTuple):
    """An audible layer at some times, and the derivative of each factor."""

    c: Floats  # the carrier g_j e_j n_j, before the rattle
    dc: Floats
    q: Floats  # the rattle
    dq: Floats
    n: Floats  # the noise
    dn: Floats


@dataclass(frozen=True)
class State:
    """Everything x and x′ are made of, at some times."""

    phi1: Floats
    dphi1: Floats
    gate: Floats  # G = (1 − cos φ_1) / 2
    dgate: Floats
    eb: Floats
    deb: Floats
    x: Floats
    dx: Floats
    parts: dict[int, Part]  # by carrier index, from 0: the audible layers and brushes, n the blend of a brush's


def state(v: Voice, times: ArrayLike, rate: int, tabs: Tables) -> State:
    """Return x, x′ and their parts at arbitrary times, with the noise from the tables by cubic interpolation."""
    t = np.asarray(times, dtype=float)
    phi1 = (TWO_PI * v.f0) * v.pitch.integral(t)
    dphi1 = (TWO_PI * v.f0) * v.pitch(t)
    eb, deb = v.body_amp(t), v.body_amp.derivative(t)
    gate, dgate = 0.5 * (1.0 - np.cos(phi1)), 0.5 * np.sin(phi1) * dphi1
    x, dx = np.zeros_like(t), np.zeros_like(t)
    for m in v.modes:
        if m.gain:
            e, de = m.env(t), m.env.derivative(t)
            sn, cs = np.sin(m.ratio * phi1), np.cos(m.ratio * phi1)
            x += m.gain * e * sn
            dx += m.gain * (de * sn + e * cs * m.ratio * dphi1)
    noises: dict[int, tuple[Floats, Floats]] = {}
    for j, table in enumerate(tabs[: len(v.layers)]):
        if table is not None:  # None: a silent layer
            noises[j] = noise.interpolate(table, rate, t)
    for i, (brush, basis) in enumerate(zip(v.brushes, brush_tables(v, tabs), strict=True)):
        if basis:  # none: a silent brush
            noises[len(v.layers) + i] = _blend(brush, basis, rate, t)
    parts: dict[int, Part] = {}
    for j, (n, dn) in noises.items():
        c = v.carriers[j]
        e, de = c.amp(t), c.amp.derivative(t)
        q = 1.0 - c.rattle * eb * gate
        dq = -c.rattle * (deb * gate + eb * dgate)
        cj, dcj = c.gain * e * n, c.gain * (de * n + e * dn)
        x += q * cj
        dx += dq * cj + q * dcj
        parts[j] = Part(cj, dcj, q, dq, n, dn)
    for ring in v.rings:
        if ring.gain:
            e, de = ring.env(t), ring.env.derivative(t)
            w = TWO_PI * ring.hz
            sn, cs = np.sin(w * t), np.cos(w * t)
            x += ring.gain * e * sn
            dx += ring.gain * (de * sn + e * cs * w)
    return State(phi1, dphi1, gate, dgate, eb, deb, x, dx, parts)


def _du(brush: Brush, t: Floats) -> Floats:
    """Return u′ = 3 s′ / (s ln 2)."""
    out: Floats = STEPS * brush.sweep.derivative(t) / (brush.sweep(t) * math.log(2))
    return out


def _blend(brush: Brush, basis: Sequence[Floats], rate: int, t: Floats) -> tuple[Floats, Floats]:
    """Return a brush's blended noise Σ B(u − k) n_k and its derivative at arbitrary times."""
    u, du = brush.u(t), _du(brush, t)
    n, dn = np.zeros_like(t), np.zeros_like(t)
    for k, table in enumerate(basis, brush.first):
        w, dw, _ = bspline(u - k)
        nk, dnk = noise.interpolate(table, rate, t)
        n += w * nk
        dn += dw * du * nk + w * dnk
    return n, dn


@dataclass(frozen=True)
class Kinks:
    """The jumps of x″ and x‴ at every knot, with x and x′ there: all the kink bound needs but α.

    Each row is one source's jump at one knot, named by the knot's path. Jumps
    at the same instant are bounded separately, which the triangle inequality
    allows. Drive acts only through the saturator, so these are computed once
    per render and the bound follows at any drive.
    """

    paths: tuple[str, ...]  # the knot's path, such as "noise[2].amp[3]", per row
    t: Floats
    x: Floats
    dx: Floats
    d2: Floats  # Δx″
    d3: Floats  # Δx‴


class Row(NamedTuple):
    """One source's jumps at a curve's knots, and the state there."""

    paths: list[str]
    t: Floats
    s: State
    d2: Floats  # Δx″
    d3: Floats  # Δx‴


def kink_terms(v: Voice, rate: int, tabs: Tables | None = None) -> Kinks:
    """Return every source's jumps at every knot (plan §5): modes, the peak, pitch knots, rattle, layers and rims."""
    if tabs is None:
        tabs = tables(v, rate)
    rows = [*_mode_rows(v, rate, tabs), *_rattle_rows(v, rate, tabs), *_pitch_rows(v, rate, tabs)]
    rows += [*_layer_rows(v, rate, tabs), *_ring_rows(v, rate, tabs), *_sweep_rows(v, rate, tabs)]
    if not rows:
        empty = np.zeros(0)
        return Kinks((), empty, empty, empty, empty, empty)
    return Kinks(
        tuple(p for row in rows for p in row.paths),
        np.concatenate([row.t for row in rows]),
        np.concatenate([row.s.x for row in rows]),
        np.concatenate([row.s.dx for row in rows]),
        np.concatenate([np.broadcast_to(row.d2, row.t.shape) for row in rows]),
        np.concatenate([np.broadcast_to(row.d3, row.t.shape) for row in rows]),
    )


def _body_paths(v: Voice) -> list[str]:
    """Return the body shape's knots' paths."""
    return [f"body.amp[{k}]" for k in range(1, v.body_amp.t.size + 1)]


def _mode_rows(v: Voice, rate: int, tabs: Tables) -> list[Row]:
    """Return the mode envelopes' jumps: each mode's compressed shape, already scaled by s² and s³."""
    rows = []
    body = _body_paths(v)
    for m in v.modes:
        if not m.gain:
            continue
        s = state(v, m.env.t, rate, tabs)
        j2, j3 = m.env.jumps()
        sn, cs = np.sin(m.ratio * s.phi1), np.cos(m.ratio * s.phi1)
        rows.append(Row(body, m.env.t, s, m.gain * j2 * sn, m.gain * (j3 * sn + 3 * j2 * cs * m.ratio * s.dphi1)))
    return rows


def _rattle_rows(v: Voice, rate: int, tabs: Tables) -> list[Row]:
    """Return the rattle's jumps: it reads the body shape, so q_j jumps where e_b does."""
    if not v.rattles:
        return []
    rows = []
    body = _body_paths(v)
    s = state(v, v.body_amp.t, rate, tabs)
    j2, j3 = v.body_amp.jumps()
    for j, part in s.parts.items():
        rattle = v.carriers[j].rattle
        if rattle:
            q2 = -rattle * j2 * s.gate
            q3 = -rattle * (j3 * s.gate + 3 * j2 * s.dgate)
            rows.append(Row(body, v.body_amp.t, s, q2 * part.c, q3 * part.c + 3 * q2 * part.dc))
    return rows


def _pitch_rows(v: Voice, rate: int, tabs: Tables) -> list[Row]:
    """Return the pitch knots' jumps: φ_i‴ = 2π f0 r_i r″ jumps where r″ does, for every mode and for the rattle."""
    if not (v.body_sounds or v.rattles):
        return []
    s = state(v, v.pitch.t, rate, tabs)
    j2, _ = v.pitch.jumps()
    dphi3 = TWO_PI * v.f0 * j2
    d3 = np.zeros_like(dphi3)
    for m in v.modes:
        if m.gain:
            d3 = d3 + m.gain * m.env(v.pitch.t) * np.cos(m.ratio * s.phi1) * m.ratio * dphi3
    for j, part in s.parts.items():
        rattle = v.carriers[j].rattle
        if rattle:
            # (1 − cos φ_1)/2 has a third derivative that jumps by sin φ_1 · Δφ_1‴ / 2.
            d3 = d3 - rattle * s.eb * 0.5 * np.sin(s.phi1) * dphi3 * part.c
    paths = [f"body.pitch[{k}]" for k in range(1, v.pitch.t.size + 1)]
    return [Row(paths, v.pitch.t, s, np.zeros_like(d3), d3)]


def _carrier_path(v: Voice, j: int) -> str:
    """Return a carrier's path: noise[j], or brush[i] after the layers."""
    return f"noise[{j + 1}]" if j < len(v.layers) else f"brush[{j - len(v.layers) + 1}]"


def _layer_rows(v: Voice, rate: int, tabs: Tables) -> list[Row]:
    """Return the jumps of the layers' and the brushes' shapes."""
    rows = []
    for j, c in enumerate(v.carriers):
        if not c.gain:  # a silent layer or brush
            continue
        s = state(v, c.amp.t, rate, tabs)
        j2, j3 = c.amp.jumps()
        q, dq, n, dn = s.parts[j].q, s.parts[j].dq, s.parts[j].n, s.parts[j].dn
        rows.append(
            Row(
                [f"{_carrier_path(v, j)}.amp[{k}]" for k in range(1, c.amp.t.size + 1)],
                c.amp.t,
                s,
                c.gain * j2 * q * n,
                c.gain * (j3 * q * n + 3 * j2 * (dq * n + q * dn)),
            ),
        )
    return rows


def crossings(brush: Brush) -> tuple[Floats, Floats, NDArray[np.intp]]:
    """Return where u crosses a whole number: the times, the numbers and each one's sweep segment.

    There B‴ jumps, and with it the blend's third derivative. Each segment of
    the sweep is monotone, so it crosses each number between its ends once.
    """
    times: list[float] = []
    levels: list[int] = []
    segments: list[int] = []
    t, r = brush.sweep.t, brush.sweep.v
    for i in range(t.size - 1):
        ra, rb = float(r[i]), float(r[i + 1])
        if ra == rb:
            continue
        # Candidates from u at the ends, one wider either way for rounding; the levels decide, as ratios.
        lowest, highest = reach(min(ra, rb), max(ra, rb))
        ms = range(lowest, highest + 1) if rb > ra else range(highest, lowest - 1, -1)  # in the order u meets them
        for m in ms:
            level = 2.0 ** (m / STEPS)
            if not min(ra, rb) <= level <= max(ra, rb):
                continue
            if level == ra:
                tau = float(t[i])
            elif level == rb:
                tau = float(t[i + 1])
            else:
                tau = brentq(lambda x, lv=level: float(brush.sweep(x)) - lv, t[i], t[i + 1], xtol=1e-15)
            if not times or (tau, m) != (times[-1], levels[-1]):  # a crossing at a knot, once
                times.append(tau)
                levels.append(m)
                segments.append(i)
    return np.array(times, dtype=float), np.array(levels, dtype=float), np.array(segments, dtype=np.intp)


def _sweep_rows(v: Voice, rate: int, tabs: Tables) -> list[Row]:
    """Return the jumps a brush's sweep makes: at its knots, through u″ and u‴, and where u crosses a whole number."""
    rows = []
    for i, (brush, basis) in enumerate(zip(v.brushes, brush_tables(v, tabs), strict=True)):
        if not basis:  # a silent brush
            continue
        j, path = len(v.layers) + i, f"brush[{i + 1}].sweep"
        sweep = brush.sweep
        # At the knots: u″ and u‴ jump with s″ and s‴, while s and s′ are continuous.
        t = sweep.t
        s = state(v, t, rate, tabs)
        r, dr = sweep(t), sweep.derivative(t)
        j2, j3 = sweep.jumps()
        c0 = STEPS / math.log(2)
        du, du2, du3 = c0 * dr / r, c0 * j2 / r, c0 * (j3 / r - 3 * dr * j2 / r**2)
        carrier, dcarrier = _brush_carrier(brush, s.parts[j], t)
        u = brush.u(t)
        d2, d3 = np.zeros_like(t), np.zeros_like(t)
        for k, table in enumerate(basis, brush.first):
            _, b1, b2 = bspline(u - k)
            nk, dnk = noise.interpolate(table, rate, t)
            w2, w3 = b1 * du2, b1 * du3 + 3 * b2 * du * du2
            d2 += carrier * w2 * nk
            d3 += w3 * carrier * nk + 3 * w2 * (dcarrier * nk + carrier * dnk)
        rows.append(Row([f"{path}[{k}]" for k in range(1, t.size + 1)], t, s, d2, d3))
        # Where u crosses a whole number m: B‴(u − k) jumps for k = m − 2 … m + 2, by u′³ each.
        tau, levels, segments = crossings(brush)
        if tau.size:
            s = state(v, tau, rate, tabs)
            carrier, _ = _brush_carrier(brush, s.parts[j], tau)
            cube = _du(brush, tau) ** 3
            d3 = np.zeros_like(tau)
            for k, table in enumerate(basis, brush.first):
                jump = np.array([B3_JUMPS.get(int(m) - k, 0.0) for m in levels])
                d3 += carrier * jump * cube * noise.interpolate(table, rate, tau)[0]
            # Named by the segment they lie in, between two knots: a crossing is not a knot.
            rows.append(Row([f"{path}[{g + 1}–{g + 2}]" for g in segments], tau, s, np.zeros_like(tau), d3))
    return rows


def _brush_carrier(brush: Brush, part: Part, t: Floats) -> tuple[Floats, Floats]:
    """Return g e q and its derivative: all of a brush but its blend."""
    e, de = brush.amp(t), brush.amp.derivative(t)
    return brush.gain * e * part.q, brush.gain * (de * part.q + e * part.dq)


def _ring_rows(v: Voice, rate: int, tabs: Tables) -> list[Row]:
    """Return the rim modes' jumps: each ring's compressed shape, like the body modes', at a constant frequency."""
    rows = []
    for ring in v.rings:
        if not ring.gain:
            continue
        s = state(v, ring.env.t, rate, tabs)
        j2, j3 = ring.env.jumps()
        w = TWO_PI * ring.hz
        sn, cs = np.sin(w * ring.env.t), np.cos(w * ring.env.t)
        paths = [f"rim[{ring.rim + 1}].amp[{k}]" for k in range(1, ring.env.t.size + 1)]
        rows.append(Row(paths, ring.env.t, s, ring.gain * j2 * sn, ring.gain * (j3 * sn + 3 * j2 * cs * w)))
    return rows


def weights(rate: int, band: float) -> tuple[float, float]:
    """Return the weights of the jumps in x″ and x‴ in the kink bound."""
    return (
        INTERPOLATION_MARGIN * 4 * ZETA3 * band / (TWO_PI * rate) ** 3,
        INTERPOLATION_MARGIN * 4 * ZETA4 * band / (TWO_PI * rate) ** 4,
    )


def kink_contributions(k: Kinks, alpha: float, rate: int, band: float) -> Floats:
    """Return each row's share of the kink bound at one drive.

    A jump J in the n-th derivative of y gives a spectrum falling as
    J / (2πf)^(n+1); sampling at F folds its parts near every multiple of F into
    the band the decimator passes, below `band`, so the error is at most
    4 ζ(n+1) · band · |J| / (2πF)^(n+1). Through the saturator y″ jumps by
    S′·Δx″ and y‴ by S′·Δx‴ + 3·S″·x′·Δx″.
    """
    w2, w3 = weights(rate, band)
    g1, g2 = slope(alpha, k.x), curvature(alpha, k.x)
    out: Floats = w2 * np.abs(g1 * k.d2) + w3 * np.abs(g1 * k.d3 + 3 * g2 * k.dx * k.d2)
    return out


def kink_bound(k: Kinks, alpha: float, rate: int, band: float) -> float:
    """Return the upper bound on the output error from the curves' kinks, at unity gain."""
    return float(np.sum(kink_contributions(k, alpha, rate, band)))


def kink_bounds(k: Kinks, alphas: ArrayLike, rate: int, band: float) -> Floats:
    """Return the kink bound at many drives at once."""
    w2, w3 = weights(rate, band)
    a = np.asarray(alphas, dtype=float)[:, None]
    driven = a > 0
    th = np.tanh(a * k.x)
    g1 = np.where(driven, a * (1.0 - th**2) / np.tanh(np.where(driven, a, 1.0)), 1.0)
    g2 = np.where(driven, -2.0 * a * th * g1, 0.0)
    out: Floats = np.sum(w2 * np.abs(g1 * k.d2) + w3 * np.abs(g1 * k.d3 + 3 * g2 * k.dx * k.d2), axis=1)
    return out


def kink_knots(k: Kinks, alpha: float, rate: int, band: float) -> list[tuple[str, float]]:
    """Return each knot's share of the kink bound, summed over its sources: (path, contribution), largest first."""
    shares: dict[str, float] = {}
    for path, c in zip(k.paths, kink_contributions(k, alpha, rate, band), strict=True):
        shares[path] = shares.get(path, 0.0) + float(c)
    return sorted(shares.items(), key=lambda row: -row[1])
