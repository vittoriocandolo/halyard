# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The voice: the saturator, the model of body, layers and rattle, the knee slew and the kink bound (§2.3–§2.5, §5)."""

import dataclasses
import functools
import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from halyard_synth import noise, voice
from halyard_synth.curve import amplitude_curve, compressed, pitch_curve
from halyard_synth.patch import Patch, loads

from .conftest import EXAMPLE, make

type Floats = NDArray[np.float64]
type Knots = list[dict[str, float]]

RATE = 1536000  # 48 kHz at 32×
EXACT = 1e-12  # the sources are the model to this
CENTS = 0.001  # every mode sounds at its ratio of the tuning to this
DECAY = 1e-3  # a mode's decay rate is ratio^β times the fundamental's to this, relatively
DEPTH = 0.01  # the rattle's measured depth is its own to this, relatively
SIDE = 1e-9  # and nothing of it is out of phase with the fundamental
JUMP = 2e-3  # the jumps agree with one-sided derivatives of the model to this, relatively
DRIVEN_JUMP = 5e-3  # and through the saturator, where rounding meets the fit, to this
AT_KNOTS = 1e-5  # the kink terms hold x at their knots to this
BAND = 24000.0  # the decimator's band at 48 kHz, for the kink bound
ZETA3 = 1.2020569031595942
AUDIBLE = 0.1  # the noise is read where it is at least this, so the gate is measured well


def amp(*points: tuple[float, float]) -> Knots:
    """Write a shape's knots as TOML parses them."""
    return [{"ms": m, "gain": g} for m, g in points]


def pitch(*points: tuple[float, float]) -> Knots:
    """Write a pitch curve's knots as TOML parses them."""
    return [{"ms": m, "ratio": r} for m, r in points]


def modes(*pairs: tuple[float, float]) -> Knots:
    """Write modes as TOML parses them."""
    return [{"ratio": r, "level_db": v} for r, v in pairs]


def rim(
    *hz: tuple[float, float],
    level_db: float = -6.0,
    damping: float = 1.0,
    shape: Knots | None = None,
) -> list[dict[str, object]]:
    """Write one [[rim]] as TOML parses it: its modes in Hz with their levels, and its shape."""
    return [
        {
            "level_db": level_db,
            "damping": damping,
            "modes": [{"hz": f, "level_db": v} for f, v in hz],
            "amp": shape or amp((0, 0), (0.15, 1), (1.2, 0.3), (8, 0)),
        },
    ]


def silent_layers(**edits: object) -> dict[str, object]:
    """Return edits that silence both layers, and these."""
    return {"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf, **edits}


def x_exact(p: Patch, times: ArrayLike) -> Floats:
    """Evaluate the plan's model (§2.3, §2.4) directly, with the noise summed exactly."""
    t = np.asarray(times, dtype=float)
    b = p.body
    eb = amplitude_curve([m / 1000 for m, _ in b.amp], [g for _, g in b.amp])
    turns = pitch_curve([m / 1000 for m, _ in b.pitch], [r for _, r in b.pitch]).integral(t)
    x = np.zeros_like(t)
    if b.level_db > -math.inf:
        for ratio, level in b.modes:
            e = compressed(eb, ratio**b.damping)
            x += 10 ** (b.level_db / 20) * 10 ** (level / 20) * e(t) * np.sin(2 * np.pi * b.tune_hz * ratio * turns)
    for layer in p.noise:
        if layer.level_db > -math.inf:
            e = amplitude_curve([m / 1000 for m, _ in layer.amp], [g for _, g in layer.amp])
            q = 1 - layer.rattle * eb(t) * (1 - np.cos(2 * np.pi * b.tune_hz * turns)) / 2
            x += 10 ** (layer.level_db / 20) * e(t) * q * noise.at(p.seed, *layer.band_hz, t)
    for r in p.rim:
        if r.level_db > -math.inf:
            er = amplitude_curve([m / 1000 for m, _ in r.amp], [g for _, g in r.amp])
            for hz, level in r.modes:
                e = compressed(er, (hz / r.modes[0][0]) ** r.damping)
                x += 10 ** (r.level_db / 20) * 10 ** (level / 20) * e(t) * np.sin(2 * np.pi * hz * t)
    return x


def x_internal(p: Patch, rate: int, n0: int, n1: int) -> Floats:
    """Return the saturator's input as the renderer computes it, over [n0, n1)."""
    v = voice.voice(p)
    return voice.sources(v, rate, n0, n1, voice.tables(v, rate))


# The saturator.


def test_drive_identities() -> None:
    """Find 0 dB the identity, S(±1) = ±1 exactly, S odd and monotone, and the small-signal gain the drive."""
    assert voice.drive_alpha(0.0) == 0.0
    for db in (0.5, 6.0, 18.0, 48.0):
        a = voice.drive_alpha(db)
        assert a / math.tanh(a) == pytest.approx(10 ** (db / 20), rel=1e-14)
        x = np.linspace(-3, 3, 10001)
        y = voice.saturate(a, x)
        assert voice.saturate(a, np.array([1.0]))[0] == 1.0
        assert voice.saturate(a, np.array([-1.0]))[0] == -1.0
        assert np.array_equal(voice.saturate(a, -x), -y)
        # The small-signal gain is the drive.
        assert voice.saturate(a, np.array([1e-9]))[0] / 1e-9 == pytest.approx(10 ** (db / 20), rel=1e-6)
        assert np.all(np.diff(y) >= 0)
        assert np.all(np.abs(y) <= 1 / math.tanh(a) * (1 + 1e-15))
    x = np.linspace(-2, 2, 101)
    assert np.array_equal(voice.saturate(0.0, x), x)


def test_slope_and_curvature_are_the_derivatives() -> None:
    """Find S′ and S″ as the saturator's numerical derivatives, and 1 and 0 without drive."""
    x = np.linspace(-1.5, 1.5, 301)
    for db in (3.0, 24.0):
        a = voice.drive_alpha(db)
        h = 1e-6
        np.testing.assert_allclose(
            voice.slope(a, x),
            (voice.saturate(a, x + h) - voice.saturate(a, x - h)) / (2 * h),
            rtol=1e-7,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            voice.curvature(a, x),
            (voice.slope(a, x + h) - voice.slope(a, x - h)) / (2 * h),
            rtol=1e-6,
            atol=1e-5,
        )
    assert np.all(voice.slope(0.0, x) == 1.0)
    assert np.all(voice.curvature(0.0, x) == 0.0)


# The model.


@pytest.mark.parametrize(
    "edits",
    [
        {},
        {"body.damping": 2.5, "noise[2].rattle": 1.0, "drive.db": 0.0},
        {
            "body.modes": modes((1, -3), (1.593, 0), (2.136, -20), (5.1, -9)),
            "body.tune_hz": 700.0,
            "body.pitch": pitch((0, 1.5), (3, 0.9), (20, 1)),
        },
        {"body.level_db": -math.inf, "noise[2].rattle": 0.8},  # a silent body still gates the rattle
        {"noise[1].level_db": -math.inf, "seed": 2**63 - 1},
        {"rim": rim((1630, 0), (4210, -8), (9870, -20))},
        {"rim": [*rim((440, -3), (2500, 0), damping=2.5), *rim((700, 0), level_db=-math.inf)], "drive.db": 24.0},
    ],
)
def test_sources_are_the_model(edits: dict[str, object]) -> None:
    """Find the renderer's x at random samples equal to the plan's model, summed directly, for five patches."""
    p = make(**edits)
    rng = np.random.default_rng(1)
    n = np.unique(np.concatenate(([0, 1, 77], rng.integers(0, int(0.27 * RATE), 300))))
    v = voice.voice(p)
    tabs = voice.tables(v, RATE)
    got = np.concatenate([voice.sources(v, RATE, int(k), int(k) + 1, tabs) for k in n])
    np.testing.assert_allclose(got, x_exact(p, n / RATE), rtol=0, atol=EXACT)


def test_blocks_and_longer_renders_never_change_samples() -> None:
    """Find a sample the same whatever block it is computed in and however long the render."""
    p = make(**{"noise[2].rattle": 0.7})
    v = voice.voice(p)
    tabs = voice.tables(v, RATE)
    whole = voice.sources(v, RATE, 0, 200000, tabs)
    for a, b in ((0, 1), (65535, 65537), (123456, 200000), (199999, 200000)):
        assert np.array_equal(voice.sources(v, RATE, a, b, tabs), whole[a:b])
    y1, _ = voice.internal(v, RATE, 150000, tabs)
    y2, _ = voice.internal(v, RATE, 300000, tabs)
    assert np.array_equal(y1, y2[:150000])


def test_every_hit_starts_at_exactly_zero() -> None:
    """Find the first sample of every hit exactly 0."""
    for edits in ({}, {"drive.db": 30.0}, {"noise[1].amp": amp((0, 0), (0.02, 1), (5, 0))}):
        v = voice.voice(make(**edits))
        assert voice.sources(v, RATE, 0, 1, voice.tables(v, RATE))[0] == 0.0


def test_identical_layers_sum_coherently() -> None:
    """Find two identical layers twice one: the same noise, +6 dB."""
    # Two identical layers at equal level make +6 dB, not +3 dB: they are the same noise.
    p = make(**{"body.level_db": -math.inf})
    one = dataclasses.replace(p, noise=(p.noise[1],))
    two = dataclasses.replace(p, noise=(p.noise[1], p.noise[1]))
    x1, x2 = x_internal(one, RATE, 0, 300000), x_internal(two, RATE, 0, 300000)
    np.testing.assert_allclose(x2, 2 * x1, rtol=1e-15, atol=0)


def test_editing_the_body_keeps_the_noise_and_changes_the_rattle() -> None:
    """Retune the body: the layers' noise stays, and the rattle follows the new fundamental."""
    base = {"body.tune_hz": 185.0}
    moved = {"body.tune_hz": 230.0, "body.modes": modes((1, 0), (1.7, -3)), "body.damping": 2.0}
    silent_body = {"body.level_db": -math.inf}

    def layers(edits: dict[str, object]) -> Floats:
        return x_internal(make(**edits, **silent_body), RATE, 0, 300000)

    assert np.array_equal(layers({**base, "noise[2].rattle": 0.0}), layers({**moved, "noise[2].rattle": 0.0}))
    assert not np.array_equal(layers({**base, "noise[2].rattle": 0.5}), layers({**moved, "noise[2].rattle": 0.5}))


def test_a_silent_body_gates_the_rattle_like_a_sounding_one() -> None:
    """Find the rattle gated by a silent body as by a sounding one."""
    loud, quiet = (
        make(**{"noise[1].level_db": -math.inf}),
        make(**{"noise[1].level_db": -math.inf, "body.level_db": -math.inf}),
    )
    body_only = make(**{"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf})
    np.testing.assert_allclose(
        x_internal(loud, RATE, 0, 300000) - x_internal(body_only, RATE, 0, 300000),
        x_internal(quiet, RATE, 0, 300000),
        rtol=0,
        atol=1e-15,
    )


def test_source_length_and_sample_count() -> None:
    """Find the source length the last audible knot, and N every sample at or before it."""
    v = voice.voice(make())
    length_s, samples = 0.26, 399361  # the wires end at 260 ms
    assert v.t_src == length_s
    assert voice.sample_count(length_s, RATE) == samples
    # The last two: t·F rounds below a whole number it reaches, and above one it falls short of.
    for t, rate in (
        (0.26, 1411200),
        (1e-5, 1536000),
        (0.0, 1536000),
        (2.0, 3072000),
        (1.0629251700680271e-05, 1411200),
        (3.543083900226757e-06, 1411200),
    ):
        n = voice.sample_count(t, rate)
        assert (n - 1) / rate <= t < n / rate


# Tuning, modes and damping.


def test_a_rim_rings_at_its_own_frequencies_whatever_the_tuning_and_glide() -> None:
    """Find a rim's modes at their Hz to 0.001 cent, untouched by the body's tuning and its glide."""
    hz = (1630.0, 4210.0, 6950.0)
    held = amp((0, 0), (1, 1), (1990, 1), (2000, 0))
    rate = 48000
    n0, n1 = int(0.1 * rate), int(1.9 * rate)
    t = np.arange(n0, n1) / rate
    seen = []
    for tune, glide in ((185.0, pitch((0, 1.5), (15, 1))), (700.0, pitch((0, 1), (15, 1)))):
        p = make(
            **silent_layers(
                **{
                    "body.level_db": -math.inf,
                    "body.tune_hz": tune,
                    "body.pitch": glide,
                    "rim": rim(*[(f, -3.0 * i) for i, f in enumerate(hz)], damping=0.0, shape=held),
                },
            ),
        )
        x = x_internal(p, rate, n0, n1)
        f = fit_frequencies(t, x, [f * (1 + 1e-5) for f in hz])
        assert np.max(np.abs(1200 * np.log2(f / np.array(hz)))) <= CENTS
        seen.append(x)
    assert np.array_equal(seen[0], seen[1])


def fit_frequencies(t: Floats, x: Floats, guesses: list[float]) -> Floats:
    """Fit a sum of sines by least squares: the frequencies, with the amplitudes and phases solved linearly."""

    def residual(f: Floats) -> Floats:
        basis = np.concatenate([np.stack([np.sin(2 * np.pi * fi * t), np.cos(2 * np.pi * fi * t)]) for fi in f]).T
        coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
        out: Floats = basis @ coef - x
        return out

    fit: Floats = least_squares(residual, guesses, x_scale=1e-3, xtol=1e-15, ftol=1e-15, gtol=1e-15).x
    return fit


@pytest.mark.parametrize("tune", [40.0, 185.0, 1000.0, 4000.0])
def test_every_mode_sounds_at_its_ratio_of_the_tuning(tune: float) -> None:
    """Fit the settled body at 40 Hz to 4 kHz and find every mode within 0.001 cent of f₀·rᵢ."""
    ratios = (1.0, 1.593, 2.136, 2.295)
    p = make(
        **silent_layers(
            **{
                "body.tune_hz": tune,
                "body.damping": 0.0,
                "body.modes": modes(*[(r, -3.0 * i) for i, r in enumerate(ratios)]),
                "body.pitch": pitch((0, 1.08), (15, 1)),
                "body.amp": amp((0, 0), (1, 1), (1990, 1), (2000, 0)),
            },
        ),
    )
    rate = 48000
    n0, n1 = int(0.1 * rate), int(1.9 * rate)
    x = x_internal(p, rate, n0, n1)
    t = np.arange(n0, n1) / rate
    f = fit_frequencies(t, x, [tune * r * (1 + 1e-5) for r in ratios])
    cents = 1200 * np.log2(f / (tune * np.array(ratios)))
    assert np.max(np.abs(cents)) <= CENTS


def mode_amplitudes(t: Floats, x: Floats, freqs: tuple[float, ...], centres: Floats, half: float) -> Floats:
    """Return each mode's amplitude at each centre, by a windowed least-squares fit with known frequencies."""
    out = []
    for c in centres:
        w = np.abs(t - c) <= half
        u = (t[w] - c) / half
        cols = []
        for f in freqs:
            s, co = np.sin(2 * np.pi * f * t[w]), np.cos(2 * np.pi * f * t[w])
            cols += [s, u * s, u * u * s, co, u * co, u * u * co]
        coef, *_ = np.linalg.lstsq(np.stack(cols).T, x[w], rcond=None)
        out.append([math.hypot(coef[6 * i], coef[6 * i + 3]) for i in range(len(freqs))])
    return np.array(out)


@pytest.mark.parametrize(("ratio", "beta"), [(1.593, 0.5), (1.593, 1.0), (2.136, 2.0), (2.295, 3.0), (4.0, 1.0)])
def test_damping_scales_decay_rates_by_ratio_to_the_beta(ratio: float, beta: float) -> None:
    """Measure two modes' decay rates and find their ratio rᵢ^β."""
    # An exponential body shape, knots every 6 dB; flat pitch, so the phases are known exactly.
    s = ratio**beta
    spacing = 5.0 * s  # the faster mode still decays 6 dB per 5 ms
    knots = [(0.0, 0.0), *((1.0 + spacing * k, 10 ** (-6 * k / 20)) for k in range(16))]
    knots.append((knots[-1][0] + spacing, 0.0))
    p = make(
        **silent_layers(
            **{
                "body.tune_hz": 1000.0,
                "body.damping": beta,
                "body.modes": modes((1, 0), (ratio, 0)),
                "body.pitch": pitch((0, 1), (15, 1)),
                "body.amp": amp(*knots),
            },
        ),
    )
    rate = 48000
    x = x_internal(p, rate, 0, int(knots[-1][0] / 1000 * rate))
    t = np.arange(x.size) / rate
    rates = []
    for i, speed in enumerate((1.0, s)):
        # Fit over whole knot intervals of this mode, away from the peak and the end.
        a, b = 1e-3 + spacing / speed * 1e-3 * 2, 1e-3 + spacing / speed * 1e-3 * 12
        centres = np.linspace(a, b, 41)
        amps = mode_amplitudes(t, x, (1000.0, 1000.0 * ratio), centres, half=1e-3)[:, i]
        rates.append(-np.polyfit(centres, np.log(amps), 1)[0])
    assert rates[1] / rates[0] == pytest.approx(s, rel=DECAY)


def test_modes_share_the_peak_and_the_fundamental_lasts_longest() -> None:
    """Find every mode peaking with the body, the fundamental's envelope the body's shape itself."""
    v = voice.voice(make(**{"body.damping": 2.0}))
    tpk = v.body_amp.t[1]
    for m in v.modes:
        assert m.env.t[1] == tpk
        assert m.env.v[1] == 1.0
        assert m.env.t[-1] <= v.body_amp.t[-1]
    assert v.modes[0].env is v.body_amp


# Rattle.


def rattle_patch(depth: float, **edits: object) -> Patch:
    """Return a patch of the wires alone, rattling at this depth under a long, flat, silent body."""
    return make(
        **{
            "noise[1].level_db": -math.inf,
            "noise[2].rattle": depth,
            "noise[2].band_hz": [2500.0, 9000.0],
            "noise[2].amp": amp((0, 0), (1, 1), (590, 1), (600, 0)),
            "body.amp": amp((0, 0), (1, 1), (390, 1), (400, 0)),
            "body.level_db": -math.inf,
            "body.pitch": pitch((0, 1.08), (15, 1)),
            **edits,
        },
    )


@pytest.mark.parametrize("depth", [0.25, 0.5, 1.0])
def test_rattle_modulates_at_the_fundamental_with_its_depth(depth: float) -> None:
    """Find the rattle the gate 1 − d(1 − cos φ₁)/2 exactly, at the fundamental's phase with the glide, of depth d."""
    p = rattle_patch(depth)
    v = voice.voice(p)
    tabs = voice.tables(v, RATE)
    n0, n1 = int(0.002 * RATE), int(0.38 * RATE)
    x = voice.sources(v, RATE, n0, n1, tabs)
    wires = tabs[1]
    assert wires is not None
    n = wires[np.arange(n0, n1) % RATE]
    ok = np.abs(n) > AUDIBLE
    t = np.arange(n0, n1)[ok] / RATE
    q = x[ok] / (10 ** (p.noise[1].level_db / 20) * n[ok])
    phi = 2 * np.pi * p.body.tune_hz * pitch_curve([0.0, 0.015], [1.08, 1.0]).integral(t)
    np.testing.assert_allclose(q, 1 - depth * (1 - np.cos(phi)) / 2, rtol=0, atol=EXACT)
    # Demodulated: a component at the fundamental's phase, glide included, of depth d.
    coef, *_ = np.linalg.lstsq(np.stack([np.ones_like(t), np.cos(phi), np.sin(phi)]).T, q, rcond=None)
    assert 2 * coef[1] == pytest.approx(depth, rel=DEPTH)
    assert abs(coef[2]) <= SIDE


def test_a_rattle_outside_the_body_shape_is_not_gated() -> None:
    """Gate a layer only where the body's shape is: one that ends before it starts plays as it is."""
    p = rattle_patch(
        0.8,
        **{
            "noise[1].level_db": -6.0,
            "noise[1].rattle": 0.8,
            "noise[1].amp": amp((0, 0), (1, 1), (8, 0)),
            "body.amp": amp((10, 0), (20, 1), (390, 1), (400, 0)),
        },
    )
    v = voice.voice(p)
    tabs = voice.tables(v, RATE)
    n0, n1 = 0, int(0.03 * RATE)  # past the stick, into the body's shape
    stick = tabs[0]
    assert stick is not None
    plain = 10 ** (p.noise[0].level_db / 20) * v.layers[0].amp.on(np.arange(n0, n1) / RATE) * stick[n0:n1]
    gated = voice.sources(v, RATE, n0, n1, tabs) - voice.sources(
        dataclasses.replace(v, layers=(v.layers[1],)),
        RATE,
        n0,
        n1,
        tabs[1:],
    )
    np.testing.assert_allclose(gated, plain, rtol=0, atol=EXACT)


def test_no_rattle_without_depth_or_body_shape() -> None:
    """Find no rattle at depth 0, nor where the body's shape is 0."""
    for p in (rattle_patch(0.0), rattle_patch(0.8, **{"body.amp": amp((500, 0), (501, 1), (560, 0))})):
        v = voice.voice(p)
        tabs = voice.tables(v, RATE)
        n0, n1 = int(0.002 * RATE), int(0.45 * RATE)
        x = voice.sources(v, RATE, n0, n1, tabs)
        wires = tabs[1]
        assert wires is not None
        np.testing.assert_array_equal(
            x,
            10 ** (p.noise[1].level_db / 20)
            * v.layers[1].amp.on(np.arange(n0, n1) / RATE)
            * wires[np.arange(n0, n1) % RATE],
        )


# The knee slew at every drive.


def brute_knee(x: Floats, alpha: float) -> float:
    """Return the knee slew at α by brute force: the largest step of αx that crosses zero or touches |αx| ≤ 1."""
    u = np.concatenate(([0.0], x))
    a, b = u[:-1], u[1:]
    step = np.abs(b - a)
    counts = (np.signbit(a) != np.signbit(b)) | (np.minimum(np.abs(a), np.abs(b)) <= 1.0 / alpha)
    return alpha * float(np.max(step, where=counts, initial=0.0))


@pytest.mark.parametrize("edits", [{}, {"noise[1].level_db": 12.0, "drive.db": 20.0}, {"body.level_db": 10.0}])
def test_knee_record_is_exact_at_every_drive(edits: dict[str, object]) -> None:
    """Find the recorded knee slew the brute force's at every drive and drop, and the crest and steepest crossing."""
    v = voice.voice(make(**edits))
    count = voice.sample_count(v.t_src, RATE)
    tabs = voice.tables(v, RATE)
    _, slew = voice.internal(v, RATE, count, tabs)
    x = np.concatenate(
        [voice.sources(v, RATE, a, min(a + voice.BLOCK, count), tabs) for a in range(0, count, voice.BLOCK)],
    )
    alphas = np.concatenate(
        [
            [voice.drive_alpha(db) for db in np.arange(0.5, 48.01, 0.5)],
            slew.knees.drops(),
            np.nextafter(slew.knees.drops(), np.inf),
        ],
    )
    for a in alphas:
        assert slew.knee(a) == pytest.approx(brute_knee(x, a), rel=1e-15, abs=0)
    assert slew.crest == int(np.argmax(np.abs(x)))
    u = np.concatenate(([0.0], x))
    crossing = np.where(np.signbit(u[:-1]) != np.signbit(u[1:]), np.abs(np.diff(u)), -1.0)
    assert slew.steep == int(np.argmax(crossing))
    assert slew.peak == float(np.max(np.abs(np.diff(u))))


def test_knees_never_count_a_step_again_once_it_stops() -> None:
    """Find a step counted only while α·reach ≤ 1, and q linear in α otherwise."""
    k = voice.Knees(0.1, np.array([0.2, 0.5]), np.array([0.3, 0.4]))
    assert k.at(1.0) == pytest.approx(0.4)  # both steps reach below 1
    assert k.at(1 / 0.5) == pytest.approx(2 * 0.4)
    assert k.at(np.nextafter(2.0, 3.0)) == pytest.approx(2 * 0.3, rel=1e-12)  # the 0.4 step has left
    assert k.at(10.0) == pytest.approx(10 * 0.1)  # only the crossing step is left
    assert k.at(0.0) == 0.0
    np.testing.assert_allclose(k.drops(), [5.0, 2.0])


# The kink bound: every jump against one-sided derivatives of the model.


def one_sided(f: Callable[[Floats], Floats], tau: float, delta: float, side: int) -> tuple[float, float]:
    """Return (f″, f‴) at tau from one side, by a least-squares polynomial on [tau, tau ± delta]."""
    u = np.linspace(0, 1, 24)[1:] * side
    coef = np.polyfit(u, f(tau + u * delta), 8)
    p = np.poly1d(coef)
    return float(p.deriv(2)(0.0)) / delta**2, float(p.deriv(3)(0.0)) / delta**3


KINKY = {
    "body.damping": 1.5,
    "body.modes": modes((1, 0), (1.593, -4)),
    "body.pitch": pitch((0, 1.3), (2.5, 1.05), (12, 1)),
    "body.amp": amp((0, 0), (0.4, 1), (3, 0.5), (9, 0.2), (30, 0)),
    "noise[1].amp": amp((0, 0), (0.05, 1), (0.15, 0.7), (2, 0.1), (6, 0)),
    "noise[2].rattle": 0.6,
    "noise[2].amp": amp((0.7, 0), (1.9, 1), (11, 0.4), (40, 0)),
}


KINKY_RIM = {
    **KINKY,
    "rim": rim((1630, 0), (4210, -6), damping=1.5, shape=amp((0.2, 0), (0.35, 1), (1.1, 0.4), (4, 0.1), (9, 0))),
}


@pytest.mark.parametrize("kinky", [KINKY, KINKY_RIM], ids=["body and layers", "with a rim"])
def test_every_jump_matches_the_model(kinky: dict[str, object]) -> None:
    """Find each knot's jumps in x″ and x‴ as one-sided derivatives of the model measure them."""
    p = make(**kinky)
    v = voice.voice(p)
    k = voice.kink_terms(v, RATE)
    times = np.unique(k.t)
    for tau in times:
        rows = k.t == tau
        others = times[times != tau]
        delta = min(2e-6, 0.25 * float(np.min(np.abs(others - tau))))
        f = functools.partial(x_exact, p)
        b2, b3 = one_sided(f, tau, delta, -1) if tau > 0 else (0.0, 0.0)
        a2, a3 = one_sided(f, tau, delta, +1)
        want2, want3 = a2 - b2, a3 - b3
        got2, got3 = float(np.sum(k.d2[rows])), float(np.sum(k.d3[rows]))
        scale2 = max(abs(a2), abs(b2), 1.0)
        scale3 = max(abs(a3), abs(b3), 1e5)
        assert got2 == pytest.approx(want2, abs=JUMP * scale2), (tau, [k.paths[i] for i in np.flatnonzero(rows)])
        assert got3 == pytest.approx(want3, abs=JUMP * scale3), (tau, [k.paths[i] for i in np.flatnonzero(rows)])
        np.testing.assert_allclose(k.x[rows], x_exact(p, np.full(rows.sum(), tau)), rtol=0, atol=AT_KNOTS)


def test_jumps_through_the_saturator() -> None:
    """Find the jumps through the saturator as one-sided derivatives of S(x) measure them."""
    # Deep in saturation y hardly moves, so the numerical derivatives meet rounding: 10⁻¹⁶ amplified by
    # the polynomial fit, over δ² and δ³. The jumps that matter are far above that floor.
    p = make(**KINKY, **{"drive.db": 18.0})
    v = voice.voice(p)
    k = voice.kink_terms(v, RATE)
    a = v.alpha
    times = np.unique(k.t)
    for tau in times[1:]:
        rows = k.t == tau
        delta = min(1e-6, 0.25 * float(np.min(np.abs(times[times != tau] - tau))))

        def f(t: Floats) -> Floats:
            return voice.saturate(a, x_exact(p, t))

        b2, b3 = one_sided(f, tau, delta, -1)
        a2, a3 = one_sided(f, tau, delta, +1)
        g1, g2 = voice.slope(a, k.x[rows]), voice.curvature(a, k.x[rows])
        got2 = float(np.sum(g1 * k.d2[rows]))
        got3 = float(np.sum(g1 * k.d3[rows] + 3 * g2 * k.dx[rows] * k.d2[rows]))
        assert got2 == pytest.approx(a2 - b2, abs=DRIVEN_JUMP * max(abs(a2), abs(b2)) + 1e-10 / delta**2)
        assert got3 == pytest.approx(a3 - b3, abs=DRIVEN_JUMP * max(abs(a3), abs(b3)) + 1e-10 / delta**3)


def test_kink_bound_form() -> None:
    """Find the kink bound's form: its sum over knots, at many drives at once, its weights and its paths."""
    p = make(**KINKY_RIM, **{"drive.db": 12.0})
    v = voice.voice(p)
    k = voice.kink_terms(v, RATE)
    total = voice.kink_bound(k, v.alpha, RATE, BAND)
    assert total > 0
    assert sum(c for _, c in voice.kink_knots(k, v.alpha, RATE, BAND)) == pytest.approx(total, rel=1e-12)
    alphas = np.array([0.0, 0.5, v.alpha, 30.0])
    np.testing.assert_allclose(
        voice.kink_bounds(k, alphas, RATE, BAND),
        [voice.kink_bound(k, a, RATE, BAND) for a in alphas],
        rtol=1e-12,
    )
    # The weights: 4 ζ(n+1) · band / (2πF)^(n+1), with the 1% interpolation margin.
    w2, w3 = voice.weights(RATE, BAND)
    assert w2 == pytest.approx(1.01 * 4 * ZETA3 * BAND / (2 * np.pi * RATE) ** 3, rel=1e-15)
    assert w3 == pytest.approx(1.01 * 4 * np.pi**4 / 90 * BAND / (2 * np.pi * RATE) ** 4, rel=1e-15)
    # Paths name knots as the parser does, counting from 1.
    paths = {path for path, _ in voice.kink_knots(k, v.alpha, RATE, BAND)}
    assert {
        "body.amp[1]",
        "body.amp[5]",
        "body.pitch[2]",
        "noise[1].amp[5]",
        "noise[2].amp[4]",
        "rim[1].amp[2]",
    } <= paths


def test_silence_has_no_kinks() -> None:
    """Find no kink and a bound of 0 for silence."""
    p = make(**{"body.level_db": -math.inf, "noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf})
    k = voice.kink_terms(voice.voice(p), RATE)
    assert k.t.size == 0
    assert voice.kink_bound(k, 0.0, RATE, BAND) == 0.0


def test_example_patch_kinks_are_dominated_by_the_stick() -> None:
    """Find the example's kink bound led by the stick's knots."""
    v = voice.voice(loads(EXAMPLE))
    top = voice.kink_knots(voice.kink_terms(v, RATE), v.alpha, RATE, BAND)[0][0]
    assert top.startswith("noise[1].amp")
