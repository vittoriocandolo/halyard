# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference method behind `halyard verify` and the grid (plan §8.2)."""

import math

import numpy as np
import pytest

from halyard_synth import reference, render
from halyard_synth.patch import loads
from halyard_synth.render import synthesize

from .conftest import EXAMPLE, make

LIMIT_DB = -100.0  # the rendering accuracy a pass needs
CONVERGED_DB = -120.0  # the reference agrees with its check 20 dB below it
LEAKAGE_DB = -110.0  # the decimator's own leakage, far below both
ALIGNED = 1000  # a shift of one sample makes the match this much worse, at least
RINGING = 1e-4  # the ideal band-limit rings before the trigger, below this of the peak within the guard
EDGES = 1e-10  # and below this at the very ends
USEFUL = 100  # the kink bound lies within this factor of the error it bounds
TINY = 1e-6  # an error that relative to a unit peak is −120 dB


def amp(*pts: tuple[float, float]) -> list[dict[str, float]]:
    """Write knots as TOML parses them."""
    return [{"ms": m, "gain": g} for m, g in pts]


def test_mask_is_flat_then_raised_cosine() -> None:
    """Find the mask 1 up to 0.4 F, 0 from 0.5 F, and falling between."""
    rate, m = 1536000, 4
    length = 1 << 16
    mask = reference.mask(length, m * rate, rate)
    f = np.arange(mask.size) * (m * rate / length)
    assert np.all(mask[f <= 0.4 * rate] == 1.0)
    assert np.all(mask[f >= 0.5 * rate] == 0.0)
    band = (f > 0.4 * rate) & (f < 0.5 * rate)
    assert np.all(np.diff(mask[band]) < 0)


def test_reference_rates_are_absolute() -> None:
    """Find the reference at 128 and 256 × base whatever K is, and refuse a multiple K does not divide."""
    v = reference.verify_take(loads(EXAMPLE))
    assert (v.reference_hz, v.check_hz) == (6144000, 12288000)
    assert "6.144 and 12.288 MHz" in str(v)
    with pytest.raises(ValueError, match="needs K dividing it"):
        reference.reference(loads(EXAMPLE), 100)  # 100 × base is not a multiple of K


def test_example_passes_with_a_converged_reference() -> None:
    """Verify the example: it passes against a converged reference, with negligible leakage, inside the domain."""
    v = reference.verify(loads(EXAMPLE))
    t = v.takes[0]
    assert v.passed
    assert t.converged
    assert t.error.rel_db <= LIMIT_DB
    assert t.error.max_db <= LIMIT_DB
    assert t.convergence.max_db <= CONVERGED_DB
    assert t.leakage_db <= LEAKAGE_DB
    assert t.domain.inside
    assert "PASS" in str(v)


def test_reference_is_aligned_with_the_shipping_output() -> None:
    """Find the reference as long as the shipping output and on the same time origin, its ringing inside the guard."""
    p = loads(EXAMPLE)
    ship = synthesize(p).out
    ref = reference.reference(p)
    assert ref.out.size == ship.size
    # Same time origin: no shift improves the match.
    errs = [np.max(np.abs(np.roll(ref.out, s) - ship)) for s in (-1, 0, 1)]
    assert errs[1] < min(errs[0], errs[2]) / ALIGNED
    # The band-limited signal rings a little before the trigger (it is ideal, not causal),
    # and the guard is wide enough for that ringing to die out.
    peak = np.max(np.abs(ref.internal))
    assert np.max(np.abs(ref.internal[: reference.GUARD])) <= RINGING * peak
    assert np.max(np.abs(ref.internal[:64])) <= EDGES * peak
    assert np.max(np.abs(ref.internal[-64:])) <= EDGES * peak


def test_aliasing_is_caught_and_predicted() -> None:
    """Find a loud driven layer failing against a converged reference, and the domain rule putting it outside."""
    # A loud noise layer at 24 dB of drive aliases audibly at 32×, and the domain rule knows it.
    v = reference.verify_take(make(**{"noise[2].level_db": 6.0, "drive.db": 24.0}))
    assert v.converged
    assert not v.passed
    assert v.error.max_db > LIMIT_DB
    assert not v.domain.inside


@pytest.mark.parametrize("rise_ms", [0.01, 0.02, 0.05])
def test_kink_bound_is_an_upper_bound(rise_ms: float) -> None:
    """Find the kink bound above the measured error of 10–50 µs rises, and within a factor of 100 of it."""
    p = make(
        **{
            "body.level_db": -math.inf,
            "noise[2].level_db": -math.inf,
            "noise[1].level_db": 0.0,
            "noise[1].band_hz": [1000.0, 20000.0],
            "noise[1].amp": amp((0, 0), (rise_ms, 1), (rise_ms + 0.1, 0.7), (2, 0.1), (6, 0)),
        },
    )
    v = reference.verify_take(p)
    bound = synthesize(p).kink
    assert v.error.max_abs <= bound
    assert v.error.max_abs >= bound / USEFUL  # and a useful one


@pytest.mark.parametrize(("rise_ms", "hz"), [(0.01, (6000.0, 11000.0, 19000.0)), (0.05, (1630.0, 4210.0))])
def test_kink_bound_holds_for_a_rim(rise_ms: float, hz: tuple[float, ...]) -> None:
    """Find the kink bound above the measured error of a rim struck in 10–50 µs, and within a factor of 100 of it."""
    p = make(
        **{
            "body.level_db": -math.inf,
            "noise[1].level_db": -math.inf,
            "noise[2].level_db": -math.inf,
            "rim": [
                {
                    "level_db": 0.0,
                    "damping": 1.0,
                    "modes": [{"hz": f, "level_db": -3.0 * i} for i, f in enumerate(hz)],
                    "amp": amp((0, 0), (rise_ms, 1), (rise_ms + 0.3, 0.4), (6, 0)),
                },
            ],
        },
    )
    v = reference.verify_take(p)
    bound = synthesize(p).kink
    assert v.passed
    assert v.error.max_abs <= bound
    assert v.error.max_abs >= bound / USEFUL


def test_verify_judges_every_take() -> None:
    """Verify two takes: seeds 1 and 2, each reported, and the count that pass."""
    v = reference.verify(make(**{"noise[2].level_db": 0.0, "drive.db": 12.0}), takes=2)
    assert [t.seed for t in v.takes] == [1, 2]
    assert "take 2, seed 2" in str(v)
    assert "takes pass" in str(v)


def test_compare() -> None:
    """Measure errors relative to the peak, and absolutely when the reference is near-silent."""
    ref = np.array([0.0, 1.0, -0.5])
    e = reference.compare(ref + np.array([0.0, TINY, 0.0]), ref)
    assert e.max_db == pytest.approx(CONVERGED_DB)
    assert e.passes()
    assert not reference.compare(ref + 1e-4, ref).passes()
    silent = reference.compare(np.zeros(3), np.zeros(3))
    assert silent.near_silent
    assert silent.passes()
    assert silent.max_abs == 0
    quiet = np.array([0.0, 1e-3, 0.0])
    assert reference.compare(quiet + 5e-8, quiet).passes()  # near-silent: the absolute bound applies
    assert not reference.compare(quiet + 5e-7, quiet).passes()


def test_verification_with_explicit_taps() -> None:
    """Verify with the shipped taps given explicitly, the reference using them too, as the grid's 16× runs do."""
    p = loads(EXAMPLE)
    taps = render.decimator(48000).taps
    v = reference.verify_take(p, taps=taps)
    assert v.passed
    assert np.array_equal(reference.reference(p, taps=taps).out, reference.reference(p).out)
    assert v.leakage_db <= LEAKAGE_DB


def test_a_near_silent_take_is_judged_absolutely() -> None:
    """Verify a take whose peak is under 0.01 against the absolute bound, and say so."""
    quiet = make(**{"body.level_db": -60.0, "noise[1].level_db": -70.0, "noise[2].level_db": -70.0})
    v = reference.verify_take(quiet)
    assert v.error.near_silent
    assert v.passed
    assert "near-silent: limit 1e-07" in str(v)
