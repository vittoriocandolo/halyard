# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import numpy as np
import pytest
from conftest import EXAMPLE, make

from halyard_synth import reference as REF
from halyard_synth.patch import loads
from halyard_synth.render import synthesize


def amp(*pts):
    return [{"ms": m, "gain": g} for m, g in pts]


def test_mask_is_flat_then_raised_cosine():
    rate, m = 1536000, 4
    length = 1 << 16
    mask = REF._mask(length, m * rate, rate)
    f = np.arange(mask.size) * (m * rate / length)
    assert np.all(mask[f <= 0.4 * rate] == 1.0)
    assert np.all(mask[f >= 0.5 * rate] == 0.0)
    band = (f > 0.4 * rate) & (f < 0.5 * rate)
    assert np.all(np.diff(mask[band]) < 0)


def test_reference_rates_are_absolute():
    v = REF.verify_take(loads(EXAMPLE))
    assert (v.reference_hz, v.check_hz) == (6144000, 12288000)
    assert "6.144 and 12.288 MHz" in str(v)
    with pytest.raises(ValueError):
        REF.reference(loads(EXAMPLE), 100)  # 100 × base is not a multiple of K


def test_example_passes_with_a_converged_reference():
    v = REF.verify(loads(EXAMPLE))
    t = v.takes[0]
    assert v.passed and t.converged
    assert t.error.rel_db <= -100 and t.error.max_db <= -100
    assert t.convergence.max_db <= -120
    assert t.leakage_db <= -110
    assert t.domain.inside
    assert "PASS" in str(v)


def test_reference_is_aligned_with_the_shipping_output():
    p = loads(EXAMPLE)
    ship = synthesize(p).out
    ref = REF.reference(p)
    assert ref.out.size == ship.size
    # Same time origin: no shift improves the match.
    errs = [np.max(np.abs(np.roll(ref.out, s) - ship)) for s in (-1, 0, 1)]
    assert errs[1] < min(errs[0], errs[2]) / 1000
    # The band-limited signal rings a little before the trigger (it is ideal, not causal),
    # and the guard is wide enough for that ringing to die out.
    peak = np.max(np.abs(ref.internal))
    assert np.max(np.abs(ref.internal[: REF.GUARD])) <= 1e-4 * peak
    assert np.max(np.abs(ref.internal[:64])) <= 1e-10 * peak
    assert np.max(np.abs(ref.internal[-64:])) <= 1e-10 * peak


def test_aliasing_is_caught_and_predicted():
    # A loud noise layer at 24 dB of drive aliases audibly at 32×, and the domain rule knows it.
    v = REF.verify_take(make(**{"noise[2].level_db": 6.0, "drive.db": 24.0}))
    assert v.converged and not v.passed and v.error.max_db > -100
    assert not v.domain.inside


@pytest.mark.parametrize("rise_ms", [0.01, 0.02, 0.05])
def test_kink_bound_is_an_upper_bound(rise_ms):
    p = make(
        **{
            "body.level_db": -math.inf,
            "noise[2].level_db": -math.inf,
            "noise[1].level_db": 0.0,
            "noise[1].band_hz": [1000.0, 20000.0],
            "noise[1].amp": amp((0, 0), (rise_ms, 1), (rise_ms + 0.1, 0.7), (2, 0.1), (6, 0)),
        }
    )
    v = REF.verify_take(p)
    bound = synthesize(p).kink
    assert v.error.max_abs <= bound
    assert v.error.max_abs >= bound / 100  # and a useful one


def test_verify_judges_every_take():
    v = REF.verify(make(**{"noise[2].level_db": 0.0, "drive.db": 12.0}), takes=2)
    assert [t.seed for t in v.takes] == [1, 2]
    assert "take 2, seed 2" in str(v) and "takes pass" in str(v)


def test_compare():
    ref = np.array([0.0, 1.0, -0.5])
    e = REF.compare(ref + np.array([0.0, 1e-6, 0.0]), ref)
    assert e.max_db == pytest.approx(-120.0) and e.passes()
    assert not REF.compare(ref + 1e-4, ref).passes()
    silent = REF.compare(np.zeros(3), np.zeros(3))
    assert silent.near_silent and silent.passes() and silent.max_abs == 0
    quiet = np.array([0.0, 1e-3, 0.0])
    assert REF.compare(quiet + 5e-8, quiet).passes()  # near-silent: the absolute bound applies
    assert not REF.compare(quiet + 5e-7, quiet).passes()
