# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math

import numpy as np
import pytest

from halyard_synth import cli
from halyard_synth import render as R
from halyard_synth import voice as V
from halyard_synth.loudness import momentary_max
from halyard_synth.patch import loads

from .conftest import EXAMPLE, make


def amp(*points):
    return [{"ms": m, "gain": g} for m, g in points]


SILENT = {"body.level_db": -math.inf, "noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}
FAST = {
    "noise[1].level_db": 12.0,
    "noise[1].band_hz": [1000.0, 20000.0],
    "noise[1].amp": amp((0, 0), (0.01, 1), (0.11, 0.7), (2, 0.1), (6, 0)),
}
HOT = {"noise[2].level_db": 6.0, "drive.db": 24.0}


def test_output_length_follows_the_timing_contract():
    for rate in (44100, 48000, 88200, 96000):
        p = make(**{"output.sample_rate": rate})
        s = R.synthesize(p)
        F, factor = R.rates(rate)
        N = V.sample_count(0.26, F)
        L = R.decimator(R.FAMILY[rate]).taps.size
        assert s.source_samples == N
        assert s.out.size == (N + L - 2) // factor + 1
        audio, report = R.render(p)
        assert audio.size == s.out.size == report.samples
        assert audio.dtype == np.float32
        # Every shape starts at 0, so every file starts at exactly zero.
        assert audio[0] == 0.0


def test_silent_sources_never_set_the_length():
    loud = R.render(make())[1]
    no_wires = R.render(make(**{"noise[2].level_db": -math.inf}))[1]
    assert loud.source_ms == 260.0 and no_wires.source_ms == 180.0
    assert no_wires.samples < loud.samples
    # A silent body still gates the rattle, inside the layer it gates, but never lengthens the file.
    long_body = make(**{"body.level_db": -math.inf, "body.amp": amp((0, 0), (1, 1), (900, 0))})
    assert R.render(long_body)[1].source_ms == 260.0


def test_repeating_a_render_changes_nothing():
    p = make(**{"drive.db": 18.0, "noise[2].rattle": 0.8})
    assert np.array_equal(R.render(p)[0], R.render(p)[0])


def test_output_gain_and_polarity_never_touch_synthesis():
    base = R.synthesize(make()).out
    for edits in (
        {"output.db": 3.0},
        {"output.polarity": -1},
        {"output.mode": "peak", "output.db": -1.0},
        {"output.format": "pcm24"},
    ):
        assert np.array_equal(R.synthesize(make(**edits)).out, base)
    pos, _ = R.render(make(**{"output.db": 0.0}))
    neg, _ = R.render(make(**{"output.db": 0.0, "output.polarity": -1}))
    assert np.array_equal(neg, -pos)
    assert np.array_equal(pos, base.astype(np.float32))
    quiet, _ = R.render(make(**{"output.db": -20.0}))
    assert np.array_equal(quiet, (base * 10 ** (-20 / 20)).astype(np.float32))


@pytest.mark.parametrize("target", [0.0, -0.5, -12.0, -60.0])
def test_peak_mode_hits_its_target(target):
    audio, report = R.render(make(**{"output.mode": "peak", "output.db": target, "drive.db": 6.0}))
    assert 20 * math.log10(np.max(np.abs(audio))) == pytest.approx(target, abs=1e-6)
    assert np.max(np.abs(audio)) <= 1.0
    assert not report.over


def test_silence_stays_silent():
    for mode in ("gain", "peak"):
        audio, report = R.render(make(**SILENT, **{"output.mode": mode, "output.db": 0.0}))
        assert np.all(audio == 0.0) and report.peak == 0.0
        assert "silent" in str(report)
        assert report.takes[0].headroom is None


def test_overs_are_reported_never_clipped():
    audio, report = R.render(make(**{"output.db": 12.0, "drive.db": 12.0}))
    assert report.over and report.peak > 1.0
    assert np.max(np.abs(audio)) == pytest.approx(report.peak)
    assert "over full scale" in str(report) and "lower output.db by at least" in str(report)
    assert not R.render(make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0}))[1].over


def test_a_longer_layer_never_changes_earlier_samples():
    short = make()
    long = make(**{"noise[2].amp": amp((0, 0), (1.5, 1), (40, 0.4), (140, 0.08), (260, 0.02), (600, 0))})
    a, b = R.synthesize(short).out, R.synthesize(long).out
    # PCHIP slopes are local: the new value at 260 ms moves the slope at 140 ms, which reaches back
    # only to the knot at 40 ms, and the decimator is causal.
    m = int(0.040 * 48000)
    assert np.array_equal(a[:m], b[:m])
    assert b.size > a.size


def test_report():
    audio, report = R.render(loads(EXAMPLE))
    assert report.samples == audio.size == 12571
    assert report.exported_ms == pytest.approx(1000 * 12571 / 48000)
    assert report.source_ms == 260.0
    assert report.internal_rate == 1536000 and report.factor == 32
    t = report.takes[0]
    assert t.true_peak >= t.peak and t.seed == 1 and t.gain_db == 0.0
    assert report.inside and t.domain.margin_db > 20
    text = str(report)
    assert "185.000 Hz, F#3 +0.0 cents; modes 185.0, 294.7, 395.2 Hz" in text
    assert "260.0 ms source, 261.9 ms exported (12571 samples at 48000 Hz, float32), seed 1" in text
    assert "predicted aliasing" in text and "margin" in text and "render" in text
    far = R.render(make(**HOT))[1]
    assert not far.inside and "OUTSIDE" in str(far) and "halyard verify" in str(far)
    # Output gain never moves the domain status: it is judged at unity gain.
    assert R.render(make(**{"output.db": -40.0}))[1].takes[0].domain == t.domain
    clap = R.render(loads(cli.reference_text("clap")))[1]
    assert "(the body is silent)" in str(clap) and not clap.body_sounds


def test_domain_rule():
    kinks = V.Kinks((), *(np.zeros(0),) * 5)

    def synth(knee_step, kink, peak):
        knees = V.Knees(knee_step, np.zeros(0), np.zeros(0))
        return R.Synthesis(np.array([peak]), 1536000, 32, 1, V.Slew(knee_step, knees), kinks, kink, 1.0, 24000.0)

    # A linear path with no kinks predicts nothing.
    d = R.domain(synth(0.0, 0.0, 1.0))
    assert d.inside and d.margin_db == math.inf
    # The saturation term alone crosses the limit at q = π² / ln(SATURATION / LIMIT); here α = 1, so q is the step.
    edge = math.pi**2 / math.log(R.SATURATION / R.LIMIT)
    assert R.domain(synth(edge * 0.999, 0.0, 1.0)).inside
    assert not R.domain(synth(edge * 1.001, 0.0, 1.0)).inside
    # Kinks add; near-silent renders use the absolute floor.
    assert not R.domain(synth(0.0, 2e-5, 1.0)).inside
    assert R.domain(synth(0.0, 0.9e-7, 1e-3)).inside
    assert not R.domain(synth(0.0, 1.1e-7, 1e-3)).inside
    assert R.allowed(0.02) == pytest.approx(2e-7) and R.allowed(0.0099) == 1e-7


def test_true_peak_catches_intersample_peaks():
    from scipy.signal.windows import tukey

    from halyard_synth.loudness import true_peak

    n = np.arange(4800)
    x = (0.5 * tukey(n.size, 0.2) * np.sin(2 * np.pi * 12000 / 48000 * n + np.pi / 4)).astype(np.float32)
    assert np.max(np.abs(x)) == pytest.approx(0.5 * math.sqrt(0.5), rel=1e-6)
    assert true_peak(x) == pytest.approx(0.5, rel=0.01)


# One render knows the rule at every drive.


def rule_inside(edits: dict, drive: float) -> bool:
    """The validated-domain status from a full render at that drive."""
    return R.domain(R.synthesize(make(**{**edits, "drive.db": drive}))).inside


@pytest.mark.parametrize("edits", [{}, {"noise[2].level_db": 6.0}, {"body.level_db": 12.0, "body.damping": 3.0}])
def test_one_render_knows_the_knee_slew_and_kink_bound_at_every_drive(edits):
    base = R.synthesize(make(**edits, **{"drive.db": 0.0}))
    for d in (0.5, 6.0, 18.0, 30.0, 48.0):
        s = R.synthesize(make(**edits, **{"drive.db": d}))
        a = V.drive_alpha(d)
        assert base.slew.knee(a) == pytest.approx(s.slew.knee(a), rel=1e-12)
        assert V.kink_bound(base.kinks, a, s.rate, s.band) == s.kink
        assert V.kink_bounds(base.kinks, [a], s.rate, s.band)[0] == pytest.approx(s.kink, rel=1e-12)
        # The probes give the exact output at the render's own drive, so their peak is the output peak.
        assert s.probes.peak([a])[0] == pytest.approx(np.max(np.abs(s.out)), rel=1e-12)
        assert base.probes.peak([a])[0] <= np.max(np.abs(s.out)) * (1 + 1e-12)  # elsewhere, a lower bound


@pytest.mark.parametrize(
    "edits", [{"drive.db": 6.0}, {**HOT}, {**FAST, "drive.db": 0.0}, {"noise[2].level_db": 3.0, "drive.db": 10.0}]
)
def test_drive_headroom_finds_where_the_status_changes(edits):
    report = R.render(make(**edits))[1]
    drive = make(**edits).drive.db
    h = report.takes[0].headroom
    assert h.db == pytest.approx(h.edge_db - drive)
    # Full renders on either side of the reported edge.
    assert rule_inside(edits, h.edge_db - 0.01) and not rule_inside(edits, h.edge_db + 0.01)
    assert (h.db > 0) == report.inside


def test_drive_headroom_never_skips_an_outside_island():
    # A frontier step with reach 0.1 counts only while α ≤ 10, so with q = α·0.1 up to there and α·0.05 after,
    # the rule is outside from α = 10·q* (q* = π²/ln(SATURATION/LIMIT)), inside again past α = 10, and outside
    # for good from α = 20·q*.
    knees = V.Knees(0.05, np.array([0.1]), np.array([0.1]))
    kinks = V.Kinks((), *(np.zeros(0),) * 5)
    probes = R.Probes(np.array([50.0]), ((0, np.array([1.0])),))  # a saturated probe: the peak is 1 at every drive
    alpha = V.drive_alpha(12.0)

    def at(a):
        return R.Synthesis(np.array([1.0]), 1536000, 32, 1, V.Slew(0.1, knees), kinks, 0.0, a, 24000.0, probes)

    threshold = math.pi**2 / math.log(R.SATURATION / R.LIMIT)
    island = threshold / 0.1
    assert R.ratios(at(alpha), [island * 0.999])[0] < 1 < R.ratios(at(alpha), [island * 1.001])[0]
    assert R.ratios(at(alpha), [10.0 * 1.0001])[0] < 1  # inside again just past the drop
    d = R.domain(at(alpha))
    assert d.inside
    h = R.drive_headroom(at(alpha), 12.0, d)
    assert R._alphas([h.edge_db])[0] == pytest.approx(island, rel=1e-5)
    # From inside the island, the way back in is below it.
    inside_island = 0.5 * (island + 10.0)
    drive = float(R._drives(inside_island))
    s = at(inside_island)
    assert not R.domain(s).inside
    h = R.drive_headroom(s, drive, R.domain(s))
    assert h.db < 0 and R._alphas([h.edge_db])[0] == pytest.approx(island, rel=1e-5)


def test_drive_headroom_at_the_extremes():
    quiet = R.render(make(**{"body.level_db": -30.0, "noise[1].level_db": -45.0, "noise[2].level_db": -45.0}))[1]
    assert quiet.takes[0].headroom.db == math.inf and "inside at every drive up to 48 dB" in str(quiet)
    # A 30 µs spike of noise aliases above the limit with no drive at all.
    spike = {
        "body.level_db": -math.inf,
        "noise[2].level_db": -math.inf,
        "drive.db": 0.0,
        "noise[1].band_hz": [1000.0, 20000.0],
        "noise[1].amp": amp((0, 0), (0.01, 1), (0.02, 0.3), (0.03, 0)),
    }
    fast = R.render(make(**spike))[1]
    assert not fast.inside and fast.takes[0].headroom.db == -math.inf
    assert "outside even without drive: the kinks alone exceed the limit" in str(fast)


def test_the_cause_names_what_to_change():
    fast = R.render(make(**FAST, **{"drive.db": 0.0}))[1]
    c = fast.takes[0].cause
    assert c.dominant == "kinks" and c.knot.startswith("noise[1].amp[") and 0 < c.share <= 1
    assert c.knot_ms == {"noise[1].amp[1]": 0.0, "noise[1].amp[2]": 0.01}[c.knot]
    assert "mostly kinks" in str(fast) and f"at {c.knot} ({c.knot_ms:g} ms)" in str(fast)
    assert c.saturation == 0.0 and c.kinks == pytest.approx(fast.takes[0].domain.predicted)
    hot = R.render(make(**HOT))[1]
    c = hot.takes[0].cause
    assert c.dominant == "saturation" and "mostly saturation" in str(hot)
    assert c.saturation + c.kinks == pytest.approx(hot.takes[0].domain.predicted)
    assert "lower drive.db to" in str(hot)


def test_knot_times_by_path():
    p = loads(EXAMPLE)
    assert R.knot_ms(p, "body.amp[2]") == 0.4
    assert R.knot_ms(p, "body.pitch[2]") == 15.0
    assert R.knot_ms(p, "noise[2].amp[5]") == 260.0


# Takes.


def test_takes_differ_only_in_their_seed():
    p = make(**{"drive.db": 9.0})
    audio, report = R.render_takes(p, 3)
    assert [t.seed for t in report.takes] == [1, 2, 3] and [t.number for t in report.takes] == [1, 2, 3]
    for j, a in enumerate(audio):
        assert np.array_equal(a, R.render(dataclasses.replace(p, seed=1 + j))[0])
    assert not np.array_equal(audio[0], audio[1])
    # Without noise the takes are the same sound.
    body = R.render_takes(make(**{"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}), 2)[0]
    assert np.array_equal(body[0], body[1])


def test_a_set_shares_one_gain():
    p = make(**{"output.mode": "peak", "output.db": -1.0})
    audio, report = R.render_takes(p, 4)
    peaks = [float(np.max(np.abs(a))) for a in audio]
    assert 20 * math.log10(max(peaks)) == pytest.approx(-1.0, abs=1e-6)
    assert len(set(peaks)) == 4  # the others keep their natural levels
    unity = [np.max(np.abs(R.synthesize(dataclasses.replace(p, seed=s)).out)) for s in (1, 2, 3, 4)]
    np.testing.assert_allclose(np.array(peaks) / max(peaks), np.array(unity) / max(unity), rtol=1e-6)
    assert report.set_gain_db == pytest.approx(-1.0 - 20 * math.log10(max(unity)), abs=1e-9)


def test_level_matching_scales_each_take_to_the_first():
    p = loads(cli.reference_text("snap"))
    loose, report = R.render_takes(p, 4)
    matched, mreport = R.render_takes(p, 4, level_match=True)
    rate = p.output.sample_rate
    first = momentary_max(matched[0], rate)
    for a in matched:
        assert momentary_max(a, rate) == pytest.approx(first, abs=0.01)
    assert mreport.loudness_spread <= 0.01 < report.loudness_spread
    assert mreport.takes[0].gain_db == 0.0 and any(t.gain_db != 0.0 for t in mreport.takes)
    assert "level-matched to take 1" in str(mreport)


def test_the_report_flags_uneven_sets():
    loud_click = make(
        **{
            "body.level_db": -math.inf,
            "noise[2].level_db": -math.inf,
            "noise[1].level_db": 0.0,
            "noise[1].amp": amp((0, 0), (0.02, 1), (0.12, 0.7), (0.5, 0.1), (1, 0)),
        }
    )
    report = R.render_takes(loud_click, 8)[1]
    assert report.uneven and "consider --level-match" in str(report)
    text = str(report)
    assert "8 takes sharing one gain" in text and "take 8" in text and "seeds 1–8" in text


def test_take_seeds_stay_in_range():
    with pytest.raises(R.RenderError):
        R.take_seeds(make(seed=2**63 - 2), 3)
    assert R.take_seeds(make(seed=2**63 - 3), 3)[-1] == 2**63 - 1
    for bad in (0, 129, True):
        with pytest.raises(R.RenderError):
            R.take_seeds(make(), bad)
