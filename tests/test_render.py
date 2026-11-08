# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering: the timing contract, the output stage, the validated domain, drive headroom and takes (§3, §5, §7)."""

import dataclasses
import math
import multiprocessing
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from halyard_synth import cli, patch, render, voice, wav
from halyard_synth.loudness import momentary_max
from halyard_synth.patch import SAMPLE_RATES, loads

from .conftest import EXAMPLE, make

type Edits = dict[str, object]

WIRES_MS, BODY_MS = 260.0, 180.0  # where the example's wires and body end
SAMPLES = 12571  # the example's output samples at 48 kHz
INTERNAL, FACTOR = 1536000, 32  # the 48 kHz family's internal rate and factor
MARGIN_DB = 20.0  # the example is this far inside the domain, at least
PEAK_DB = 1e-6  # peak mode reaches its target within this
MATCHED_DB = 0.01  # level matching brings the takes within this
TAKES = 4
STICK_MS = 0.4  # the body's peak, its pitch's end and the wires' end in the example
PITCH_MS = 15.0
PARABOLA = 1e-7  # the parabola finds the reconstruction's crest between internal samples to this, relatively
FLOOR = 1e-7  # the absolute bound of near-silent renders
BISECTION = 1e-5  # the headroom's edge is refined to this, relatively in α


def amp(*points: tuple[float, float]) -> list[dict[str, float]]:
    """Write knots as TOML parses them."""
    return [{"ms": m, "gain": g} for m, g in points]


SILENT = {"body.level_db": -math.inf, "noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}
FAST = {
    "noise[1].level_db": 12.0,
    "noise[1].band_hz": [1000.0, 20000.0],
    "noise[1].amp": amp((0, 0), (0.01, 1), (0.11, 0.7), (2, 0.1), (6, 0)),
}
HOT = {"noise[2].level_db": 6.0, "drive.db": 24.0}


def test_output_length_follows_the_timing_contract() -> None:
    """Find N internal samples and (N + L − 2) / factor + 1 outputs at every rate, the first exactly 0."""
    for rate in SAMPLE_RATES:
        p = make(**{"output.sample_rate": rate})
        s = render.synthesize(p)
        internal, factor = render.rates(rate)
        n = voice.sample_count(WIRES_MS / 1000, internal)
        length = render.decimator(render.FAMILY[rate]).taps.size
        assert s.source_samples == n
        assert s.out.size == (n + length - 2) // factor + 1
        audio, report = render.render(p)
        assert audio.size == s.out.size == report.samples
        assert audio.dtype == np.float32
        # Every shape starts at 0, so every file starts at exactly zero.
        assert audio[0] == 0.0


def test_silent_sources_never_set_the_length() -> None:
    """Find the length the audible sources' latest end: a silent body never lengthens the file."""
    loud = render.render(make())[1]
    no_wires = render.render(make(**{"noise[2].level_db": -math.inf}))[1]
    assert loud.source_ms == WIRES_MS
    assert no_wires.source_ms == BODY_MS
    assert no_wires.samples < loud.samples
    # A silent body still gates the rattle, inside the layer it gates, but never lengthens the file.
    long_body = make(**{"body.level_db": -math.inf, "body.amp": amp((0, 0), (1, 1), (900, 0))})
    assert render.render(long_body)[1].source_ms == WIRES_MS


def test_repeating_a_render_changes_nothing() -> None:
    """Render twice and find the same samples."""
    p = make(**{"drive.db": 18.0, "noise[2].rattle": 0.8})
    assert np.array_equal(render.render(p)[0], render.render(p)[0])


def test_output_gain_and_polarity_never_touch_synthesis() -> None:
    """Find gain, polarity, mode and format applied after synthesis, exactly, never changing it."""
    base = render.synthesize(make()).out
    for edits in (
        {"output.db": 3.0},
        {"output.polarity": -1},
        {"output.mode": "peak", "output.db": -1.0},
        {"output.format": "pcm24"},
    ):
        assert np.array_equal(render.synthesize(make(**edits)).out, base)
    pos, _ = render.render(make(**{"output.db": 0.0}))
    neg, _ = render.render(make(**{"output.db": 0.0, "output.polarity": -1}))
    assert np.array_equal(neg, -pos)
    assert np.array_equal(pos, base.astype(np.float32))
    quiet, _ = render.render(make(**{"output.db": -20.0}))
    assert np.array_equal(quiet, (base * 10 ** (-20 / 20)).astype(np.float32))


@pytest.mark.parametrize("target", [0.0, -0.5, -12.0, -60.0])
def test_peak_mode_hits_its_target(target: float) -> None:
    """Bring the peak to targets from 0 to −60 dBFS, never over."""
    audio, report = render.render(make(**{"output.mode": "peak", "output.db": target, "drive.db": 6.0}))
    assert 20 * math.log10(np.max(np.abs(audio))) == pytest.approx(target, abs=PEAK_DB)
    assert np.max(np.abs(audio)) <= 1.0
    assert not report.over


def test_silence_stays_silent() -> None:
    """Render silence as zeros in either mode, reported silent, with no headroom."""
    for mode in ("gain", "peak"):
        audio, report = render.render(make(**SILENT, **{"output.mode": mode, "output.db": 0.0}))
        assert np.all(audio == 0.0)
        assert report.peak == 0.0
        assert "silent" in str(report)
        assert report.takes[0].headroom is None


def test_overs_are_reported_never_clipped() -> None:
    """Report a gain-mode render over full scale, and say what to lower, without clipping it."""
    audio, report = render.render(make(**{"output.db": 12.0, "drive.db": 12.0}))
    assert report.over
    assert report.peak > 1.0
    assert np.max(np.abs(audio)) == pytest.approx(report.peak)
    assert "over full scale" in str(report)
    assert "lower output.db by at least" in str(report)
    assert not render.render(make(**{"output.mode": "peak", "output.db": 0.0, "drive.db": 12.0}))[1].over
    takes = render.render_takes(make(**{"output.db": 12.0, "drive.db": 12.0}), 2)[1]
    assert takes.over
    assert "peak     +" in str(takes)  # a set's report says how far over it is, once


def test_a_longer_layer_never_changes_earlier_samples() -> None:
    """Lengthen a layer's tail and find the output before its last unchanged knot unchanged."""
    short = make()
    long = make(**{"noise[2].amp": amp((0, 0), (1.5, 1), (40, 0.4), (140, 0.08), (260, 0.02), (600, 0))})
    a, b = render.synthesize(short).out, render.synthesize(long).out
    # PCHIP slopes are local: the new value at 260 ms moves the slope at 140 ms, which reaches back
    # only to the knot at 40 ms, and the decimator is causal.
    m = int(0.040 * 48000)
    assert np.array_equal(a[:m], b[:m])
    assert b.size > a.size


def test_report() -> None:
    """Find the report's tuning, length, levels and domain, its status judged at unity gain, and a silent body named."""
    audio, report = render.render(loads(EXAMPLE))
    assert report.samples == audio.size == SAMPLES
    assert report.exported_ms == pytest.approx(1000 * SAMPLES / 48000)
    assert report.source_ms == WIRES_MS
    assert report.internal_rate == INTERNAL
    assert report.factor == FACTOR
    t = report.takes[0]
    assert t.true_peak >= t.peak
    assert t.seed == 1
    assert t.gain_db == 0.0
    assert report.inside
    assert t.domain.margin_db > MARGIN_DB
    text = str(report)
    assert "185.000 Hz, F#3 +0.0 cents; modes 185.0, 294.7, 395.2 Hz" in text
    assert "260.0 ms source, 261.9 ms exported (12571 samples at 48000 Hz, float32), seed 1" in text
    assert "predicted aliasing" in text
    assert "margin" in text
    assert "render" in text
    far = render.render(make(**HOT))[1]
    assert not far.inside
    assert "OUTSIDE" in str(far)
    assert "halyard verify" in str(far)
    # Output gain never moves the domain status: it is judged at unity gain.
    assert render.render(make(**{"output.db": -40.0}))[1].takes[0].domain == t.domain
    clap = render.render(loads(cli.reference_text("clap")))[1]
    assert "(the body is silent)" in str(clap)
    assert not clap.body_sounds
    rims = [
        {
            "level_db": -6.0,
            "damping": 1.0,
            "modes": [{"hz": f, "level_db": 0.0}],
            "amp": [{"ms": m, "gain": g} for m, g in ((0, 0), (0.2, 1), (6, 0))],
        }
        for f in (1630.0, 700.0)
    ]
    rims[1]["level_db"] = -math.inf
    struck = str(render.render(make(rim=rims))[1])
    assert "; rim 1 1630.0 Hz" in struck
    assert "rim 2" not in struck  # a silent rim has no tuning to report


def test_domain_rule() -> None:
    """Find the saturation term crossing the limit where it should, kinks adding, and the floor for near-silence."""
    kinks = voice.Kinks((), *(np.zeros(0),) * 5)

    def synth(knee_step: float, kink: float, peak: float) -> render.Synthesis:
        knees = voice.Knees(knee_step, np.zeros(0), np.zeros(0))
        return render.Synthesis(
            np.array([peak]),
            1536000,
            32,
            1,
            voice.Slew(knee_step, knees),
            kinks,
            kink,
            1.0,
            24000.0,
            peak,
        )

    # A linear path with no kinks predicts nothing.
    d = render.domain(synth(0.0, 0.0, 1.0))
    assert d.inside
    assert d.margin_db == math.inf
    # The saturation term alone crosses the limit at q = π² / ln(SATURATION / LIMIT); here α = 1, so q is the step.
    edge = math.pi**2 / math.log(render.SATURATION / render.LIMIT)
    assert render.domain(synth(edge * 0.999, 0.0, 1.0)).inside
    assert not render.domain(synth(edge * 1.001, 0.0, 1.0)).inside
    # Kinks add; near-silent renders use the absolute floor.
    assert not render.domain(synth(0.0, 2e-5, 1.0)).inside
    assert render.domain(synth(0.0, 0.9e-7, 1e-3)).inside
    assert not render.domain(synth(0.0, 1.1e-7, 1e-3)).inside
    assert render.allowed(0.02) == pytest.approx(2e-7)
    assert render.allowed(0.0099) == FLOOR


@pytest.mark.parametrize("hz", [1000.0, 12000.0, 19900.0])
def test_the_true_peak_is_the_reconstruction_s_crest(hz: float) -> None:
    """Find the true peak the crest of the output's reconstruction, evaluated directly between internal samples."""
    taps = render.decimator(48000).taps
    n = INTERNAL // 20
    length = n / INTERNAL

    def y(t: NDArray[np.float64], phase: float) -> NDArray[np.float64]:
        """Return a Hann-shaped sine at times t, a pure function of time as the internal signal is."""
        inside = (t >= 0) & (t < length)
        out: NDArray[np.float64] = np.where(inside, 0.25 * (1 - np.cos(2 * np.pi * t / length)), 0.0)
        return out * np.sin(2 * np.pi * hz * t + phase)

    k = np.arange(taps.size)
    for phase in (0.0, 1.0, 2.5):
        x = y(np.arange(n) / INTERNAL, phase)
        true = render.reconstructed_peak(x, taps)
        crest = int(np.argmax(np.abs(fftconvolve(x, taps))))
        between = np.linspace(-1, 1, 2001)  # a thousandth of an internal sample apart
        dense = max(abs(float(np.sum(taps * y((crest + u - k) / INTERNAL, phase)))) for u in between)
        assert true == pytest.approx(dense, rel=PARABOLA)
        assert true >= np.max(np.abs(render.decimate(taps, x, FACTOR))) * (1 - 1e-12)


def test_the_true_peak_of_edge_cases() -> None:
    """Find the true peak of nothing, of silence, of a crest at the very end, and of a flat top."""
    one = np.ones(1)
    assert render.reconstructed_peak(np.zeros(0), one) == 0.0
    assert render.reconstructed_peak(np.zeros(3), one) == 0.0
    assert render.reconstructed_peak(np.array([0.0, -0.5]), one) == 1 / 2  # a crest at the very end
    assert render.reconstructed_peak(np.ones(3), one) == 1.0  # flat: no parabola


def test_true_peak_mode_brings_the_true_peak_to_its_target() -> None:
    """Bring the true peak, not the sample peak, to the target, in a render and in a pack."""
    p = make(**{"output.mode": "true-peak", "output.db": -1.0, "drive.db": 12.0})
    audio, report = render.render(p)
    take = report.takes[0]
    assert 20 * math.log10(take.true_peak) == pytest.approx(-1.0, abs=1e-9)
    assert take.peak <= take.true_peak
    sample = render.render(make(**{"output.mode": "peak", "output.db": -1.0, "drive.db": 12.0}))[1].takes[0]
    assert 20 * math.log10(sample.peak) == pytest.approx(-1.0, abs=PEAK_DB)
    assert sample.true_peak >= sample.peak
    assert "dBTP true" in str(report)
    assert float(np.max(np.abs(audio))) == take.peak


# One render knows the rule at every drive.


def rule_inside(edits: Edits, drive: float) -> bool:
    """Say whether a patch is inside the validated domain, from a full render at that drive."""
    return render.domain(render.synthesize(make(**{**edits, "drive.db": drive}))).inside


@pytest.mark.parametrize("edits", [{}, {"noise[2].level_db": 6.0}, {"body.level_db": 12.0, "body.damping": 3.0}])
def test_one_render_knows_the_knee_slew_and_kink_bound_at_every_drive(edits: Edits) -> None:
    """Find one render's knee slew, kink bound and probes the same as full renders at other drives."""
    base = render.synthesize(make(**edits, **{"drive.db": 0.0}))
    for d in (0.5, 6.0, 18.0, 30.0, 48.0):
        s = render.synthesize(make(**edits, **{"drive.db": d}))
        a = voice.drive_alpha(d)
        assert base.slew.knee(a) == pytest.approx(s.slew.knee(a), rel=1e-12)
        assert voice.kink_bound(base.kinks, a, s.rate, s.band) == s.kink
        assert voice.kink_bounds(base.kinks, [a], s.rate, s.band)[0] == pytest.approx(s.kink, rel=1e-12)
        # The probes give the exact output at the render's own drive, so their peak is the output peak.
        assert s.probes is not None
        assert base.probes is not None
        assert s.probes.peak([a])[0] == pytest.approx(np.max(np.abs(s.out)), rel=1e-12)
        assert base.probes.peak([a])[0] <= np.max(np.abs(s.out)) * (1 + 1e-12)  # elsewhere, a lower bound


@pytest.mark.parametrize(
    "edits",
    [{"drive.db": 6.0}, {**HOT}, {**FAST, "drive.db": 0.0}, {"noise[2].level_db": 3.0, "drive.db": 10.0}],
)
def test_drive_headroom_finds_where_the_status_changes(edits: Edits) -> None:
    """Find full renders inside just below the reported edge and outside just above it."""
    report = render.render(make(**edits))[1]
    drive = make(**edits).drive.db
    h = report.takes[0].headroom
    assert h is not None
    assert h.db == pytest.approx(h.edge_db - drive)
    # Full renders on either side of the reported edge.
    assert rule_inside(edits, h.edge_db - 0.01)
    assert not rule_inside(edits, h.edge_db + 0.01)
    assert (h.db > 0) == report.inside


def test_drive_headroom_never_skips_an_outside_island() -> None:
    """Find the outside island just below a drop of the knee slew, from inside it and from below."""
    # A frontier step with reach 0.1 counts only while α ≤ 10, so with q = α·0.1 up to there and α·0.05 after,
    # the rule is outside from α = 10·q* (q* = π²/ln(SATURATION/LIMIT)), inside again past α = 10, and outside
    # for good from α = 20·q*.
    knees = voice.Knees(0.05, np.array([0.1]), np.array([0.1]))
    kinks = voice.Kinks((), *(np.zeros(0),) * 5)
    probes = render.Probes(np.array([50.0]), ((0, np.array([1.0])),))  # a saturated probe: the peak is 1 at every drive
    alpha = voice.drive_alpha(12.0)

    def at(a: float) -> render.Synthesis:
        return render.Synthesis(
            np.array([1.0]),
            1536000,
            32,
            1,
            voice.Slew(0.1, knees),
            kinks,
            0.0,
            a,
            24000.0,
            1.0,
            probes,
        )

    threshold = math.pi**2 / math.log(render.SATURATION / render.LIMIT)
    island = threshold / 0.1
    assert render.ratios(at(alpha), [island * 0.999])[0] < 1 < render.ratios(at(alpha), [island * 1.001])[0]
    assert render.ratios(at(alpha), [10.0 * 1.0001])[0] < 1  # inside again just past the drop
    d = render.domain(at(alpha))
    assert d.inside
    h = render.drive_headroom(at(alpha), 12.0, d)
    assert h is not None
    assert render.alphas_of([h.edge_db])[0] == pytest.approx(island, rel=BISECTION)
    # From inside the island, the way back in is below it.
    inside_island = 0.5 * (island + 10.0)
    drive = float(render.drives_of(inside_island))
    s = at(inside_island)
    assert not render.domain(s).inside
    h = render.drive_headroom(s, drive, render.domain(s))
    assert h is not None
    assert h.db < 0
    assert render.alphas_of([h.edge_db])[0] == pytest.approx(island, rel=BISECTION)


def test_drive_headroom_at_the_extremes() -> None:
    """Say a quiet patch stays inside at every drive, and a 30 µs spike is outside even without it."""
    quiet = render.render(make(**{"body.level_db": -30.0, "noise[1].level_db": -45.0, "noise[2].level_db": -45.0}))[1]
    headroom = quiet.takes[0].headroom
    assert headroom is not None
    assert headroom.db == math.inf
    assert "inside at every drive up to 48 dB" in str(quiet)
    # A 30 µs spike of noise aliases above the limit with no drive at all.
    spike = {
        "body.level_db": -math.inf,
        "noise[2].level_db": -math.inf,
        "drive.db": 0.0,
        "noise[1].band_hz": [1000.0, 20000.0],
        "noise[1].amp": amp((0, 0), (0.01, 1), (0.02, 0.3), (0.03, 0)),
    }
    fast = render.render(make(**spike))[1]
    assert not fast.inside
    headroom = fast.takes[0].headroom
    assert headroom is not None
    assert headroom.db == -math.inf
    assert "outside even without drive: the kinks alone exceed the limit" in str(fast)


def test_the_cause_names_what_to_change() -> None:
    """Name the larger term, and for kinks the knot that contributes most, its time and its share."""
    fast = render.render(make(**FAST, **{"drive.db": 0.0}))[1]
    c = fast.takes[0].cause
    assert c.dominant == "kinks"
    assert c.knot is not None
    assert c.knot.startswith("noise[1].amp[")
    assert 0 < c.share <= 1
    assert c.knot_ms == {"noise[1].amp[1]": 0.0, "noise[1].amp[2]": 0.01}[c.knot]
    assert "mostly kinks" in str(fast)
    assert f"at {c.knot} ({c.knot_ms:g} ms)" in str(fast)
    assert c.saturation == 0.0
    assert c.kinks == pytest.approx(fast.takes[0].domain.predicted)
    hot = render.render(make(**HOT))[1]
    c = hot.takes[0].cause
    assert c.dominant == "saturation"
    assert "mostly saturation" in str(hot)
    assert c.saturation + c.kinks == pytest.approx(hot.takes[0].domain.predicted)
    assert "lower drive.db to" in str(hot)


RIM_RISE_MS = 0.01  # a rim struck this fast


def test_knot_times_by_path() -> None:
    """Find a knot's time by its path, as the patch writes it."""
    p = loads(EXAMPLE)
    assert render.knot_ms(p, "body.amp[2]") == STICK_MS
    assert render.knot_ms(p, "body.pitch[2]") == PITCH_MS
    assert render.knot_ms(p, "noise[2].amp[5]") == WIRES_MS
    rim = {"level_db": 0.0, "damping": 1.0, "modes": [{"hz": 9000.0, "level_db": 0.0}]}
    struck = make(rim=[dict(rim, amp=[{"ms": m, "gain": g} for m, g in ((0, 0), (0.01, 1), (3, 0))])])
    assert render.knot_ms(struck, "rim[1].amp[2]") == RIM_RISE_MS
    # A rim struck in 10 µs leads the kinks, and the report names its knot.
    assert render.render(struck)[1].takes[0].cause.knot == "rim[1].amp[2]"


# Takes.


def test_takes_differ_only_in_their_seed() -> None:
    """Find each take its seed's render, and a patch without noise the same in every take."""
    p = make(**{"drive.db": 9.0})
    audio, report = render.render_takes(p, 3)
    assert [t.seed for t in report.takes] == [1, 2, 3]
    assert [t.number for t in report.takes] == [1, 2, 3]
    for j, a in enumerate(audio):
        assert np.array_equal(a, render.render(dataclasses.replace(p, seed=1 + j))[0])
    assert not np.array_equal(audio[0], audio[1])
    # Without noise the takes are the same sound.
    body = render.render_takes(make(**{"noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}), 2)[0]
    assert np.array_equal(body[0], body[1])


def test_a_set_shares_one_gain() -> None:
    """Bring the loudest take to the target with one gain, the others keeping their natural levels."""
    p = make(**{"output.mode": "peak", "output.db": -1.0})
    audio, report = render.render_takes(p, TAKES)
    peaks = [float(np.max(np.abs(a))) for a in audio]
    assert 20 * math.log10(max(peaks)) == pytest.approx(-1.0, abs=PEAK_DB)
    assert len(set(peaks)) == TAKES  # the others keep their natural levels
    unity = [np.max(np.abs(render.synthesize(dataclasses.replace(p, seed=s)).out)) for s in (1, 2, 3, 4)]
    np.testing.assert_allclose(np.array(peaks) / max(peaks), np.array(unity) / max(unity), rtol=1e-6)
    assert report.set_gain_db == pytest.approx(-1.0 - 20 * math.log10(max(unity)), abs=1e-9)


def test_level_matching_scales_each_take_to_the_first() -> None:
    """Level-match the snap reference: every take as loud as the first, which keeps its own gain."""
    p = loads(cli.reference_text("snap"))
    _loose, report = render.render_takes(p, 4)
    matched, mreport = render.render_takes(p, 4, level_match=True)
    rate = p.output.sample_rate
    first = momentary_max(matched[0], rate)
    for a in matched:
        assert momentary_max(a, rate) == pytest.approx(first, abs=MATCHED_DB)
    assert mreport.loudness_spread <= MATCHED_DB < report.loudness_spread
    assert mreport.takes[0].gain_db == 0.0
    assert any(t.gain_db != 0.0 for t in mreport.takes)
    assert "level-matched to take 1" in str(mreport)


def test_the_report_flags_uneven_sets() -> None:
    """Flag a set whose takes spread over 1 dB, and list every take."""
    loud_click = make(
        **{
            "body.level_db": -math.inf,
            "noise[2].level_db": -math.inf,
            "noise[1].level_db": 0.0,
            "noise[1].amp": amp((0, 0), (0.02, 1), (0.12, 0.7), (0.5, 0.1), (1, 0)),
        },
    )
    report = render.render_takes(loud_click, 8)[1]
    assert report.uneven
    assert "consider --level-match" in str(report)
    text = str(report)
    assert "8 takes sharing one gain" in text
    assert "take 8" in text
    assert "seeds 1–8" in text


def test_take_seeds_stay_in_range() -> None:
    """Refuse takes whose seeds would pass 2**63 − 1, and takes outside 1 … 128."""
    with pytest.raises(render.RenderError, match="pass 2\\*\\*63 - 1"):
        render.take_seeds(make(seed=2**63 - 2), 3)
    assert render.take_seeds(make(seed=2**63 - 3), 3)[-1] == 2**63 - 1
    for bad in (0, 129, True):
        with pytest.raises(render.RenderError, match="needs 1 to 128 takes"):
            render.take_seeds(make(), bad)


def test_the_loudest_outputs_are_picked_apart() -> None:
    """Pick the largest values first, never two neighbours nor a neighbour of one already taken, and never a zero."""
    mags = np.array([3.0, 2.0, 1.0, 0.0, 5.0])
    assert render.loudest(mags, 10, 4, []) == [14, 10, 12]
    assert render.loudest(mags, 10, 4, [13]) == [10]  # 14 and 12 are 13's neighbours
    assert render.loudest(mags, 10, 1, []) == [14]
    assert render.loudest(mags[:3], 10, 4, []) == [10, 12]  # every candidate seen, fewer picked than asked


def test_a_silent_set_reports_each_take_silent() -> None:
    """Report every take of a silent set as silent, with nothing to say about aliasing."""
    report = render.render_takes(make(**SILENT), 2)[1]
    text = str(report)
    assert text.count("seed 1: silent") == 1
    assert text.count("seed 2: silent") == 1
    assert "mostly" not in text
    assert all(t.detail(report.drive_db) is None for t in report.takes)


def test_explicit_taps_are_used_as_given() -> None:
    """Synthesize with taps given explicitly, the shipped ones or others, and find them used."""
    p = loads(EXAMPLE)
    shipped = render.decimator(48000).taps
    assert np.array_equal(render.synthesize(p, taps=shipped).out, render.synthesize(p).out)
    halved = render.synthesize(p, taps=shipped / 2).out
    np.testing.assert_allclose(halved, render.synthesize(p).out / 2, rtol=1e-12, atol=1e-15)


def test_a_crossing_is_named_by_its_segment_and_timed_by_itself() -> None:
    """Name a kink where a brush's sweep crosses a table by the segment it lies in, at its own time, not a knot's."""
    p = make(
        brush=[
            {
                "level_db": 0.0,
                "band_hz": [1500.0, 9000.0],
                "rattle": 0.0,
                "sweep": [{"ms": m, "ratio": r} for m, r in ((10.0, 1.0), (12.0, 2.0))],
                "amp": [{"ms": m, "gain": g} for m, g in ((0, 0), (15, 1), (200, 0.3), (400, 0))],
            },
        ],
    )
    s = render.synthesize(p)
    crossing = np.array(["–" in name for name in s.kinks.paths])
    k = s.kinks
    only = voice.Kinks(
        tuple(n for n, c in zip(k.paths, crossing, strict=True) if c),
        *(a[crossing] for a in (k.t, k.x, k.dx, k.d2, k.d3)),
    )
    assert set(only.paths) == {"brush[1].sweep[1–2]"}  # u = 3·log2 s crosses 1 and 2 between the knots
    c = render.cause_of(p, dataclasses.replace(s, kinks=only), render.domain(s))
    assert c.knot == "brush[1].sweep[1–2]"
    assert c.knot_ms is not None
    assert CROSSING_MS[0] < c.knot_ms < CROSSING_MS[1]


CROSSING_MS = (10.0, 12.0)  # the segment's knots


def test_takes_that_differ_render_as_played_and_their_files_render_again(tmp_path: Path) -> None:
    """Render takes that differ in strength, find their loudness spread, and each file's own patch render it again."""
    p = dataclasses.replace(loads(cli.reference_text("house")), takes=patch.Takes(3.0, 0.0))
    audio, report = render.render_takes(p, 6)
    loudness = [t.loudness for t in report.takes]
    assert max(loudness) - min(loudness) > 1.0  # struck up to 3 dB harder or softer
    for a, take in zip(audio, report.takes, strict=True):
        path = tmp_path / f"take{take.number}.wav"
        render.export(path, a, p, report, take)
        meta = wav.read(path).meta
        assert meta is not None
        own = loads(meta["patch"])
        assert own.takes == p.takes
        again, _ = render.render(own)  # a gain-mode set: every take at the patch's own gain
        np.testing.assert_allclose(again, a, rtol=0, atol=1e-7)


def test_level_matching_would_undo_takes_and_is_refused() -> None:
    """Refuse to match the loudness of takes that [takes] strikes harder and softer on purpose."""
    p = dataclasses.replace(loads(cli.reference_text("house")), takes=patch.Takes(2.0, 0.0))
    with pytest.raises(render.RenderError, match="would undo"):
        render.render_takes(p, 3, level_match=True)


def _renders_as(text: str, expected: bytes) -> None:
    """Render a patch in this process and fail unless it is exactly `expected`."""
    if render.render(loads(text))[0].tobytes() != expected:
        raise AssertionError


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no fork")
def test_a_process_forked_after_a_render_renders_too() -> None:
    """Render, fork as multiprocessing does by default on Linux before Python 3.14, and render the same in the child.

    The child inherits the parent's caches and locks but none of its threads, so nothing may wait on those threads.
    """
    expected = render.render(loads(EXAMPLE))[0].tobytes()
    with warnings.catch_warnings():  # Python warns that forking a process with threads may deadlock: the point here
        warnings.simplefilter("ignore", DeprecationWarning)
        child = multiprocessing.get_context("fork").Process(target=_renders_as, args=(EXAMPLE, expected))
        child.start()
    child.join(timeout=60)
    alive = child.is_alive()
    child.kill()
    child.join()
    assert not alive, "the forked child hung"
    assert child.exitcode == 0
