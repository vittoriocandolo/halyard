# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Loudness after ITU-R BS.1770: the K-weighting at every rate, EBU Tech 3341, and the report's figure."""

import math

import numpy as np
import pytest
from scipy.signal import sosfreqz

from halyard_synth import render
from halyard_synth.loudness import k_weighting, momentary_max
from halyard_synth.patch import loads

from .conftest import EXAMPLE, make

# ITU-R BS.1770-4, table 1 and table 2: the K-weighting biquads at 48 kHz.
SHELF_48K = [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585]
HIGHPASS_48K = [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]
RATES = (44100, 48000, 88200, 96000)
COEFFICIENTS = 1e-14  # how close the derived biquads come to the published ones
RESPONSE_DB = 0.03  # how close the filter at another rate comes to the 48 kHz one, 20 Hz – 20 kHz
METER_DB = 0.1  # EBU Tech 3341's tolerance
WINDOW_DB = 0.02  # a burst a quarter of the window long reads 6 dB lower, within this
RATE_SPREAD_DB = 0.05  # the same sound reads the same at every rate, within this
GAIN_DB = 1e-5  # a change of gain reads as itself, within this


def test_k_weighting_reproduces_the_published_coefficients() -> None:
    """Derive the 48 kHz biquads from the analog filters and find BS.1770's coefficients."""
    sos = k_weighting(48000)
    assert np.max(np.abs(sos[0] - SHELF_48K)) < COEFFICIENTS
    assert np.max(np.abs(sos[1] - HIGHPASS_48K)) < COEFFICIENTS


@pytest.mark.parametrize("rate", [44100, 88200, 96000])
def test_k_weighting_is_the_same_filter_at_every_rate(rate: int) -> None:
    """Find the K-weighting at 44.1, 88.2 and 96 kHz within 0.03 dB of the 48 kHz one across the audio band."""
    f = np.geomspace(20, 20000, 400)
    ref = np.abs(sosfreqz(k_weighting(48000), worN=f, fs=48000)[1])
    got = np.abs(sosfreqz(k_weighting(rate), worN=f, fs=rate)[1])
    assert np.max(np.abs(20 * np.log10(got / ref))) < RESPONSE_DB


@pytest.mark.parametrize("rate", RATES)
def test_ebu_tech_3341_sine(rate: int) -> None:
    """Read EBU Tech 3341's 1 kHz sine, and the same sine 10 dB softer and 20 dB louder, at every rate."""
    # EBU Tech 3341, test case 1: a 1 kHz sine at −23 dBFS in both channels reads −23.0 ± 0.1 LUFS.
    # This meter reads one channel, so it reads 10·log10(2) lower; the level scales linearly.
    t = np.arange(4 * rate) / rate
    for level in (-23.0, -33.0, -3.0):
        x = 10 ** (level / 20) * np.sin(2 * np.pi * 1000 * t)
        assert momentary_max(x, rate) + 10 * math.log10(2) == pytest.approx(level, abs=METER_DB)


def test_the_largest_window_counts_every_position() -> None:
    """Find a short burst read over the whole window, at the start of a file or its very end, and silence as −inf."""
    rate = 48000
    t = np.arange(2 * rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    steady = momentary_max(tone, rate)
    # A 100 ms burst fills a quarter of the 400 ms window.
    burst = np.concatenate((tone[: rate // 10], np.zeros(rate)))
    assert momentary_max(burst, rate) == pytest.approx(steady + 10 * math.log10(0.25), abs=WINDOW_DB)
    # A burst at the very end of a file reads the same: windows run past the end.
    assert momentary_max(np.concatenate((np.zeros(rate), tone[: rate // 10])), rate) == pytest.approx(
        momentary_max(burst, rate),
        abs=1e-9,
    )
    assert momentary_max(np.zeros(1000), rate) == -math.inf


def test_the_report_gives_the_loudness_of_the_export() -> None:
    """Find the report's loudness that of the exported audio, following its gain, and silence named."""
    audio, report = render.render(loads(EXAMPLE))
    assert report.takes[0].loudness == momentary_max(audio, 48000)
    assert "LUFS, the largest 400 ms momentary loudness (BS.1770)" in str(report)
    quieter = render.render(make(**{"output.db": -18.0}))[1]
    assert quieter.takes[0].loudness == pytest.approx(report.takes[0].loudness - 12.0, abs=GAIN_DB)
    silent = render.render(
        make(**{"body.level_db": -math.inf, "noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}),
    )[1]
    assert silent.takes[0].loudness == -math.inf
    assert "loudness silent" in str(silent)


def test_loudness_agrees_across_export_rates() -> None:
    """Render one patch at every export rate and find it equally loud."""
    # The same sound reads the same at every rate: the K-weighting is one analog filter.
    levels = [render.render(make(**{"output.sample_rate": rate}))[1].takes[0].loudness for rate in RATES]
    assert max(levels) - min(levels) < RATE_SPREAD_DB
