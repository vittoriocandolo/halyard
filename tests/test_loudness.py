import math

import numpy as np
import pytest
from conftest import EXAMPLE, make
from scipy.signal import sosfreqz

from snare import render as R
from snare.loudness import k_weighting, momentary_max
from snare.patch import loads

# ITU-R BS.1770-4, table 1 and table 2: the K-weighting biquads at 48 kHz.
SHELF_48K = [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585]
HIGHPASS_48K = [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]
RATES = (44100, 48000, 88200, 96000)


def test_k_weighting_reproduces_the_published_coefficients():
    sos = k_weighting(48000)
    assert np.max(np.abs(sos[0] - SHELF_48K)) < 1e-14
    assert np.max(np.abs(sos[1] - HIGHPASS_48K)) < 1e-14


@pytest.mark.parametrize("rate", [44100, 88200, 96000])
def test_k_weighting_is_the_same_filter_at_every_rate(rate):
    f = np.geomspace(20, 20000, 400)
    ref = np.abs(sosfreqz(k_weighting(48000), worN=f, fs=48000)[1])
    got = np.abs(sosfreqz(k_weighting(rate), worN=f, fs=rate)[1])
    assert np.max(np.abs(20 * np.log10(got / ref))) < 0.03


@pytest.mark.parametrize("rate", RATES)
def test_ebu_tech_3341_sine(rate):
    # EBU Tech 3341, test case 1: a 1 kHz sine at −23 dBFS in both channels reads −23.0 ± 0.1 LUFS.
    # This meter reads one channel, so it reads 10·log10(2) lower; the level scales linearly.
    t = np.arange(4 * rate) / rate
    for level in (-23.0, -33.0, -3.0):
        x = 10 ** (level / 20) * np.sin(2 * np.pi * 1000 * t)
        assert momentary_max(x, rate) + 10 * math.log10(2) == pytest.approx(level, abs=0.1)


def test_the_largest_window_counts_every_position():
    rate = 48000
    t = np.arange(2 * rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    steady = momentary_max(tone, rate)
    # A 100 ms burst fills a quarter of the 400 ms window.
    burst = np.concatenate((tone[: rate // 10], np.zeros(rate)))
    assert momentary_max(burst, rate) == pytest.approx(steady + 10 * math.log10(0.25), abs=0.02)
    # A burst at the very end of a file reads the same: windows run past the end.
    assert momentary_max(np.concatenate((np.zeros(rate), tone[: rate // 10])), rate) == pytest.approx(
        momentary_max(burst, rate), abs=1e-9
    )
    assert momentary_max(np.zeros(1000), rate) == -math.inf


def test_the_report_gives_the_loudness_of_the_export():
    audio, report = R.render(loads(EXAMPLE))
    assert report.takes[0].loudness == momentary_max(audio, 48000)
    assert "LUFS, the largest 400 ms momentary loudness (BS.1770)" in str(report)
    quieter = R.render(make(**{"output.db": -18.0}))[1]
    assert quieter.takes[0].loudness == pytest.approx(report.takes[0].loudness - 12.0, abs=1e-5)
    silent = R.render(
        make(**{"body.level_db": -math.inf, "noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf})
    )[1]
    assert silent.takes[0].loudness == -math.inf and "loudness silent" in str(silent)


def test_loudness_agrees_across_export_rates():
    # The same sound reads the same at every rate: the K-weighting is one analog filter.
    levels = [R.render(make(**{"output.sample_rate": rate}))[1].takes[0].loudness for rate in RATES]
    assert max(levels) - min(levels) < 0.05
