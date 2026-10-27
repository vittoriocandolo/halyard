# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Loudness after ITU-R BS.1770, for the report, level matching, packs and audition (plan §7.4).

K-weighting is a high shelf of about +4 dB above 1.7 kHz followed by a
high-pass at about 38 Hz. The standard publishes both biquads at 48 kHz only.
The analog parameters below (as derived for libebur128) reproduce those
coefficients, and the bilinear transform with pre-warping gives the same
filters at every other rate.

The momentary loudness is −0.691 + 10·log₁₀ of the mean square of the
K-weighted signal over 400 ms (one channel, weight 1). For a one-shot the
report gives the largest value over every window position, which a meter
updating every 100 ms can only approach from below.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import resample_poly, sosfilt

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    type Floats = NDArray[np.float64]

SHELF_HZ, SHELF_DB, SHELF_Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
SHELF_VB = 0.4996667741545416  # the shelf's mid-band exponent
HIGHPASS_HZ, HIGHPASS_Q = 38.13547087602444, 0.5003270373238773
WINDOW_S = 0.4
OFFSET = -0.691


def k_weighting(rate: int) -> Floats:
    """Return the two K-weighting biquads at `rate`, as second-order sections."""
    k = math.tan(math.pi * SHELF_HZ / rate)
    vh = 10.0 ** (SHELF_DB / 20.0)
    vb = vh**SHELF_VB
    a0 = 1.0 + k / SHELF_Q + k * k
    shelf = [
        (vh + vb * k / SHELF_Q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / SHELF_Q + k * k) / a0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / SHELF_Q + k * k) / a0,
    ]
    k = math.tan(math.pi * HIGHPASS_HZ / rate)
    a0 = 1.0 + k / HIGHPASS_Q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / HIGHPASS_Q + k * k) / a0]
    return np.array([shelf, highpass])


def momentary_max(x: ArrayLike, rate: int) -> float:
    """Return the largest 400 ms momentary loudness in LUFS, over every window position; −inf for silence.

    The signal is silent before and after the file, so windows that run past
    its end are counted, and a hit shorter than the window reads as its whole
    energy spread over 400 ms.
    """
    w = round(WINDOW_S * rate)
    z = sosfilt(k_weighting(rate), np.concatenate((np.asarray(x, dtype=float), np.zeros(w))))
    energy = np.concatenate(([0.0], np.cumsum(z * z)))
    ms = float(np.max(energy[w:] - energy[:-w])) / w
    return OFFSET + 10.0 * math.log10(ms) if ms > 0 else -math.inf


def true_peak(audio: ArrayLike) -> float:
    """Return a 4× oversampled peak estimate (information only)."""
    x = np.concatenate((np.asarray(audio, dtype=float), np.zeros(32)))
    return max(float(np.max(np.abs(resample_poly(x, 4, 1)))), float(np.max(np.abs(audio), initial=0.0)))
