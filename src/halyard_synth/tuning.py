# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tuning: MIDI note numbers and names, and the frequencies a note name or a number gives.

Shared by the patch's editing commands, the render report and the WAV root note.
"""

from __future__ import annotations

import math
import re

_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_pitch(hz: float) -> float:
    """Return the fractional MIDI note number (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(midi: int) -> str:
    """Name a MIDI note, such as F#3."""
    return f"{_NOTES[midi % 12]}{midi // 12 - 1}"


def describe_tuning(hz: float) -> str:
    """Describe a frequency as its nearest note and the cents from it, such as 'F#3 -3.2 cents'."""
    p = midi_pitch(hz)
    n = round(p)
    cents = round(100 * (p - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{note_name(n)} {cents:+.1f} cents"


def parse_tune(text: str) -> float:
    """Parse a note name such as D3, C#4 or Eb3 (Hz rounded to 3 decimals), or a frequency in Hz."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text.strip())
    if m:
        midi = _LETTERS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) + 1)
        return round(float(440.0 * 2.0 ** ((midi - 69) / 12)), 3)
    msg = f"not a note name or frequency: {text!r}"
    try:
        hz = float(text)
    except ValueError:
        raise ValueError(msg) from None
    if not math.isfinite(hz):
        raise ValueError(msg)
    return hz
