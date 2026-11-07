# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The usability tasks of plan §8.4, from each reference patch.

Each task takes at most three field edits or one command, and must move the
sound in the expected direction, judged here by a simple measurement.
Measurements fade their windows out rather than cutting them, so the onset
stays and the cut does not leak. Listening decides the rest; this checks that
the edits are short and predictable.
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from halyard_synth import patch
from halyard_synth.cli import reference_text
from halyard_synth.loudness import momentary_max
from halyard_synth.render import render

from .conftest import edit as set_path

type Floats = NDArray[np.float64]

RATE = 48000
SNARES = ("tone", "crack", "fat", "driven", "machine", "house", "long-snare", "rimshot")
CLAPS = ("clap", "short-clap", "big-clap")
EDITS = 3  # a task takes at most this many field edits
CENTS = 5  # the measured tuning is right within this
PING_CENTS = 10  # and the snap's ping, a shorter sound, within this
LOUDER = 1.4  # a task that adds something adds at least this much energy where it should
GHOST_DB = 8  # a ghost note 18 dB softer reads at least this much quieter
DENSER_DB = 1.0  # more drive lifts the tail against the onset by at least this


def ref(name: str) -> patch.Patch:
    """Return a reference patch."""
    return patch.loads(reference_text(name))


def edit(p: patch.Patch, **fields: object) -> patch.Patch:
    """Apply field edits by patch path, such as **{"noise[2].level_db": -3.0}; at most three."""
    assert len(fields) <= EDITS
    d = patch.to_dict(p)
    for path, value in fields.items():
        set_path(d, path, value)
    return patch.from_dict(d)


def audio(p: patch.Patch) -> Floats:
    """Render a patch, which must be inside the domain and not over."""
    a, report = render(p)
    assert report.inside, str(report)
    assert not report.over, str(report)
    return a.astype(float)


def faded(x: Floats, start_ms: float, length_ms: float) -> Floats:
    """Return a window from start_ms, flat for its first half and faded out over the second."""
    a, n = int(start_ms / 1000 * RATE), int(length_ms / 1000 * RATE)
    seg = np.zeros(n)
    part = x[a : a + n]
    seg[: part.size] = part
    fade = np.ones(n)
    fade[n // 2 :] = 0.5 * (1 + np.cos(np.pi * np.arange(n - n // 2) / (n - n // 2)))
    return seg * fade


def spectrum(seg: Floats) -> tuple[Floats, Floats]:
    """Return a window's frequencies and power."""
    power = np.abs(np.fft.rfft(seg, 1 << 16)) ** 2
    return np.fft.rfftfreq(1 << 16, 1 / RATE), power


def centroid(x: Floats, start_ms: float = 0.0, length_ms: float = 120.0) -> float:
    """Return a window's spectral centroid."""
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(f * power) / np.sum(power))


def band_energy(x: Floats, lo: float, hi: float, start_ms: float = 0.0, length_ms: float = 120.0) -> float:
    """Return a window's energy in a band."""
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(power[(f >= lo) & (f < hi)]))


def decay_ms(x: Floats, drop_db: float = 40) -> float:
    """Return when the 5 ms RMS envelope last sits within drop_db of its peak."""
    w = RATE // 200
    rms = np.sqrt(np.convolve(x**2, np.ones(w) / w, mode="same"))
    return 1000 * float(np.flatnonzero(rms >= rms.max() * 10 ** (-drop_db / 20))[-1]) / RATE


def body_hz(x: Floats, p: patch.Patch) -> float:
    """Return the body's settled frequency: the spectral peak near the tuning, after the glide."""
    start = p.body.pitch[-1][0] + 5
    length = min(80.0, max(20.0, 0.5 * p.body.amp[-1][0] - start))
    f, power = spectrum(faded(x, start, length) * np.hanning(int(length / 1000 * RATE)))
    near = (f > 0.8 * p.body.tune_hz) & (f < 1.25 * p.body.tune_hz)
    i = np.flatnonzero(near)[np.argmax(power[near])]
    a, b, c = np.log(power[i - 1 : i + 2])
    return float(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))


def crest_db(x: Floats, length_ms: float = 60.0) -> float:
    """Return the onset's crest factor, peak over RMS, in dB."""
    seg = x[: int(length_ms / 1000 * RATE)]
    return 20 * math.log10(np.max(np.abs(seg)) / np.sqrt(np.mean(seg**2)))


def loudness(x: Floats) -> float:
    """Return the largest momentary loudness."""
    return momentary_max(x, RATE)


def layer_index(p: patch.Patch, *, fastest: bool) -> int:
    """Return the stick (the layer whose shape ends first) or the wires (the longest), counting from 1."""
    ends = [layer.amp[-1][0] for layer in p.noise]
    return 1 + (ends.index(min(ends)) if fastest else ends.index(max(ends)))


# From each snare.


@pytest.mark.parametrize("name", SNARES)
def test_change_the_tuning(name: str) -> None:
    """Tune each snare up a whole tone with one field, and hear it measured at its tuning, then 200 cents up."""
    p = ref(name)
    up = round(p.body.tune_hz * 2 ** (2 / 12), 3)  # one field: a whole tone up
    before, after = body_hz(audio(p), p), body_hz(audio(edit(p, **{"body.tune_hz": up})), p)
    assert 1200 * math.log2(before / p.body.tune_hz) == pytest.approx(0, abs=CENTS)
    assert 1200 * math.log2(after / before) == pytest.approx(200, abs=CENTS)


@pytest.mark.parametrize("name", SNARES)
def test_shorten_the_tail(name: str) -> None:
    """Shorten each snare's tail with one command."""
    p = ref(name)
    shorter = patch.stretch(p, "all", 0.6)  # one command
    assert decay_ms(audio(shorter)) < 0.8 * decay_ms(audio(p))


@pytest.mark.parametrize("name", SNARES)
def test_add_crack(name: str) -> None:
    """Add crack to each snare with one field: the stick's level, more high end in the first millisecond."""
    p = ref(name)
    j = layer_index(p, fastest=True)
    louder = edit(p, **{f"noise[{j}].level_db": p.noise[j - 1].level_db + 6})  # one field: the stick
    # More high end in the first millisecond; in the driven reference less, since there the edge is
    # mostly the saturated wires and body.
    assert band_energy(audio(louder), 4000, 20000, 0, 1) > LOUDER * band_energy(audio(p), 4000, 20000, 0, 1)


@pytest.mark.parametrize("name", SNARES)
def test_add_body(name: str) -> None:
    """Add body to each snare with one field: more of the body's band against the noise's."""
    p = ref(name)
    fuller = edit(p, **{"body.level_db": p.body.level_db + 4})  # one field

    def tone_to_noise(x: Floats) -> float:
        return band_energy(x, 0, 2.5 * p.body.tune_hz) / band_energy(x, 2000, 20000)

    assert tone_to_noise(audio(fuller)) > 1.5 * tone_to_noise(audio(p))


@pytest.mark.parametrize("name", SNARES)
def test_add_wires(name: str) -> None:
    """Add wires to each snare with two fields: more wire energy in the tail, above the body's modes."""
    p = ref(name)
    j = layer_index(p, fastest=False)
    layer = p.noise[j - 1]
    buzzier = edit(
        p,
        **{f"noise[{j}].level_db": layer.level_db + 3, f"noise[{j}].rattle": min(1.0, layer.rattle + 0.3)},
    )
    # More wire energy in the tail, 20 ms on, above the body's modes.
    assert band_energy(audio(buzzier), 2000, 12000, 20, 150) > 1.3 * band_energy(audio(p), 2000, 12000, 20, 150)


@pytest.mark.parametrize("name", SNARES)
def test_darken(name: str) -> None:
    """Darken each snare with a field per layer: the layers' upper edges lower, the centroid lower."""
    p = ref(name)
    darker = edit(
        p,
        **{
            f"noise[{j}].band_hz": [layer.band_hz[0], round(0.6 * layer.band_hz[1], 1)]
            for j, layer in enumerate(p.noise, 1)
        },
    )  # one field per layer: at most three
    assert centroid(audio(darker)) < 0.9 * centroid(audio(p))


def sustain_db(x: Floats) -> float:
    """Return the body of the hit against its onset: RMS over 20–80 ms relative to the first 10 ms."""
    tail, onset = x[int(0.02 * RATE) : int(0.08 * RATE)], x[: int(0.01 * RATE)]
    return 20 * math.log10(np.sqrt(np.mean(tail**2)) / np.sqrt(np.mean(onset**2)))


@pytest.mark.parametrize("name", SNARES)
def test_add_aggression(name: str) -> None:
    """Add aggression to each snare with two fields: more drive, less output gain, a denser tail."""
    p = ref(name)
    rougher = edit(p, **{"drive.db": p.drive.db + 6, "output.db": p.output.db - 4})  # two fields
    # Denser: the drive lifts the tail against the onset, even where the peaks are already saturated.
    assert sustain_db(audio(rougher)) > sustain_db(audio(p)) + DENSER_DB


@pytest.mark.parametrize("name", SNARES)
def test_make_a_ghost_note(name: str) -> None:
    """Make a ghost note of each snare with one command: quieter, and darker too."""
    p = ref(name)
    ghost = patch.softer(p, 18)  # one command
    before, after = audio(p), audio(ghost)
    assert loudness(after) < loudness(before) - GHOST_DB
    assert centroid(after) < centroid(before)  # softer is darker, not just quieter


# From the clap.


def energy_time_ms(x: Floats) -> float:
    """Return the centre of gravity of the energy in time."""
    t = np.arange(x.size) / RATE
    return 1000 * float(np.sum(t * x**2) / np.sum(x**2))


@pytest.mark.parametrize("name", CLAPS)
def test_tighten_the_clap(name: str) -> None:
    """Tighten the clap with one command: the bursts and the tail close up, the energy earlier."""
    p = ref(name)
    tighter = patch.stretch(p, "all", 0.7, after_ms=0.0)  # one command: the bursts and the tail close up together
    assert energy_time_ms(audio(tighter)) < 0.85 * energy_time_ms(audio(p))


@pytest.mark.parametrize("name", CLAPS)
def test_shorten_the_clap_tail(name: str) -> None:
    """Shorten the clap's tail with one command."""
    p = ref(name)
    shorter = patch.stretch(p, "all", 0.6)  # one command
    assert decay_ms(audio(shorter)) < 0.8 * decay_ms(audio(p))


@pytest.mark.parametrize("name", CLAPS)
def test_darken_the_clap(name: str) -> None:
    """Darken the clap with a field per layer."""
    p = ref(name)
    darker = edit(
        p,
        **{
            f"noise[{j}].band_hz": [round(0.7 * layer.band_hz[0], 1), round(0.6 * layer.band_hz[1], 1)]
            for j, layer in enumerate(p.noise, 1)
        },
    )
    assert centroid(audio(darker)) < 0.85 * centroid(audio(p))


@pytest.mark.parametrize("name", CLAPS)
def test_add_aggression_to_the_clap(name: str) -> None:
    """Add aggression to the clap with two fields: a lower crest factor."""
    p = ref(name)
    rougher = edit(p, **{"drive.db": p.drive.db + 8, "output.db": p.output.db - 5})
    assert crest_db(audio(rougher), 40) < crest_db(audio(p), 40) - DENSER_DB


# From the snap.


def test_shorten_the_snap() -> None:
    """Shorten the snap with one command."""
    p = ref("snap")
    shorter = patch.stretch(p, "all", 0.6)
    assert decay_ms(audio(shorter)) < 0.8 * decay_ms(audio(p))


def test_brighten_the_snap() -> None:
    """Brighten the snap with two fields: its burst's band higher and louder."""
    p = ref("snap")
    band = p.noise[0].band_hz
    brighter = edit(
        p,
        **{
            "noise[1].band_hz": [1.5 * band[0], min(20000.0, 1.3 * band[1])],
            "noise[1].level_db": p.noise[0].level_db + 3,
        },
    )  # two fields
    assert centroid(audio(brighter)) > 1.2 * centroid(audio(p))


def test_move_the_ping() -> None:
    """Move the snap's ping three semitones down with one field."""
    p = ref("snap")
    moved = edit(p, **{"body.tune_hz": round(p.body.tune_hz * 2 ** (-3 / 12), 3)})  # one field: three semitones down
    before, after = body_hz(audio(p), p), body_hz(audio(moved), moved)
    assert 1200 * math.log2(after / before) == pytest.approx(-300, abs=PING_CENTS)


# From the played end and the rim click.


def test_ring_the_rimshot_more() -> None:
    """Make the rimshot's hoop ring more with one field: more of its band after the onset."""
    p = ref("rimshot")
    ringing = edit(p, **{"rim[1].level_db": p.rim[0].level_db + 6})  # one field
    assert band_energy(audio(ringing), 1000, 8000, 3, 40) > LOUDER * band_energy(audio(p), 1000, 8000, 3, 40)


def test_shorten_the_knock() -> None:
    """Shorten the cross-stick and the rim click with one command."""
    for name in ("cross-stick", "rim-click"):
        p = ref(name)
        assert decay_ms(audio(patch.stretch(p, "all", 0.6))) < 0.8 * decay_ms(audio(p))


def test_make_the_cross_stick_woodier() -> None:
    """Make the cross-stick woodier with one field: the rim's damping up, its upper modes gone sooner."""
    p = ref("cross-stick")
    woodier = edit(p, **{"rim[1].damping": p.rim[0].damping + 1})  # one field
    assert centroid(audio(woodier), 5, 40) < 0.9 * centroid(audio(p), 5, 40)


def test_move_the_click() -> None:
    """Move the rim click three semitones down with its two modes' fields."""
    p = ref("rim-click")
    down = 2 ** (-3 / 12)
    moved = edit(p, **{f"rim[1].modes[{i}].hz": round(hz * down, 3) for i, (hz, _) in enumerate(p.rim[0].modes, 1)})

    def loudest_hz(x: Floats) -> float:
        f, power = spectrum(faded(x, 1, 30))
        return float(f[np.argmax(power)])

    assert 1200 * math.log2(loudest_hz(audio(moved)) / loudest_hz(audio(p))) == pytest.approx(-300, abs=PING_CENTS)


def test_widen_the_sweep() -> None:
    """Widen the brush's sweep with one field: its band higher at the top of the sweep."""
    p = ref("brush")
    (top_ms, top), *_ = sorted(p.brush[0].sweep, key=lambda k: -k[1])
    k = 1 + p.brush[0].sweep.index((top_ms, top))
    wider = edit(p, **{f"brush[1].sweep[{k}].ratio": 2 * min(r for _, r in p.brush[0].sweep)})  # one field: an octave
    assert centroid(audio(wider), top_ms - 20, 40) > 1.1 * centroid(audio(p), top_ms - 20, 40)


def test_slow_the_sweep() -> None:
    """Slow the brush's sweep with one command: its brightest moment later."""
    p = ref("brush")
    slower = patch.stretch(p, "brush[1].sweep", 1.5)  # one command

    def brightest_ms(x: Floats) -> float:
        starts = np.arange(0, 300, 10.0)
        return float(starts[np.argmax([centroid(x, s, 40) for s in starts])])

    assert brightest_ms(audio(slower)) > brightest_ms(audio(p)) + 30


# From trance and builds.


def dominant_hz(x: Floats, start_ms: float, length_ms: float) -> float:
    """Return a window's loudest frequency."""
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(f[np.argmax(power)])


def test_shorten_the_zap() -> None:
    """Shorten the zap's fall with one command: lower already 4 ms in, where the written fall is still high."""
    p = ref("zap")
    quicker = patch.stretch(p, "body.pitch", 0.5)  # one command: a fall half as long
    assert dominant_hz(audio(quicker), 3, 3) < 0.8 * dominant_hz(audio(p), 3, 3)


def test_start_the_zap_lower() -> None:
    """Start the zap's fall an octave lower with one field."""
    p = ref("zap")
    lower = edit(p, **{"body.pitch[1].ratio": p.body.pitch[0][1] / 2})  # one field
    assert dominant_hz(audio(lower), 0, 2) < 0.8 * dominant_hz(audio(p), 0, 2)


def test_fit_the_reverse_to_a_shorter_build() -> None:
    """Fit the reverse snare to a shorter build with one command: its swell peaks sooner."""
    p = ref("reverse")
    shorter = patch.stretch(p, "all", 0.5, after_ms=0.0)  # one command

    def peak_ms(x: Floats) -> float:
        return 1000 * float(np.argmax(np.abs(x))) / RATE

    assert peak_ms(audio(shorter)) < 0.6 * peak_ms(audio(p))


def test_close_the_snare_slower() -> None:
    """Close the synth snare's band slower with one command: brighter 100 ms in."""
    p = ref("closing")
    slower = patch.stretch(p, "brush[1].sweep", 2.0)  # one command
    assert centroid(audio(slower), 80, 40) > 1.1 * centroid(audio(p), 80, 40)
