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
from conftest import edit as set_path

from halyard_synth import patch as P
from halyard_synth.cli import reference_text
from halyard_synth.render import render

RATE = 48000
SNARES = ("tone", "crack", "fat", "driven")


def ref(name):
    return P.loads(reference_text(name))


def edit(p, **fields):
    """Field edits by patch path, such as edit(p, **{"noise[2].level_db": -3.0}); the caller counts them."""
    assert len(fields) <= 3
    d = P.to_dict(p)
    for path, value in fields.items():
        set_path(d, path, value)
    return P.from_dict(d)


def audio(p):
    a, report = render(p)
    assert report.inside and not report.over, str(report)
    return a.astype(float)


def faded(x, start_ms, length_ms):
    """A window from start_ms, flat for its first half and faded out over the second."""
    a, n = int(start_ms / 1000 * RATE), int(length_ms / 1000 * RATE)
    seg = np.zeros(n)
    part = x[a : a + n]
    seg[: part.size] = part
    fade = np.ones(n)
    fade[n // 2 :] = 0.5 * (1 + np.cos(np.pi * np.arange(n - n // 2) / (n - n // 2)))
    return seg * fade


def spectrum(seg):
    power = np.abs(np.fft.rfft(seg, 1 << 16)) ** 2
    return np.fft.rfftfreq(1 << 16, 1 / RATE), power


def centroid(x, start_ms=0.0, length_ms=120.0):
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(f * power) / np.sum(power))


def band_energy(x, lo, hi, start_ms=0.0, length_ms=120.0):
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(power[(f >= lo) & (f < hi)]))


def share(x, lo, hi, start_ms=0.0, length_ms=120.0):
    f, power = spectrum(faded(x, start_ms, length_ms))
    return float(np.sum(power[(f >= lo) & (f < hi)]) / np.sum(power))


def decay_ms(x, drop_db=40):
    """When the 5 ms RMS envelope last sits within drop_db of its peak."""
    w = RATE // 200
    rms = np.sqrt(np.convolve(x**2, np.ones(w) / w, mode="same"))
    return 1000 * np.flatnonzero(rms >= rms.max() * 10 ** (-drop_db / 20))[-1] / RATE


def body_hz(x, p):
    """The body's settled frequency: the spectral peak near the tuning, after the glide."""
    start = p.body.pitch[-1][0] + 5
    length = min(80.0, max(20.0, 0.5 * p.body.amp[-1][0] - start))
    f, power = spectrum(faded(x, start, length) * np.hanning(int(length / 1000 * RATE)))
    near = (f > 0.8 * p.body.tune_hz) & (f < 1.25 * p.body.tune_hz)
    i = np.flatnonzero(near)[np.argmax(power[near])]
    a, b, c = np.log(power[i - 1 : i + 2])
    return float(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))


def crest_db(x, length_ms=60.0):
    seg = x[: int(length_ms / 1000 * RATE)]
    return 20 * math.log10(np.max(np.abs(seg)) / np.sqrt(np.mean(seg**2)))


def loudness(x):
    from halyard_synth.loudness import momentary_max

    return momentary_max(x, RATE)


def layer_index(p, fastest: bool) -> int:
    """The stick (the layer whose shape ends first) or the wires (the longest), 1-based."""
    ends = [layer.amp[-1][0] for layer in p.noise]
    return 1 + (ends.index(min(ends)) if fastest else ends.index(max(ends)))


# From each snare.


@pytest.mark.parametrize("name", SNARES)
def test_change_the_tuning(name):
    p = ref(name)
    up = round(p.body.tune_hz * 2 ** (2 / 12), 3)  # one field: a whole tone up
    before, after = body_hz(audio(p), p), body_hz(audio(edit(p, **{"body.tune_hz": up})), p)
    assert 1200 * math.log2(before / p.body.tune_hz) == pytest.approx(0, abs=5)
    assert 1200 * math.log2(after / before) == pytest.approx(200, abs=5)


@pytest.mark.parametrize("name", SNARES)
def test_shorten_the_tail(name):
    p = ref(name)
    shorter = P.stretch(p, "all", 0.6)  # one command
    assert decay_ms(audio(shorter)) < 0.8 * decay_ms(audio(p))


@pytest.mark.parametrize("name", SNARES)
def test_add_crack(name):
    p = ref(name)
    j = layer_index(p, fastest=True)
    louder = edit(p, **{f"noise[{j}].level_db": p.noise[j - 1].level_db + 6})  # one field: the stick
    # More high end in the first millisecond; in the driven reference less, since there the edge is
    # mostly the saturated wires and body.
    assert band_energy(audio(louder), 4000, 20000, 0, 1) > 1.4 * band_energy(audio(p), 4000, 20000, 0, 1)


@pytest.mark.parametrize("name", SNARES)
def test_add_body(name):
    p = ref(name)
    fuller = edit(p, **{"body.level_db": p.body.level_db + 4})  # one field

    def tone_to_noise(x):
        return band_energy(x, 0, 2.5 * p.body.tune_hz) / band_energy(x, 2000, 20000)

    assert tone_to_noise(audio(fuller)) > 1.5 * tone_to_noise(audio(p))


@pytest.mark.parametrize("name", SNARES)
def test_add_wires(name):
    p = ref(name)
    j = layer_index(p, fastest=False)
    layer = p.noise[j - 1]
    buzzier = edit(
        p, **{f"noise[{j}].level_db": layer.level_db + 3, f"noise[{j}].rattle": min(1.0, layer.rattle + 0.3)}
    )
    # More wire energy in the tail, 20 ms on, above the body's modes.
    assert band_energy(audio(buzzier), 2000, 12000, 20, 150) > 1.3 * band_energy(audio(p), 2000, 12000, 20, 150)


@pytest.mark.parametrize("name", SNARES)
def test_darken(name):
    p = ref(name)
    darker = edit(
        p,
        **{
            f"noise[{j}].band_hz": [layer.band_hz[0], round(0.6 * layer.band_hz[1], 1)]
            for j, layer in enumerate(p.noise, 1)
        },
    )  # one field per layer: at most three
    assert centroid(audio(darker)) < 0.9 * centroid(audio(p))


def sustain_db(x):
    """The body of the hit against its onset: RMS over 20–80 ms relative to the first 10 ms."""
    tail, onset = x[int(0.02 * RATE) : int(0.08 * RATE)], x[: int(0.01 * RATE)]
    return 20 * math.log10(np.sqrt(np.mean(tail**2)) / np.sqrt(np.mean(onset**2)))


@pytest.mark.parametrize("name", SNARES)
def test_add_aggression(name):
    p = ref(name)
    rougher = edit(p, **{"drive.db": p.drive.db + 6, "output.db": p.output.db - 4})  # two fields
    # Denser: the drive lifts the tail against the onset, even where the peaks are already saturated.
    assert sustain_db(audio(rougher)) > sustain_db(audio(p)) + 1.0


@pytest.mark.parametrize("name", SNARES)
def test_make_a_ghost_note(name):
    p = ref(name)
    ghost = P.softer(p, 18)  # one command
    before, after = audio(p), audio(ghost)
    assert loudness(after) < loudness(before) - 8
    assert centroid(after) < centroid(before)  # softer is darker, not just quieter


# From the clap.


def energy_time_ms(x):
    """The centre of gravity of the energy in time."""
    t = np.arange(x.size) / RATE
    return 1000 * float(np.sum(t * x**2) / np.sum(x**2))


def test_tighten_the_clap():
    p = ref("clap")
    tighter = P.stretch(p, "all", 0.7, after_ms=0.0)  # one command: the bursts and the tail close up together
    assert energy_time_ms(audio(tighter)) < 0.85 * energy_time_ms(audio(p))


def test_shorten_the_clap_tail():
    p = ref("clap")
    shorter = P.stretch(p, "all", 0.6)  # one command
    assert decay_ms(audio(shorter)) < 0.8 * decay_ms(audio(p))


def test_darken_the_clap():
    p = ref("clap")
    darker = edit(
        p,
        **{
            f"noise[{j}].band_hz": [round(0.7 * layer.band_hz[0], 1), round(0.6 * layer.band_hz[1], 1)]
            for j, layer in enumerate(p.noise, 1)
        },
    )
    assert centroid(audio(darker)) < 0.85 * centroid(audio(p))


def test_add_aggression_to_the_clap():
    p = ref("clap")
    rougher = edit(p, **{"drive.db": p.drive.db + 8, "output.db": p.output.db - 5})
    assert crest_db(audio(rougher), 40) < crest_db(audio(p), 40) - 1.0


# From the snap.


def test_shorten_the_snap():
    p = ref("snap")
    shorter = P.stretch(p, "all", 0.6)
    assert decay_ms(audio(shorter)) < 0.8 * decay_ms(audio(p))


def test_brighten_the_snap():
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


def test_move_the_ping():
    p = ref("snap")
    moved = edit(p, **{"body.tune_hz": round(p.body.tune_hz * 2 ** (-3 / 12), 3)})  # one field: three semitones down
    before, after = body_hz(audio(p), p), body_hz(audio(moved), moved)
    assert 1200 * math.log2(after / before) == pytest.approx(-300, abs=10)
