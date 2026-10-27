# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math

import numpy as np
import pytest

from halyard_synth import patch as P
from halyard_synth.curve import amplitude_curve, pitch_curve
from halyard_synth.patch import PatchError, dumps, from_dict, loads

from .conftest import EXAMPLE, make


def problems(fn, *args, **kwargs):
    with pytest.raises(PatchError) as e:
        fn(*args, **kwargs)
    return e.value.problems


def reject(**edits):
    return problems(make, **edits)


def amp(*points):
    return [{"ms": m, "gain": g} for m, g in points]


def pitch(*points):
    return [{"ms": m, "ratio": r} for m, r in points]


def modes(*pairs):
    return [{"ratio": r, "level_db": v} for r, v in pairs]


# Parsing.


def test_example_parses():
    p = loads(EXAMPLE)
    assert p.seed == 1
    assert p.body.tune_hz == 185.0 and p.body.damping == 1.0
    assert p.body.modes == ((1.0, 0.0), (1.593, -5.0), (2.136, -11.0))
    assert p.body.pitch == ((0.0, 1.08), (15.0, 1.0))
    assert len(p.noise) == 2
    assert p.noise[1] == P.Layer(
        -6.0, (2500.0, 9000.0), 0.5, ((0.0, 0.0), (1.5, 1.0), (40.0, 0.4), (140.0, 0.08), (260.0, 0.0))
    )
    assert p.drive == P.Drive(6.0)
    assert p.output == P.Output(48000, "float32", "gain", -6.0, 1)
    assert p.length_ms == 260.0


def random_patch(rng) -> P.Patch:
    def times(n, first, limit):
        return [first, *(first + np.cumsum(rng.uniform(0.01, limit / n, n - 1)))]

    def shape(n, first, limit):
        g = list(rng.uniform(0, 1, n))
        g[0] = g[-1] = 0.0
        g[int(rng.integers(1, n - 1))] = 1.0
        return tuple(zip(times(n, first, limit), g))

    tune = float(rng.uniform(40, 4000))
    count = int(rng.integers(1, 5))
    top = min(8.0, 20000 / tune / 1.01)
    ratios = [1.0, *np.sort(rng.uniform(1.0001, max(top, 1.0002), count - 1))]
    if len(set(ratios)) < count:
        ratios, count = [1.0], 1
    levels = list(rng.uniform(-60, 0, count))
    levels[int(rng.integers(count))] = 0.0
    n = int(rng.integers(2, 33))
    r = list(rng.uniform(10 / tune, 20000 / (tune * ratios[-1]), n))
    r[-1] = 1.0
    layers = []
    for _ in range(int(rng.integers(1, 5))):
        lo = float(rng.uniform(20, 19000))
        layers.append(
            P.Layer(
                float(rng.choice([-math.inf, rng.uniform(-90, 24)])),
                (lo, float(rng.uniform(lo + 1, 20000))),
                float(rng.uniform(0, 1)),
                shape(int(rng.integers(3, 33)), float(rng.uniform(0, 5)), 990),
            )
        )
    mode = str(rng.choice(["gain", "peak"]))
    return P.Patch(
        seed=int(rng.integers(0, 2**63 - 1)),
        body=P.Body(
            tune,
            float(rng.choice([-math.inf, rng.uniform(-90, 24)])),
            float(rng.uniform(0, 3)),
            tuple(zip(ratios, levels)),
            tuple(zip(times(n, 0.0, 2000), r)),
            shape(int(rng.integers(3, 33)), float(rng.uniform(0, 5)), 1990),
        ),
        noise=tuple(layers),
        drive=P.Drive(float(rng.uniform(0, 48))),
        output=P.Output(
            int(rng.choice(P.SAMPLE_RATES)),
            str(rng.choice(P.FORMATS)),
            mode,
            float(rng.uniform(-60, 0 if mode == "peak" else 24)),
            int(rng.choice([1, -1])),
        ),
    )


def test_dump_round_trips_exactly():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = random_patch(rng)
        text = dumps(p)
        assert loads(text) == p
        assert dumps(loads(text)) == text


def test_dump_of_example_keeps_the_plan_layout():
    ours, plan = dumps(loads(EXAMPLE)).splitlines(), EXAMPLE.splitlines()
    assert len(ours) == len(plan)
    for a, b in zip(ours, plan):
        # The plan annotates its layers; the canonical comments only number them.
        assert a.split("#")[0].rstrip() == b.split("#")[0].rstrip()
    assert ours[:25] == plan[:25]


def test_instrument_comes_first(example_dict):
    d = {k: v for k, v in example_dict.items() if k != "instrument"}
    assert problems(from_dict, d) == ['instrument: missing; a snare patch starts with instrument = "snare"']
    d = dict(example_dict, instrument="kick", colour="red")
    assert problems(from_dict, d) == ["instrument: this is a 'kick' patch, not a \"snare\" one"]


def test_schema_versions(example_dict):
    for bad, message in ((2, "unsupported version 2"), ("1", "expected an integer"), (True, "expected an integer")):
        assert message in problems(from_dict, dict(example_dict, schema=bad))[0]
    d = {k: v for k, v in example_dict.items() if k != "schema"}
    assert problems(from_dict, d) == ["schema: missing"]
    # A future schema is reported alone, not as a list of field errors.
    assert len(problems(from_dict, dict(example_dict, schema=2, colour="red"))) == 1


def test_integers_accepted_for_floats_but_not_booleans():
    p = make(**{"body.tune_hz": 185, "drive.db": 0, "noise[1].band_hz": [1500, 16000], "body.modes[2].ratio": 2})
    assert type(p.body.tune_hz) is float and type(p.drive.db) is float
    assert all(type(x) is float for x in p.noise[0].band_hz)
    assert type(p.body.modes[1][0]) is float
    assert "body.damping: expected a number, found a boolean" in reject(**{"body.damping": False})
    assert any("output.sample_rate: expected an integer" in m for m in reject(**{"output.sample_rate": 48000.0}))
    assert any("output.polarity: expected an integer" in m for m in reject(**{"output.polarity": True}))
    assert any("seed: expected an integer" in m for m in reject(seed=1.0))
    assert any("noise[2].rattle: expected a number, found a string" in m for m in reject(**{"noise[2].rattle": "0.5"}))


def test_unknown_and_missing_fields(example_dict):
    d = example_dict
    d["colour"] = "red"
    d["body"]["decay"] = 1.0
    del d["noise"][1]["rattle"]
    d["body"]["pitch"][1]["curve"] = "exp"
    d["body"]["modes"][0]["q"] = 3
    del d["drive"]
    found = problems(from_dict, d)
    assert "colour: unknown field" in found
    assert "body.decay: unknown field" in found
    assert "noise[2].rattle: missing" in found
    assert "body.pitch[2].curve: unknown field" in found
    assert "body.modes[1].q: unknown field" in found
    assert "drive: missing" in found


def test_every_field_is_required(example_dict):
    for section in ("body", "drive", "output"):
        for key in list(example_dict[section]):
            d = dict(example_dict)
            d[section] = {k: v for k, v in example_dict[section].items() if k != key}
            assert f"{section}.{key}: missing" in problems(from_dict, d)
    for key in list(example_dict["noise"][0]):
        d = dict(example_dict)
        d["noise"] = [{k: v for k, v in example_dict["noise"][0].items() if k != key}, example_dict["noise"][1]]
        assert f"noise[1].{key}: missing" in problems(from_dict, d)
    for key in ("seed", "body", "noise", "drive", "output"):
        d = {k: v for k, v in example_dict.items() if k != key}
        assert f"{key}: missing" in problems(from_dict, d)


def test_non_finite_numbers():
    assert make(**{"body.level_db": -math.inf}).body.level_db == -math.inf
    assert make(**{"noise[2].level_db": -math.inf}).noise[1].level_db == -math.inf
    for path in ("body.level_db", "noise[1].level_db"):
        assert any("must be finite" in m for m in reject(**{path: math.inf}))
        assert any("must be finite" in m for m in reject(**{path: math.nan}))
    for path in ("body.tune_hz", "body.damping", "drive.db", "output.db", "noise[1].rattle", "body.modes[2].level_db"):
        assert any("must be finite" in m for m in reject(**{path: -math.inf}))
        assert any("must be finite" in m for m in reject(**{path: math.nan}))
    assert loads(EXAMPLE.replace("level_db = -4.0", "level_db = -inf")).body.level_db == -math.inf


def test_knot_times():
    assert any(
        "repeats the previous knot time" in m for m in reject(**{"body.amp": amp((0, 0), (1, 1), (1, 0.5), (2, 0))})
    )
    assert any(
        "is earlier than the previous knot" in m for m in reject(**{"body.amp": amp((0, 0), (2, 1), (1, 0.5), (3, 0))})
    )
    found = reject(**{"noise[1].amp": amp((0, 0), (0.005, 1), (1, 0))})
    assert any(m.startswith("noise[1].amp[2].ms: is 0.005 ms after the previous knot") for m in found)
    # Exactly 10 µs is allowed, even where binary rounding makes the float gap smaller.
    assert 0.03 - 0.02 < 0.01
    make(**{"noise[1].amp": amp((0, 0), (0.02, 0.5), (0.03, 1), (1, 0))})
    assert any("body.amp[1].ms" in m for m in reject(**{"body.amp": amp((-1, 0), (1, 1), (2, 0))}))


def test_amplitude_shapes():
    assert "body.amp[1].gain: an amplitude shape must start at 0" in reject(
        **{"body.amp": amp((0, 0.1), (1, 1), (2, 0))}
    )
    assert "body.amp[3].gain: an amplitude shape must end at 0" in reject(**{"body.amp": amp((0, 0), (1, 1), (2, 0.1))})
    found = reject(**{"noise[2].amp": amp((0, 0), (1, 0.9999999), (2, 0))})
    assert "noise[2].amp: the highest gain must be exactly 1, found 0.9999999" in found
    assert "body.amp[2].gain: 1.5 is outside 0 … 1" in reject(**{"body.amp": amp((0, 0), (1, 1.5), (2, 1), (3, 0))})
    assert "body.amp[3].gain: -0.1 is outside 0 … 1" in reject(**{"body.amp": amp((0, 0), (1, 1), (2, -0.1), (3, 0))})
    # Delayed shapes are allowed: the first knot need not be at 0 ms.
    assert make(**{"noise[1].amp": amp((5, 0), (6, 1), (20, 0))}).noise[0].amp[0] == (5.0, 0.0)


def test_pitch_curves():
    assert "body.pitch[1].ms: the pitch curve must start at 0 ms" in reject(**{"body.pitch": pitch((1, 1.1), (15, 1))})
    assert "body.pitch[2].ratio: the pitch curve must end at ratio 1" in reject(
        **{"body.pitch": pitch((0, 1.1), (15, 1.01))}
    )
    assert "body.pitch[1].ratio: must be positive" in reject(**{"body.pitch": pitch((0, -1), (15, 1))})
    # Dips below the tuning are fine while every frequency stays in range.
    make(**{"body.pitch": pitch((0, 1.2), (5, 0.9), (15, 1))})


def test_every_mode_stays_in_the_audio_band_at_every_pitch_knot():
    # At 4 kHz, a ratio of 5 is exactly 20 kHz: allowed, but no glide above it.
    make(**{"body.tune_hz": 4000.0, "body.modes": modes((1, 0), (5, -6)), "body.pitch": pitch((0, 1), (15, 1))})
    found = reject(
        **{"body.tune_hz": 4000.0, "body.modes": modes((1, 0), (5, -6)), "body.pitch": pitch((0, 1.001), (15, 1))}
    )
    assert found == ["body.pitch[1].ratio: body frequency 20020 Hz for mode 2 is outside 10 Hz … 20 kHz"]
    found = reject(
        **{
            "body.tune_hz": 40.0,
            "body.modes": modes(
                (1, 0),
            ),
            "body.pitch": pitch((0, 0.2), (15, 1)),
        }
    )
    assert found == ["body.pitch[1].ratio: body frequency 8 Hz is outside 10 Hz … 20 kHz"]


def test_modes():
    make(**{"body.modes": modes((1, 0))})
    make(**{"body.modes": modes((1, -3), (1.5, 0), (2, -60), (8, -10))})
    assert "body.modes: needs 1 to 4 modes, found 0" in reject(**{"body.modes": []})
    assert "body.modes: needs 1 to 4 modes, found 5" in reject(
        **{"body.modes": modes((1, 0), (2, 0), (3, 0), (4, 0), (5, 0))}
    )
    assert any(
        m.startswith("body.modes[1].ratio: the first mode defines the tuning")
        for m in reject(**{"body.modes": modes((1.01, 0), (2, -3))})
    )
    assert "body.modes[3].ratio: mode ratios must strictly increase" in reject(
        **{"body.modes": modes((1, 0), (2, -3), (2, -6))}
    )
    assert "body.modes[2].ratio: mode ratios must strictly increase" in reject(**{"body.modes": modes((1, 0), (1, -3))})
    assert any(
        m.startswith("body.modes[2].ratio: 8.01 is outside")
        for m in reject(**{"body.modes": modes((1, 0), (8.01, -3))})
    )
    assert "body.modes: the loudest mode must be exactly 0 dB, found -1.0" in reject(
        **{"body.modes": modes((1, -1), (2, -3))}
    )
    assert any(
        m.startswith("body.modes[2].level_db: -61 is outside")
        for m in reject(**{"body.modes": modes((1, 0), (2, -61))})
    )
    assert any(
        m.startswith("body.modes[1].level_db: 1 is outside") for m in reject(**{"body.modes": modes((1, 1), (2, 0))})
    )


def test_noise_layers(example_dict):
    one = dict(example_dict, noise=example_dict["noise"][:1])
    assert len(from_dict(one).noise) == 1
    four = dict(example_dict, noise=example_dict["noise"] * 2)
    assert len(from_dict(four).noise) == 4
    assert "noise: needs 1 to 4 layers, found 5" in problems(
        from_dict, dict(example_dict, noise=example_dict["noise"] * 2 + example_dict["noise"][:1])
    )
    assert "noise: needs 1 to 4 layers, found 0" in problems(from_dict, dict(example_dict, noise=[]))
    assert any(
        m.startswith("noise: expected an array of tables ([[noise]])")
        for m in problems(from_dict, dict(example_dict, noise=example_dict["noise"][0]))
    )


def test_noise_bands():
    make(**{"noise[1].band_hz": [20, 20000]})
    for bad, path in (
        ([19, 1000], "noise[1].band_hz[1]"),
        ([100, 20001], "noise[1].band_hz[2]"),
        ([500, 500], "noise[1].band_hz"),
        ([900, 800], "noise[1].band_hz"),
    ):
        assert any(m.startswith(path + ":") for m in reject(**{"noise[1].band_hz": bad}))
    assert "noise[2].band_hz: expected [low, high] in Hz" in reject(**{"noise[2].band_hz": [100.0]})


def test_knot_counts_and_lengths():
    assert any("needs 2 to 32 knots, found 1" in m for m in reject(**{"body.pitch": pitch((0, 1))}))
    many = pitch(*[(float(i), 1.1) for i in range(32)], (40.0, 1.0))
    assert any("needs 2 to 32 knots, found 33" in m for m in reject(**{"body.pitch": many}))
    assert any("needs 3 to 32 knots, found 2" in m for m in reject(**{"noise[1].amp": amp((0, 1), (1, 0))}))
    make(**{"body.amp": amp((0, 0), (1, 1), (2000, 0)), "body.pitch": pitch((0, 1.1), (2000, 1))})
    assert any("body.amp[3].ms" in m for m in reject(**{"body.amp": amp((0, 0), (1, 1), (2000.001, 0))}))
    assert any("body.pitch[2].ms" in m for m in reject(**{"body.pitch": pitch((0, 1.1), (2001, 1))}))
    make(**{"noise[2].amp": amp((0, 0), (1, 1), (1000, 0))})
    assert any("noise[2].amp[3].ms" in m for m in reject(**{"noise[2].amp": amp((0, 0), (1, 1), (1000.5, 0))}))


@pytest.mark.parametrize(
    "path, good, bad",
    [
        ("body.tune_hz", [40, 4000], [39.999, 4000.001]),
        ("body.level_db", [-90, 24], [-90.1, 24.1]),
        ("body.damping", [0, 3], [-0.001, 3.001]),
        ("noise[1].level_db", [-90, 24], [-91, 25]),
        ("noise[2].rattle", [0, 1], [-0.01, 1.01]),
        ("drive.db", [0, 48], [-0.1, 48.1]),
        ("output.db", [-60, 24], [-60.5, 24.5]),
        ("output.sample_rate", [44100, 48000, 88200, 96000], [22050, 192000]),
        ("output.format", ["float32", "pcm24", "pcm16"], ["pcm32", "float64", 24]),
        ("output.mode", ["gain", "peak"], ["loudness"]),
        ("output.polarity", [1, -1], [0, 2]),
        ("seed", [0, 2**63 - 1], [-1, 2**63]),
    ],
)
def test_domain_edges(path, good, bad):
    for v in good:
        make(**{path: v, "output.db": min(0.0, -6.0)} if path == "output.mode" else {path: v})
    for v in bad:
        assert any(m.startswith(path) for m in reject(**{path: v}))


def test_peak_target_is_at_most_full_scale():
    make(**{"output.mode": "peak", "output.db": 0.0})
    make(**{"output.mode": "peak", "output.db": -60.0})
    assert any(m.startswith("output.db") for m in reject(**{"output.mode": "peak", "output.db": 0.1}))


def test_all_problems_are_reported_together():
    found = reject(
        **{
            "body.damping": 4.0,
            "drive.db": 60.0,
            "output.mode": "loud",
            "noise[2].rattle": 2.0,
            "body.amp": amp((0, 0), (1, 0.5), (2, 0)),
        }
    )
    assert len(found) == 5


def test_toml_syntax_errors_are_patch_errors():
    assert problems(loads, 'instrument = "snare"\nschema = 1\n[body\n')[0].startswith("TOML:")


def test_programmatic_edits_are_validated():
    p = loads(EXAMPLE)
    with pytest.raises(PatchError):
        dataclasses.replace(p, body=dataclasses.replace(p.body, damping=3.5))
    with pytest.raises(PatchError):
        dataclasses.replace(p, seed=-5)
    with pytest.raises(PatchError):
        dataclasses.replace(p, noise=p.noise * 3)
    assert dataclasses.replace(p, body=dataclasses.replace(p.body, tune_hz=200.0)).body.tune_hz == 200.0


def test_source_length_counts_only_audible_sources():
    assert make().length_ms == 260.0
    assert make(**{"noise[2].level_db": -math.inf}).length_ms == 180.0  # the body, then the stick
    assert make(**{"noise[2].level_db": -math.inf, "body.level_db": -math.inf}).length_ms == 8.0
    assert make(**{"body.amp": amp((0, 0), (1, 1), (900, 0)), "body.level_db": -math.inf}).length_ms == 260.0
    silent = {"body.level_db": -math.inf, "noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}
    assert make(**silent).length_ms == 0.0


def test_parse_tune_and_describe():
    assert P.parse_tune("D3") == 146.832
    assert P.parse_tune("A4") == 440.0
    assert P.parse_tune("C#4") == P.parse_tune("Db4")
    assert P.parse_tune("185.5") == 185.5
    for bad in ("H2", "nan", "inf", ""):
        with pytest.raises(ValueError):
            P.parse_tune(bad)
    assert P.describe_tuning(185.0) == "F#3 +0.0 cents"
    assert P.describe_tuning(190.0) == "F#3 +46.2 cents"
    assert P.describe_tuning(440.0) == "A4 +0.0 cents"
    assert P.note_name(50) == "D3"


# Editing commands.


def test_tune_sets_the_fundamental_and_keeps_the_ratios():
    p = loads(EXAMPLE)
    q = P.tune(p, "D3")
    assert q.body.tune_hz == 146.832
    assert dataclasses.replace(q, body=dataclasses.replace(q.body, tune_hz=185.0)) == p
    assert P.tune(p, "220").body.tune_hz == 220.0
    # A mode or the glide leaving the audio band fails; nothing is clamped.
    high = make(**{"body.modes": modes((1, 0), (7, -6))})
    assert P.tune(high, "2500").body.tune_hz == 2500.0
    found = problems(P.tune, high, "3000")
    assert found == [
        "body.pitch[1].ratio: body frequency 22680 Hz for mode 2 is outside 10 Hz … 20 kHz",
        "body.pitch[2].ratio: body frequency 21000 Hz for mode 2 is outside 10 Hz … 20 kHz",
    ]
    assert any(m.startswith("body.tune_hz") for m in problems(P.tune, p, "C9"))
    with pytest.raises(ValueError):
        P.tune(p, "X3")


def test_stretch_rescales_after_each_peak():
    p = loads(EXAMPLE)
    q = P.stretch(p, "all", 0.8)
    assert q.body.amp == ((0.0, 0.0), (0.4, 1.0), (20.08, 0.45), (72.08, 0.1), (144.08, 0.0))
    assert q.noise[0].amp == ((0.0, 0.0), (0.05, 1.0), (0.13, 0.7), (2.41, 0.1), (6.41, 0.0))
    assert q.noise[1].amp == ((0.0, 0.0), (1.5, 1.0), (32.3, 0.4), (112.3, 0.08), (208.3, 0.0))
    assert q.body.pitch == p.body.pitch  # `all` names the amplitude shapes
    assert P.stretch(p, ["body.pitch"], 2.0).body.pitch == ((0.0, 1.08), (30.0, 1.0))
    r = P.stretch(p, ["noise[2].amp", "body.amp"], 1.5, after_ms=10.0)
    assert r.noise[1].amp == ((0.0, 0.0), (1.5, 1.0), (55.0, 0.4), (205.0, 0.08), (385.0, 0.0))
    assert r.body.amp[:2] == p.body.amp[:2] and r.body.amp[2] == (32.5, 0.45)
    assert r.noise[0] == p.noise[0]
    # Times are written to 1 µs.
    assert P.stretch(p, "noise[1].amp", 1 / 3).noise[0].amp == (
        (0.0, 0.0),
        (0.05, 1.0),
        (0.083, 0.7),
        (1.033, 0.1),
        (2.7, 0.0),
    )


def test_stretch_failures():
    p = loads(EXAMPLE)
    assert problems(P.stretch, p, "noise[3].amp", 2.0)[0].startswith("noise[3].amp: not a curve of this patch")
    assert problems(P.stretch, p, "attack.amp", 2.0)[0].startswith("attack.amp: not a curve of this patch")
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(PatchError):
            P.stretch(p, "all", bad)
    with pytest.raises(PatchError):
        P.stretch(p, "all", 2.0, after_ms=-1.0)
    # Stretching past a curve's span limit fails rather than clamping.
    assert any(m.startswith("noise[2].amp[5].ms") for m in problems(P.stretch, p, "noise[2].amp", 4.0))


def test_bursts_write_a_clap_train():
    p = loads(EXAMPLE)
    q = P.bursts(p, "noise[1]", P.Train(4, 11.0))
    knots = q.noise[0].amp
    assert len(knots) == 3 * 4 + 2
    assert knots[0] == (0.0, 0.0)
    peaks = [knots[1 + 3 * i] for i in range(4)]
    assert peaks == [(0.1, 1.0), (11.1, 1.0), (22.1, 1.0), (33.1, 1.0)]
    shoulders = [knots[2 + 3 * i] for i in range(4)]
    assert [round(ms - pk[0], 6) for (ms, _), pk in zip(shoulders, peaks)] == [0.1] * 4
    dips = [knots[3 + 3 * i] for i in range(3)]
    assert dips == [(11.0, 0.1), (22.0, 0.1), (33.0, 0.1)]
    assert knots[-2] == (59.767, 0.1) and knots[-1] == (113.1, 0.0)  # 20 dB down a third of the way
    assert q.noise[1] == p.noise[1] and q.body == p.body
    # The shoulder lies on the exponential that the burst's decay follows.
    assert shoulders[0][1] == pytest.approx(math.exp(-math.log(10) / 10.9 * 0.1), abs=5e-5)
    assert shoulders[-1][1] == pytest.approx(10 ** (-3 * 0.1 / 80), abs=5e-5)


def test_burst_options():
    p = loads(EXAMPLE)
    q = P.bursts(p, "noise[2]", P.Train(3, 10.0, start=2.0, growth=1.5, rise=0.2, dip=0.25, taper=0.5, tail=50.0))
    knots = q.noise[1].amp
    assert len(knots) == 3 * 3 + 3
    assert knots[:2] == ((0.0, 0.0), (2.0, 0.0))
    assert [knots[2 + 3 * i] for i in range(3)] == [(2.2, 1.0), (12.2, 0.5), (27.2, 0.25)]  # gaps 10, then 15
    assert [knots[4 + 3 * i] for i in range(2)] == [(12.0, 0.125), (27.0, 0.0625)]  # dips under the lower peak
    assert knots[-1] == (77.2, 0.0)
    rising = P.bursts(p, "noise[2]", P.Train(3, 10.0, taper=2.0)).noise[1].amp
    assert [rising[1 + 3 * i][1] for i in range(3)] == [0.25, 0.5, 1.0]
    single = P.bursts(p, "noise[1]", P.Train(1, 10.0)).noise[0].amp
    assert len(single) == 5 and single[1] == (0.1, 1.0)
    silent_dips = P.bursts(p, "noise[1]", P.Train(2, 10.0, dip=0.0)).noise[0].amp
    assert silent_dips[3] == (10.0, 0.0)


def test_burst_shapes_are_ordinary_valid_shapes():
    p = loads(EXAMPLE)
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 11))
        kwargs = dict(
            start=float(rng.choice([0.0, rng.uniform(0.01, 20)])),
            growth=float(rng.uniform(0.8, 1.3)),
            rise=float(rng.uniform(0.01, 1.0)),
            dip=float(rng.uniform(0, 0.9)),
            taper=float(rng.uniform(0.5, 2)),
            tail=float(rng.uniform(1, 300)),
        )
        try:
            q = P.bursts(p, "noise[1]", P.Train(n, float(rng.uniform(1, 30)), **kwargs))
        except PatchError as e:
            # The only acceptable failures: too many knots, a rise that does not fit, or too long a train.
            assert all("knots" in m or "--rise" in m or ".ms:" in m for m in e.problems), e.problems
            continue
        knots = q.noise[0].amp
        assert len(knots) == 3 * n + 2 + (kwargs["start"] > 0)
        gains = [g for _, g in knots]
        assert max(gains) == 1.0 and gains[0] == gains[-1] == 0.0
        # Each burst's rise is one segment between zero-slope knots.
        curve = amplitude_curve([ms / 1000 for ms, _ in knots], gains)
        assert [b - a for a, b in P.rises(gains)] == [1] * n
        for a, b in P.rises(gains):
            assert curve.d[a] == 0.0 and curve.d[b] == 0.0


def test_burst_failures():
    p = loads(EXAMPLE)
    assert problems(P.bursts, p, "noise[3]", P.Train(4, 10.0))[0].startswith("noise[3]: not a layer of this patch")
    assert problems(P.bursts, p, "noise[1]", P.Train(11, 10.0)) == ["--count: needs 1 to 10 bursts, found 11"]
    assert problems(P.bursts, p, "noise[1]", P.Train(0, 10.0)) == ["--count: needs 1 to 10 bursts, found 0"]
    for option, value in (
        ("rise", 0.005),
        ("dip", 1.0),
        ("dip", -0.1),
        ("taper", 0.0),
        ("growth", 0.0),
        ("start", -1.0),
        ("tail", 0.3),
        ("rise", math.nan),
    ):
        assert problems(P.bursts, p, "noise[1]", P.Train(4, 10.0, **{option: value}))[0].startswith(f"--{option}:")
    assert problems(P.bursts, p, "noise[1]", P.Train(4, 0.0))[0].startswith("--spacing:")
    assert problems(P.bursts, p, "noise[1]", P.Train(4, 1.0, rise=0.9))[0].startswith("--rise: burst 1 rises for 0.9 ms")
    assert problems(P.bursts, p, "noise[1]", P.Train(10, 5.0, start=1.0)) == [
        "noise[1].amp: 10 bursts need 33 knots, more than the 32 allowed"
    ]
    make_long = problems(P.bursts, p, "noise[1]", P.Train(10, 100.0, tail=200.0))
    assert any(m.startswith("noise[1].amp[") and "outside 0 … 1000" in m for m in make_long)


def test_rises():
    assert P.rises([0, 1, 0.45, 0.1, 0]) == [(0, 1)]
    assert P.rises([0, 0.5, 1, 0.2, 0.2, 0.7, 0]) == [(0, 2), (4, 5)]
    assert P.rises([0, 0.5, 0.5, 1, 0]) == [(0, 1), (2, 3)]
    assert P.rises([0, 0, 1, 0]) == [(1, 2)]


def test_softer_follows_the_plan_figures():
    p = loads(EXAMPLE)
    q = P.softer(p, 12)
    assert q.body.level_db == -16.0 and [n.level_db for n in q.noise] == [-22.0, -18.0]
    assert q.body.pitch == ((0.0, 1.005048), (15.0, 1.0))  # a 1.08 glide becomes 1.005
    assert q.noise[0].amp[1] == (0.066, 1.0)  # a 0.05 ms rise becomes 0.066 ms
    assert q.noise[0].band_hz == (1500.0, 12137.2)  # a 16 kHz upper edge becomes 12.1 kHz
    # Later knots shift by the added time, so every decay keeps its duration.
    assert q.noise[0].amp == ((0.0, 0.0), (0.066, 1.0), (0.166, 0.7), (3.016, 0.1), (8.016, 0.0))
    assert q.body.amp == ((0.0, 0.0), (0.527, 1.0), (25.127, 0.45), (90.127, 0.1), (180.127, 0.0))
    # Nothing else changes.
    assert (q.body.tune_hz, q.body.damping, q.body.modes) == (p.body.tune_hz, p.body.damping, p.body.modes)
    assert [n.rattle for n in q.noise] == [n.rattle for n in p.noise]
    assert [n.band_hz[0] for n in q.noise] == [n.band_hz[0] for n in p.noise]
    assert (q.drive, q.output, q.seed) == (p.drive, p.output, p.seed)


def test_softer_by_zero_changes_nothing_and_exponents_turn_rules_off():
    p = loads(EXAMPLE)
    assert P.softer(p, 0) is p
    q = P.softer(p, 12, glide=0, contact=0)
    assert q.body.pitch == p.body.pitch and q.body.amp == p.body.amp
    assert [(n.amp, n.band_hz) for n in q.noise] == [(n.amp, n.band_hz) for n in p.noise]
    assert q.body.level_db == -16.0
    silent = make(**{"noise[1].level_db": -math.inf})
    assert P.softer(silent, 6).noise[0].level_db == -math.inf


def close(p, q):
    """Equal up to the decimal rounding softer writes.

    One edit keeps every knot within 0.5 µs of its exact place. A second edit
    lengthens the rises as written, each up to 1 µs off, so a knot may move by
    up to 1 µs more for every rise before it.
    """
    assert p.body.level_db == pytest.approx(q.body.level_db, abs=2e-6)
    for (m1, r1), (m2, r2) in zip(p.body.pitch, q.body.pitch, strict=True):
        assert m1 == m2 and r1 == pytest.approx(r2, abs=2e-6)
    for a, b in zip((p.body, *p.noise), (q.body, *q.noise), strict=True):
        ends = [j for _, j in P.rises([g for _, g in b.amp])]
        for k, ((m1, g1), (m2, g2)) in enumerate(zip(a.amp, b.amp, strict=True)):
            before = sum(j < k for j in ends)
            assert m1 == pytest.approx(m2, abs=1e-3 * (1.001 + before)) and g1 == g2
    for a, b in zip(p.noise, q.noise, strict=True):
        assert a.level_db == pytest.approx(b.level_db, abs=2e-6)
        assert a.band_hz[0] == b.band_hz[0] and a.band_hz[1] == pytest.approx(b.band_hz[1], abs=0.2)


def test_softer_composes():
    p = loads(EXAMPLE)
    clap = P.bursts(p, "noise[1]", P.Train(4, 11.0))
    for base in (p, clap):
        for a, b in ((6, 6), (3, 9), (12, 12), (0.5, 20)):
            close(P.softer(P.softer(base, a), b), P.softer(base, a + b))


def test_softer_rules_are_exact_on_the_whole_curve():
    # Each segment of a rewritten shape is the original segment read at a linearly mapped time,
    # and the glide's excess scales on the whole curve, not only at its knots.
    p = P.bursts(loads(EXAMPLE), "noise[1]", P.Train(4, 11.0))
    for db in (6, 12, 24, 36):
        q = P.softer(p, db)
        for old, new in zip((p.body.amp, *[n.amp for n in p.noise]), (q.body.amp, *[n.amp for n in q.noise])):
            t0, g = np.array([ms for ms, _ in old]) / 1000, [g for _, g in old]
            t1 = np.array([ms for ms, _ in new]) / 1000
            e0, e1 = amplitude_curve(t0, g), amplitude_curve(t1, g)
            u = np.linspace(0, 1, 101)
            for i in range(len(t0) - 1):
                np.testing.assert_allclose(
                    e1(t1[i] + u * (t1[i + 1] - t1[i])), e0(t0[i] + u * (t0[i + 1] - t0[i])), rtol=0, atol=1e-12
                )
        a = (10 ** (-db / 20)) ** 2
        t = np.array([ms for ms, _ in p.body.pitch]) / 1000
        r = np.array([ratio for _, ratio in p.body.pitch])
        x = np.linspace(-0.001, t[-1] + 0.001, 997)
        np.testing.assert_allclose(
            pitch_curve(t, 1 + a * (r - 1))(x), 1 + a * (pitch_curve(t, r)(x) - 1), rtol=0, atol=1e-15
        )


def test_softer_fails_rather_than_clamping():
    p = loads(EXAMPLE)
    found = problems(P.softer, p, 90)
    assert "body.level_db: -94 is outside -90 … 24 (or -inf for silence)" in found
    assert "noise[2].band_hz: the low edge 2500 Hz must be below the high edge 1133 Hz" in found
    for bad in (math.nan, math.inf):
        with pytest.raises(PatchError):
            P.softer(p, bad)
    # A harder hit is the inverse edit, and fails where it leaves the domain.
    back = P.softer(P.softer(p, -3), 3).noise[0].amp
    np.testing.assert_allclose(np.array(back), np.array(p.noise[0].amp), rtol=0, atol=1e-3)
    assert any(m.startswith("noise[1].band_hz[2]") for m in problems(P.softer, p, -12))


@pytest.mark.parametrize("name", ["tone", "crack", "fat", "driven", "clap", "snap"])
def test_softer_ladders_on_every_reference(name):
    # Plan §8.1: levels fall by exactly DB, the glide's excess scales by v², rises by v^(−1/5) and upper
    # band edges by v^(1/5), up to the decimals softer writes; and the ladder composes.
    from halyard_synth import cli

    p = loads(cli.reference_text(name))
    for db in (0, 6, 12, 18, 24, 30, 36):
        q = P.softer(p, db)
        v = 10 ** (-db / 20)
        for a, b in zip((p.body, *p.noise), (q.body, *q.noise)):
            assert b.level_db == (a.level_db if a.level_db == -math.inf else pytest.approx(a.level_db - db, abs=1e-9))
            for (a0, _), (a1, _) in zip(P.rises([g for _, g in a.amp]), P.rises([g for _, g in b.amp])):
                assert (a0, _) == (a1, _)
            for i, j in P.rises([g for _, g in a.amp]):
                before = a.amp[j][0] - a.amp[i][0]
                after = b.amp[j][0] - b.amp[i][0]
                assert after == pytest.approx(before * v**-0.2, abs=1.5e-3)
        for (m0, r0), (m1, r1) in zip(p.body.pitch, q.body.pitch):
            assert m0 == m1 and r1 - 1 == pytest.approx((r0 - 1) * v**2, abs=1e-6)
        for a, b in zip(p.noise, q.noise):
            assert b.band_hz[0] == a.band_hz[0] and b.band_hz[1] == pytest.approx(a.band_hz[1] * v**0.2, abs=0.05)
        if db:
            close(P.softer(P.softer(p, db / 2), db / 2), q)
