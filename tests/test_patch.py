# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The patch: parsing, every problem it names, canonical TOML, and the editing commands (plan §4, §5)."""

import codecs
import copy
import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from halyard_synth import cli, patch
from halyard_synth.curve import amplitude_curve, pitch_curve
from halyard_synth.patch import PatchError, dumps, from_dict, loads

from .conftest import EXAMPLE, edit, make

type Knots = list[dict[str, float]]

TUNE_HZ = 185.0  # the example's tuning
WIRES_MS, BODY_MS, STICK_MS = 260.0, 180.0, 8.0  # where its wires, body and stick end
D3, A4 = 146.832, 440.0
LAYERS = 2  # the example's layers
SOFTER_DB = -4.0 - 12  # its body 12 dB softer
SPACING_MS = 0.01  # knots are at least 10 µs apart
PROBLEMS = 5  # problems reported together by one patch with five
SINGLE_BURST = 5  # knots of a train of one burst
ROUND_US = 1.5e-3  # a rise's rewritten length is within this of exact, in ms
BAND_HZ = 0.05  # an upper band edge is within this of exact
DB = 1e-9  # a level falls by exactly DB, to rounding
GLIDE = 1e-6  # the glide's excess scales within this


def problems(fn: Callable[..., object], *args: object, **kwargs: object) -> list[str]:
    """Return every problem that calling fn reports."""
    with pytest.raises(PatchError) as e:
        fn(*args, **kwargs)
    return e.value.problems


def reject(**edits: object) -> list[str]:
    """Return every problem of the example patch with these fields set by path."""
    return problems(make, **edits)


def amp(*points: tuple[float, float]) -> Knots:
    """Write a shape's knots as TOML parses them."""
    return [{"ms": m, "gain": g} for m, g in points]


def pitch(*points: tuple[float, float]) -> Knots:
    """Write a pitch curve's knots as TOML parses them."""
    return [{"ms": m, "ratio": r} for m, r in points]


def modes(*pairs: tuple[float, float]) -> Knots:
    """Write modes as TOML parses them."""
    return [{"ratio": r, "level_db": v} for r, v in pairs]


# Parsing.


def test_example_parses() -> None:
    """Parse the plan's example: every field, as numbers and tuples."""
    p = loads(EXAMPLE)
    assert p.seed == 1
    assert p.body.tune_hz == TUNE_HZ
    assert p.body.damping == 1.0
    assert p.body.modes == ((1.0, 0.0), (1.593, -5.0), (2.136, -11.0))
    assert p.body.pitch == ((0.0, 1.08), (15.0, 1.0))
    assert len(p.noise) == LAYERS
    assert p.noise[1] == patch.Layer(
        -6.0,
        (2500.0, 9000.0),
        0.5,
        ((0.0, 0.0), (1.5, 1.0), (40.0, 0.4), (140.0, 0.08), (260.0, 0.0)),
    )
    assert p.drive == patch.Drive(6.0)
    assert p.output == patch.Output(48000, "float32", "gain", -6.0, 1)
    assert p.length_ms == WIRES_MS


def random_patch(rng: np.random.Generator) -> patch.Patch:
    """Draw a valid patch from the whole domain, built in code."""

    def times(n: int, first: float, limit: float) -> list[float]:
        return [first, *(first + np.cumsum(rng.uniform(0.01, limit / n, n - 1)))]

    def shape(n: int, first: float, limit: float) -> patch.Knots:
        g = list(rng.uniform(0, 1, n))
        g[0] = g[-1] = 0.0
        g[int(rng.integers(1, n - 1))] = 1.0
        return tuple(zip(times(n, first, limit), g, strict=True))

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
            patch.Layer(
                float(rng.choice([-math.inf, rng.uniform(-90, 24)])),
                (lo, float(rng.uniform(lo + 1, 20000))),
                float(rng.uniform(0, 1)),
                shape(int(rng.integers(3, 33)), float(rng.uniform(0, 5)), 990),
            ),
        )
    rims = []
    for _ in range(int(rng.integers(0, patch.MAX_RIMS + 1))):
        k = int(rng.integers(1, patch.MAX_RIM_MODES + 1))
        hz = np.sort(rng.choice(np.arange(100, 20001), k, replace=False)).astype(float)
        db = list(rng.uniform(-60, 0, k))
        db[int(rng.integers(k))] = 0.0
        rims.append(
            patch.Rim(
                float(rng.choice([-math.inf, rng.uniform(-90, 24)])),
                float(rng.uniform(0, 3)),
                tuple(zip(hz.tolist(), db, strict=True)),
                shape(int(rng.integers(3, 33)), float(rng.uniform(0, 5)), 1990),
            ),
        )
    brushes = []
    for _ in range(int(rng.integers(0, patch.MAX_BRUSHES + 1))):
        lo = float(rng.uniform(40, 9000))
        low = float(rng.uniform(0.5, 1))
        sweep_r = rng.uniform(low, min(2.0, 2 * low, 19999 / (lo + 1)), int(rng.integers(2, 33)))
        sweep_r[int(rng.integers(sweep_r.size))] = low
        brushes.append(
            patch.Brush(
                float(rng.choice([-math.inf, rng.uniform(-90, 24)])),
                (lo, float(rng.uniform(lo + 1, 20000 / sweep_r.max()))),
                float(rng.uniform(0, 1)),
                tuple(zip(times(sweep_r.size, 0.0, 990), sweep_r.tolist(), strict=True)),
                shape(int(rng.integers(3, 33)), float(rng.uniform(0, 5)), 990),
            ),
        )
    mode = str(rng.choice(patch.MODES))
    return patch.Patch(
        seed=int(rng.integers(0, 2**63 - 1)),
        body=patch.Body(
            tune,
            float(rng.choice([-math.inf, rng.uniform(-90, 24)])),
            float(rng.uniform(0, 3)),
            tuple(zip(ratios, levels, strict=True)),
            tuple(zip(times(n, 0.0, 2000), r, strict=True)),
            shape(int(rng.integers(3, 33)), float(rng.uniform(0, 5)), 1990),
        ),
        noise=tuple(layers),
        rim=tuple(rims),
        brush=tuple(brushes),
        drive=patch.Drive(float(rng.uniform(0, 48))),
        output=patch.Output(
            int(rng.choice(patch.SAMPLE_RATES)),
            str(rng.choice(patch.FORMATS)),
            mode,
            float(rng.uniform(-60, 24 if mode == "gain" else 0)),
            int(rng.choice([1, -1])),
        ),
    )


def test_dump_round_trips_exactly() -> None:
    """Write 300 random patches and read them back exactly, the text itself fixed by one round."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = random_patch(rng)
        text = dumps(p)
        assert loads(text) == p
        assert dumps(loads(text)) == text


def test_dump_of_example_keeps_the_plan_layout() -> None:
    """Write the example in the plan's own layout, line by line."""
    ours, plan = dumps(loads(EXAMPLE)).splitlines(), EXAMPLE.splitlines()
    assert len(ours) == len(plan)
    for a, b in zip(ours, plan, strict=True):
        # The plan annotates its layers; the canonical comments only number them.
        assert a.split("#")[0].rstrip() == b.split("#")[0].rstrip()
    assert ours[:25] == plan[:25]


def test_instrument_comes_first(example_dict: dict[str, Any]) -> None:
    """Refuse a patch without the instrument, or of another, alone."""
    d = {k: v for k, v in example_dict.items() if k != "instrument"}
    assert problems(from_dict, d) == ['instrument: missing; a snare patch starts with instrument = "snare"']
    d = dict(example_dict, instrument="kick", colour="red")
    assert problems(from_dict, d) == ["instrument: this is a 'kick' patch, not a \"snare\" one"]


def test_schema_versions(example_dict: dict[str, Any]) -> None:
    """Refuse another or a missing schema, alone, and name a wrongly typed one."""
    for bad, message in ((2, "unsupported version 2"), ("1", "expected an integer"), (True, "expected an integer")):
        assert message in problems(from_dict, dict(example_dict, schema=bad))[0]
    d = {k: v for k, v in example_dict.items() if k != "schema"}
    assert problems(from_dict, d) == ["schema: missing"]
    # A future schema is reported alone, not as a list of field errors.
    assert len(problems(from_dict, dict(example_dict, schema=2, colour="red"))) == 1


def test_integers_accepted_for_floats_but_not_booleans() -> None:
    """Read TOML integers as floats, but no boolean as a number, nor a float as an integer."""
    p = make(**{"body.tune_hz": 185, "drive.db": 0, "noise[1].band_hz": [1500, 16000], "body.modes[2].ratio": 2})
    assert type(p.body.tune_hz) is float
    assert type(p.drive.db) is float
    assert all(type(x) is float for x in p.noise[0].band_hz)
    assert type(p.body.modes[1][0]) is float
    assert "body.damping: expected a number, found a boolean" in reject(**{"body.damping": False})
    assert any("output.sample_rate: expected an integer" in m for m in reject(**{"output.sample_rate": 48000.0}))
    assert any("output.polarity: expected an integer" in m for m in reject(**{"output.polarity": True}))
    assert any("seed: expected an integer" in m for m in reject(seed=1.0))
    assert any("noise[2].rattle: expected a number, found a string" in m for m in reject(**{"noise[2].rattle": "0.5"}))


def test_unknown_and_missing_fields(example_dict: dict[str, Any]) -> None:
    """Report every unknown field and every missing one, by path."""
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


def test_every_field_is_required(example_dict: dict[str, Any]) -> None:
    """Require every field of every section and layer, and every section."""
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


def test_non_finite_numbers() -> None:
    """Allow −inf for silence in levels only, and no other infinity or nan anywhere."""
    assert make(**{"body.level_db": -math.inf}).body.level_db == -math.inf
    assert make(**{"noise[2].level_db": -math.inf}).noise[1].level_db == -math.inf
    for path in ("body.level_db", "noise[1].level_db"):
        assert any("must be finite" in m for m in reject(**{path: math.inf}))
        assert any("must be finite" in m for m in reject(**{path: math.nan}))
    for path in ("body.tune_hz", "body.damping", "drive.db", "output.db", "noise[1].rattle", "body.modes[2].level_db"):
        assert any("must be finite" in m for m in reject(**{path: -math.inf}))
        assert any("must be finite" in m for m in reject(**{path: math.nan}))
    assert loads(EXAMPLE.replace("level_db = -4.0", "level_db = -inf")).body.level_db == -math.inf


def test_knot_times() -> None:
    """Refuse repeated, earlier and closer than 10 µs knot times, as written, and negative times."""
    assert any(
        "repeats the previous knot time" in m for m in reject(**{"body.amp": amp((0, 0), (1, 1), (1, 0.5), (2, 0))})
    )
    assert any(
        "is earlier than the previous knot" in m for m in reject(**{"body.amp": amp((0, 0), (2, 1), (1, 0.5), (3, 0))})
    )
    found = reject(**{"noise[1].amp": amp((0, 0), (0.005, 1), (1, 0))})
    assert any(m.startswith("noise[1].amp[2].ms: is 0.005 ms after the previous knot") for m in found)
    # Exactly 10 µs is allowed, even where binary rounding makes the float gap smaller.
    assert SPACING_MS > 0.03 - 0.02
    make(**{"noise[1].amp": amp((0, 0), (0.02, 0.5), (0.03, 1), (1, 0))})
    assert any("body.amp[1].ms" in m for m in reject(**{"body.amp": amp((-1, 0), (1, 1), (2, 0))}))


def test_amplitude_shapes() -> None:
    """Require shapes from 0 to 0 within 0 … 1, peaking at exactly 1, and allow them to start late."""
    assert "body.amp[1].gain: an amplitude shape must start at 0" in reject(
        **{"body.amp": amp((0, 0.1), (1, 1), (2, 0))},
    )
    assert "body.amp[3].gain: an amplitude shape must end at 0" in reject(**{"body.amp": amp((0, 0), (1, 1), (2, 0.1))})
    found = reject(**{"noise[2].amp": amp((0, 0), (1, 0.9999999), (2, 0))})
    assert "noise[2].amp: the highest gain must be exactly 1, found 0.9999999" in found
    assert "body.amp[2].gain: 1.5 is outside 0 … 1" in reject(**{"body.amp": amp((0, 0), (1, 1.5), (2, 1), (3, 0))})
    assert "body.amp[3].gain: -0.1 is outside 0 … 1" in reject(**{"body.amp": amp((0, 0), (1, 1), (2, -0.1), (3, 0))})
    # Delayed shapes are allowed: the first knot need not be at 0 ms.
    assert make(**{"noise[1].amp": amp((5, 0), (6, 1), (20, 0))}).noise[0].amp[0] == (5.0, 0.0)


def test_pitch_curves() -> None:
    """Require pitch curves from 0 ms to ratio 1, positive, allowing dips below the tuning."""
    assert "body.pitch[1].ms: the pitch curve must start at 0 ms" in reject(**{"body.pitch": pitch((1, 1.1), (15, 1))})
    assert "body.pitch[2].ratio: the pitch curve must end at ratio 1" in reject(
        **{"body.pitch": pitch((0, 1.1), (15, 1.01))},
    )
    assert "body.pitch[1].ratio: must be positive" in reject(**{"body.pitch": pitch((0, -1), (15, 1))})
    # Dips below the tuning are fine while every frequency stays in range.
    make(**{"body.pitch": pitch((0, 1.2), (5, 0.9), (15, 1))})


def test_every_mode_stays_in_the_audio_band_at_every_pitch_knot() -> None:
    """Refuse a glide that takes any mode outside 10 Hz … 20 kHz at any pitch knot."""
    # At 4 kHz, a ratio of 5 is exactly 20 kHz: allowed, but no glide above it.
    make(**{"body.tune_hz": 4000.0, "body.modes": modes((1, 0), (5, -6)), "body.pitch": pitch((0, 1), (15, 1))})
    found = reject(
        **{"body.tune_hz": 4000.0, "body.modes": modes((1, 0), (5, -6)), "body.pitch": pitch((0, 1.001), (15, 1))},
    )
    assert found == ["body.pitch[1].ratio: body frequency 20020 Hz for mode 2 is outside 10 Hz … 20 kHz"]
    found = reject(
        **{
            "body.tune_hz": 40.0,
            "body.modes": modes(
                (1, 0),
            ),
            "body.pitch": pitch((0, 0.2), (15, 1)),
        },
    )
    assert found == ["body.pitch[1].ratio: body frequency 8 Hz is outside 10 Hz … 20 kHz"]


def test_modes() -> None:
    """Allow 1 to 4 modes, the first at ratio 1, increasing to at most 8, the loudest at 0 dB."""
    make(**{"body.modes": modes((1, 0))})
    make(**{"body.modes": modes((1, -3), (1.5, 0), (2, -60), (8, -10))})
    assert "body.modes: needs 1 to 4 modes, found 0" in reject(**{"body.modes": []})
    assert "body.modes: needs 1 to 4 modes, found 5" in reject(
        **{"body.modes": modes((1, 0), (2, 0), (3, 0), (4, 0), (5, 0))},
    )
    assert any(
        m.startswith("body.modes[1].ratio: the first mode defines the tuning")
        for m in reject(**{"body.modes": modes((1.01, 0), (2, -3))})
    )
    assert "body.modes[3].ratio: mode ratios must strictly increase" in reject(
        **{"body.modes": modes((1, 0), (2, -3), (2, -6))},
    )
    assert "body.modes[2].ratio: mode ratios must strictly increase" in reject(**{"body.modes": modes((1, 0), (1, -3))})
    assert any(
        m.startswith("body.modes[2].ratio: 8.01 is outside")
        for m in reject(**{"body.modes": modes((1, 0), (8.01, -3))})
    )
    assert "body.modes: the loudest mode must be exactly 0 dB, found -1.0" in reject(
        **{"body.modes": modes((1, -1), (2, -3))},
    )
    assert any(
        m.startswith("body.modes[2].level_db: -61 is outside")
        for m in reject(**{"body.modes": modes((1, 0), (2, -61))})
    )
    assert any(
        m.startswith("body.modes[1].level_db: 1 is outside") for m in reject(**{"body.modes": modes((1, 1), (2, 0))})
    )


def test_noise_layers(example_dict: dict[str, Any]) -> None:
    """Allow 1 to 4 layers, given as an array of tables."""
    one = dict(example_dict, noise=example_dict["noise"][:1])
    assert len(from_dict(one).noise) == 1
    four = dict(example_dict, noise=example_dict["noise"] * 2)
    assert len(from_dict(four).noise) == patch.MAX_LAYERS
    assert "noise: needs 1 to 4 layers, found 5" in problems(
        from_dict,
        dict(example_dict, noise=example_dict["noise"] * 2 + example_dict["noise"][:1]),
    )
    assert "noise: needs 1 to 4 layers, found 0" in problems(from_dict, dict(example_dict, noise=[]))
    assert any(
        m.startswith("noise: expected an array of tables ([[noise]])")
        for m in problems(from_dict, dict(example_dict, noise=example_dict["noise"][0]))
    )


def test_noise_bands() -> None:
    """Require bands within 20 Hz … 20 kHz, the low edge below the high one."""
    make(**{"noise[1].band_hz": [20, 20000]})
    for bad, path in (
        ([19, 1000], "noise[1].band_hz[1]"),
        ([100, 20001], "noise[1].band_hz[2]"),
        ([500, 500], "noise[1].band_hz"),
        ([900, 800], "noise[1].band_hz"),
    ):
        assert any(m.startswith(path + ":") for m in reject(**{"noise[1].band_hz": bad}))
    assert "noise[2].band_hz: expected [low, high] in Hz" in reject(**{"noise[2].band_hz": [100.0]})


def test_knot_counts_and_lengths() -> None:
    """Require 2 to 32 knots (3 for shapes), within the body's 2 s and the noise's 1 s."""
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
    ("path", "good", "bad"),
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
        ("output.mode", ["gain", "peak", "true-peak"], ["loudness", "truepeak"]),
        ("output.polarity", [1, -1], [0, 2]),
        ("seed", [0, 2**63 - 1], [-1, 2**63]),
    ],
)
def test_domain_edges(path: str, good: list[object], bad: list[object]) -> None:
    """Accept every edge of the valid domain, and refuse just past it."""
    for v in good:
        make(**{path: v, "output.db": min(0.0, -6.0)} if path == "output.mode" else {path: v})
    for v in bad:
        assert any(m.startswith(path) for m in reject(**{path: v}))


def test_peak_target_is_at_most_full_scale() -> None:
    """Allow a peak target up to 0 dBFS, and no higher."""
    for mode in ("peak", "true-peak"):
        make(**{"output.mode": mode, "output.db": 0.0})
        make(**{"output.mode": mode, "output.db": -60.0})
        assert any(m.startswith("output.db") for m in reject(**{"output.mode": mode, "output.db": 0.1}))
    assert any(m.startswith("output.db") for m in reject(**{"output.mode": "peak", "output.db": 0.1}))


def test_all_problems_are_reported_together() -> None:
    """Report five problems of one patch together."""
    found = reject(
        **{
            "body.damping": 4.0,
            "drive.db": 60.0,
            "output.mode": "loud",
            "noise[2].rattle": 2.0,
            "body.amp": amp((0, 0), (1, 0.5), (2, 0)),
        },
    )
    assert len(found) == PROBLEMS


def test_toml_syntax_errors_are_patch_errors() -> None:
    """Report broken TOML as a patch problem."""
    assert problems(loads, 'instrument = "snare"\nschema = 1\n[body\n')[0].startswith("TOML:")


def test_programmatic_edits_are_validated() -> None:
    """Validate a patch built or edited in code, as the parser does."""
    p = loads(EXAMPLE)
    with pytest.raises(PatchError, match=r"body\.damping"):
        dataclasses.replace(p, body=dataclasses.replace(p.body, damping=3.5))
    with pytest.raises(PatchError, match="seed"):
        dataclasses.replace(p, seed=-5)
    with pytest.raises(PatchError, match="noise: needs 1 to 4 layers"):
        dataclasses.replace(p, noise=p.noise * 3)
    hz = 200.0
    assert dataclasses.replace(p, body=dataclasses.replace(p.body, tune_hz=hz)).body.tune_hz == hz


def test_source_length_counts_only_audible_sources() -> None:
    """Find the length the latest end of the audible sources' shapes, 0 for silence."""
    assert make().length_ms == WIRES_MS
    assert make(**{"noise[2].level_db": -math.inf}).length_ms == BODY_MS  # the body, then the stick
    assert make(**{"noise[2].level_db": -math.inf, "body.level_db": -math.inf}).length_ms == STICK_MS
    assert make(**{"body.amp": amp((0, 0), (1, 1), (900, 0)), "body.level_db": -math.inf}).length_ms == WIRES_MS
    silent = {"body.level_db": -math.inf, "noise[1].level_db": -math.inf, "noise[2].level_db": -math.inf}
    assert make(**silent).length_ms == 0.0


def test_parse_tune_and_describe() -> None:
    """Read note names and frequencies, and describe a frequency as a note and cents."""
    assert patch.parse_tune("D3") == D3
    assert patch.parse_tune("A4") == A4
    assert patch.parse_tune("C#4") == patch.parse_tune("Db4")
    assert patch.parse_tune("185.5") == TUNE_HZ + 0.5
    for bad in ("H2", "nan", "inf", ""):
        with pytest.raises(ValueError, match="not a note name or frequency"):
            patch.parse_tune(bad)
    assert patch.describe_tuning(185.0) == "F#3 +0.0 cents"
    assert patch.describe_tuning(190.0) == "F#3 +46.2 cents"
    assert patch.describe_tuning(440.0) == "A4 +0.0 cents"
    assert patch.note_name(50) == "D3"


# Editing commands.


def test_tune_sets_the_fundamental_and_keeps_the_ratios() -> None:
    """Retune a patch, changing nothing else, and refuse a tuning that takes a mode out of range."""
    p = loads(EXAMPLE)
    q = patch.tune(p, "D3")
    assert q.body.tune_hz == D3
    assert dataclasses.replace(q, body=dataclasses.replace(q.body, tune_hz=TUNE_HZ)) == p
    assert patch.tune(p, "220").body.tune_hz == A4 / 2
    # A mode or the glide leaving the audio band fails; nothing is clamped.
    high = make(**{"body.modes": modes((1, 0), (7, -6))})
    high_hz = 2500.0
    assert patch.tune(high, str(high_hz)).body.tune_hz == high_hz
    found = problems(patch.tune, high, "3000")
    assert found == [
        "body.pitch[1].ratio: body frequency 22680 Hz for mode 2 is outside 10 Hz … 20 kHz",
        "body.pitch[2].ratio: body frequency 21000 Hz for mode 2 is outside 10 Hz … 20 kHz",
    ]
    assert any(m.startswith("body.tune_hz") for m in problems(patch.tune, p, "C9"))
    with pytest.raises(ValueError, match="not a note name or frequency"):
        patch.tune(p, "X3")


def test_stretch_rescales_after_each_peak() -> None:
    """Stretch every shape after its peak, or chosen curves after a time, writing times to 1 µs."""
    p = loads(EXAMPLE)
    q = patch.stretch(p, "all", 0.8)
    assert q.body.amp == ((0.0, 0.0), (0.4, 1.0), (20.08, 0.45), (72.08, 0.1), (144.08, 0.0))
    assert q.noise[0].amp == ((0.0, 0.0), (0.05, 1.0), (0.13, 0.7), (2.41, 0.1), (6.41, 0.0))
    assert q.noise[1].amp == ((0.0, 0.0), (1.5, 1.0), (32.3, 0.4), (112.3, 0.08), (208.3, 0.0))
    assert q.body.pitch == p.body.pitch  # `all` names the amplitude shapes
    assert patch.stretch(p, ["body.pitch"], 2.0).body.pitch == ((0.0, 1.08), (30.0, 1.0))
    r = patch.stretch(p, ["noise[2].amp", "body.amp"], 1.5, after_ms=10.0)
    assert r.noise[1].amp == ((0.0, 0.0), (1.5, 1.0), (55.0, 0.4), (205.0, 0.08), (385.0, 0.0))
    assert r.body.amp[:2] == p.body.amp[:2]
    assert r.body.amp[2] == (32.5, 0.45)
    assert r.noise[0] == p.noise[0]
    # Times are written to 1 µs.
    assert patch.stretch(p, "noise[1].amp", 1 / 3).noise[0].amp == (
        (0.0, 0.0),
        (0.05, 1.0),
        (0.083, 0.7),
        (1.033, 0.1),
        (2.7, 0.0),
    )


def test_stretch_failures() -> None:
    """Refuse unknown curves, factors that are not positive, a negative anchor, and a curve stretched too long."""
    p = loads(EXAMPLE)
    assert problems(patch.stretch, p, "noise[3].amp", 2.0)[0].startswith("noise[3].amp: not a curve of this patch")
    assert problems(patch.stretch, p, "attack.amp", 2.0)[0].startswith("attack.amp: not a curve of this patch")
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(PatchError, match="the factor must be a positive number"):
            patch.stretch(p, "all", bad)
    with pytest.raises(PatchError, match="--after"):
        patch.stretch(p, "all", 2.0, after_ms=-1.0)
    # Stretching past a curve's span limit fails rather than clamping.
    assert any(m.startswith("noise[2].amp[5].ms") for m in problems(patch.stretch, p, "noise[2].amp", 4.0))


def test_bursts_write_a_clap_train() -> None:
    """Write four bursts as peaks, shoulders on their exponentials, dips and a tail, and change nothing else."""
    p = loads(EXAMPLE)
    q = patch.bursts(p, "noise[1]", patch.Train(4, 11.0))
    knots = q.noise[0].amp
    assert len(knots) == 3 * 4 + 2
    assert knots[0] == (0.0, 0.0)
    peaks = [knots[1 + 3 * i] for i in range(4)]
    assert peaks == [(0.1, 1.0), (11.1, 1.0), (22.1, 1.0), (33.1, 1.0)]
    shoulders = [knots[2 + 3 * i] for i in range(4)]
    assert [round(ms - pk[0], 6) for (ms, _), pk in zip(shoulders, peaks, strict=True)] == [0.1] * 4
    dips = [knots[3 + 3 * i] for i in range(3)]
    assert dips == [(11.0, 0.1), (22.0, 0.1), (33.0, 0.1)]
    assert knots[-2] == (59.767, 0.1)
    assert knots[-1] == (113.1, 0.0)
    assert q.noise[1] == p.noise[1]
    assert q.body == p.body
    # The shoulder lies on the exponential that the burst's decay follows.
    assert shoulders[0][1] == pytest.approx(math.exp(-math.log(10) / 10.9 * 0.1), abs=5e-5)
    assert shoulders[-1][1] == pytest.approx(10 ** (-3 * 0.1 / 80), abs=5e-5)


def test_burst_options() -> None:
    """Follow the train's start, growth, rise, dip, taper and tail, rising or single, with silent dips."""
    p = loads(EXAMPLE)
    q = patch.bursts(
        p,
        "noise[2]",
        patch.Train(3, 10.0, start=2.0, growth=1.5, rise=0.2, dip=0.25, taper=0.5, tail=50.0),
    )
    knots = q.noise[1].amp
    assert len(knots) == 3 * 3 + 3
    assert knots[:2] == ((0.0, 0.0), (2.0, 0.0))
    assert [knots[2 + 3 * i] for i in range(3)] == [(2.2, 1.0), (12.2, 0.5), (27.2, 0.25)]  # gaps 10, then 15
    assert [knots[4 + 3 * i] for i in range(2)] == [(12.0, 0.125), (27.0, 0.0625)]  # dips under the lower peak
    assert knots[-1] == (77.2, 0.0)
    rising = patch.bursts(p, "noise[2]", patch.Train(3, 10.0, taper=2.0)).noise[1].amp
    assert [rising[1 + 3 * i][1] for i in range(3)] == [0.25, 0.5, 1.0]
    single = patch.bursts(p, "noise[1]", patch.Train(1, 10.0)).noise[0].amp
    assert len(single) == SINGLE_BURST
    assert single[1] == (0.1, 1.0)
    silent_dips = patch.bursts(p, "noise[1]", patch.Train(2, 10.0, dip=0.0)).noise[0].amp
    assert silent_dips[3] == (10.0, 0.0)


def test_burst_shapes_are_ordinary_valid_shapes() -> None:
    """Write 200 random trains as valid shapes, each burst one rise between flat knots, or refuse them for cause."""
    p = loads(EXAMPLE)
    rng = np.random.default_rng(3)
    failures: list[list[str]] = []
    for _ in range(200):
        n = int(rng.integers(1, 11))
        kwargs = {
            "start": float(rng.choice([0.0, rng.uniform(0.01, 20)])),
            "growth": float(rng.uniform(0.8, 1.3)),
            "rise": float(rng.uniform(0.01, 1.0)),
            "dip": float(rng.uniform(0, 0.9)),
            "taper": float(rng.uniform(0.5, 2)),
            "tail": float(rng.uniform(1, 300)),
        }
        try:
            q = patch.bursts(p, "noise[1]", patch.Train(n, float(rng.uniform(1, 30)), **kwargs))
        except PatchError as e:
            failures.append(e.problems)
            continue
        knots = q.noise[0].amp
        assert len(knots) == 3 * n + 2 + (kwargs["start"] > 0)
        gains = [g for _, g in knots]
        assert max(gains) == 1.0
        assert gains[0] == gains[-1] == 0.0
        # Each burst's rise is one segment between zero-slope knots.
        curve = amplitude_curve([ms / 1000 for ms, _ in knots], gains)
        assert [b - a for a, b in patch.rises(gains)] == [1] * n
        for a, b in patch.rises(gains):
            assert curve.d[a] == 0.0
            assert curve.d[b] == 0.0
    # The only acceptable failures: too many knots, a rise that does not fit, or too long a train.
    assert all("knots" in m or "--rise" in m or ".ms:" in m for found in failures for m in found), failures


def test_burst_failures() -> None:
    """Refuse an unknown layer, too many bursts, settings out of range, rises that do not fit, and too many knots."""
    p = loads(EXAMPLE)
    assert problems(patch.bursts, p, "noise[3]", patch.Train(4, 10.0))[0].startswith(
        "noise[3]: not a layer of this patch",
    )
    assert problems(patch.bursts, p, "noise[1]", patch.Train(11, 10.0)) == ["--count: needs 1 to 10 bursts, found 11"]
    assert problems(patch.bursts, p, "noise[1]", patch.Train(0, 10.0)) == ["--count: needs 1 to 10 bursts, found 0"]
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
        assert problems(patch.bursts, p, "noise[1]", patch.Train(4, 10.0, **{option: value}))[0].startswith(
            f"--{option}:",
        )
    assert problems(patch.bursts, p, "noise[1]", patch.Train(4, 0.0))[0].startswith("--spacing:")
    assert problems(patch.bursts, p, "noise[1]", patch.Train(4, 1.0, rise=0.9))[0].startswith(
        "--rise: burst 1 rises for 0.9 ms",
    )
    assert problems(patch.bursts, p, "noise[1]", patch.Train(10, 5.0, start=1.0)) == [
        "noise[1].amp: 10 bursts need 33 knots, more than the 32 allowed",
    ]
    make_long = problems(patch.bursts, p, "noise[1]", patch.Train(10, 100.0, tail=200.0))
    assert any(m.startswith("noise[1].amp[") and "outside 0 … 1000" in m for m in make_long)


def test_rises() -> None:
    """Find every maximal run of increasing knots."""
    assert patch.rises([0, 1, 0.45, 0.1, 0]) == [(0, 1)]
    assert patch.rises([0, 0.5, 1, 0.2, 0.2, 0.7, 0]) == [(0, 2), (4, 5)]
    assert patch.rises([0, 0.5, 0.5, 1, 0]) == [(0, 1), (2, 3)]
    assert patch.rises([0, 0, 1, 0]) == [(1, 2)]


def test_softer_follows_the_plan_figures() -> None:
    """Find the plan's figures 12 dB softer: levels, glide, rises shifting later knots, upper edges, nothing else."""
    p = loads(EXAMPLE)
    q = patch.softer(p, 12)
    assert q.body.level_db == SOFTER_DB
    assert [n.level_db for n in q.noise] == [-22.0, -18.0]
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


def test_softer_by_zero_changes_nothing_and_exponents_turn_rules_off() -> None:
    """Return the patch itself 0 dB softer, turn each rule off with its exponent, and keep silence silent."""
    p = loads(EXAMPLE)
    assert patch.softer(p, 0) is p
    q = patch.softer(p, 12, glide=0, contact=0)
    assert q.body.pitch == p.body.pitch
    assert q.body.amp == p.body.amp
    assert [(n.amp, n.band_hz) for n in q.noise] == [(n.amp, n.band_hz) for n in p.noise]
    assert q.body.level_db == SOFTER_DB
    silent = make(**{"noise[1].level_db": -math.inf})
    assert patch.softer(silent, 6).noise[0].level_db == -math.inf


def sources(p: patch.Patch) -> tuple[patch.Body | patch.Layer, ...]:
    """Return the body and the layers: every source with a level and a shape."""
    return (p.body, *p.noise)


def close(p: patch.Patch, q: patch.Patch) -> None:
    """Check two patches equal up to the decimal rounding softer writes.

    One edit keeps every knot within 0.5 µs of its exact place. A second edit
    lengthens the rises as written, each up to 1 µs off, so a knot may move by
    up to 1 µs more for every rise before it.
    """
    assert p.body.level_db == pytest.approx(q.body.level_db, abs=2e-6)
    for (m1, r1), (m2, r2) in zip(p.body.pitch, q.body.pitch, strict=True):
        assert m1 == m2
        assert r1 == pytest.approx(r2, abs=2e-6)
    for a, b in zip(sources(p), sources(q), strict=True):
        ends = [j for _, j in patch.rises([g for _, g in b.amp])]
        for k, ((m1, g1), (m2, g2)) in enumerate(zip(a.amp, b.amp, strict=True)):
            before = sum(j < k for j in ends)
            assert m1 == pytest.approx(m2, abs=1e-3 * (1.001 + before))
            assert g1 == g2
    for a, b in zip(p.noise, q.noise, strict=True):
        assert a.level_db == pytest.approx(b.level_db, abs=2e-6)
        assert a.band_hz[0] == b.band_hz[0]
        assert a.band_hz[1] == pytest.approx(b.band_hz[1], abs=0.2)


def test_softer_composes() -> None:
    """Find a + b dB softer equal to a then b, up to the rounding softer writes."""
    p = loads(EXAMPLE)
    clap = patch.bursts(p, "noise[1]", patch.Train(4, 11.0))
    for base in (p, clap):
        for a, b in ((6, 6), (3, 9), (12, 12), (0.5, 20)):
            close(patch.softer(patch.softer(base, a), b), patch.softer(base, a + b))


def test_softer_rules_are_exact_on_the_whole_curve() -> None:
    """Find each rewritten segment the original read at a mapped time, and the glide's excess scaled everywhere."""
    # Each segment of a rewritten shape is the original segment read at a linearly mapped time,
    # and the glide's excess scales on the whole curve, not only at its knots.
    p = patch.bursts(loads(EXAMPLE), "noise[1]", patch.Train(4, 11.0))
    for db in (6, 12, 24, 36):
        q = patch.softer(p, db)
        for old, new in zip(
            (p.body.amp, *[n.amp for n in p.noise]),
            (q.body.amp, *[n.amp for n in q.noise]),
            strict=True,
        ):
            t0, g = np.array([ms for ms, _ in old]) / 1000, [g for _, g in old]
            t1 = np.array([ms for ms, _ in new]) / 1000
            e0, e1 = amplitude_curve(t0, g), amplitude_curve(t1, g)
            u = np.linspace(0, 1, 101)
            for i in range(len(t0) - 1):
                np.testing.assert_allclose(
                    e1(t1[i] + u * (t1[i + 1] - t1[i])),
                    e0(t0[i] + u * (t0[i + 1] - t0[i])),
                    rtol=0,
                    atol=1e-12,
                )
        a = (10 ** (-db / 20)) ** 2
        t = np.array([ms for ms, _ in p.body.pitch]) / 1000
        r = np.array([ratio for _, ratio in p.body.pitch])
        x = np.linspace(-0.001, t[-1] + 0.001, 997)
        np.testing.assert_allclose(
            pitch_curve(t, 1 + a * (r - 1))(x),
            1 + a * (pitch_curve(t, r)(x) - 1),
            rtol=0,
            atol=1e-15,
        )


def test_softer_fails_rather_than_clamping() -> None:
    """Refuse what leaves the domain, a non-finite DB, and a harder hit too hard, never clamping."""
    p = loads(EXAMPLE)
    found = problems(patch.softer, p, 90)
    assert "body.level_db: -94 is outside -90 … 24 (or -inf for silence)" in found
    assert "noise[2].band_hz: the low edge 2500 Hz must be below the high edge 1133 Hz" in found
    for bad in (math.nan, math.inf):
        with pytest.raises(PatchError, match="DB: must be a finite number"):
            patch.softer(p, bad)
    # A harder hit is the inverse edit, and fails where it leaves the domain.
    back = patch.softer(patch.softer(p, -3), 3).noise[0].amp
    np.testing.assert_allclose(np.array(back), np.array(p.noise[0].amp), rtol=0, atol=1e-3)
    assert any(m.startswith("noise[1].band_hz[2]") for m in problems(patch.softer, p, -12))


@pytest.mark.parametrize("name", ["tone", "crack", "fat", "driven", "clap", "snap"])
def test_softer_ladders_on_every_reference(name: str) -> None:
    """Make each reference 0 to 36 dB softer, by the plan's rules exactly, to the decimals softer writes."""
    # Plan §8.1: levels fall by exactly DB, the glide's excess scales by v², rises by v^(−1/5) and upper
    # band edges by v^(1/5), up to the decimals softer writes; and the ladder composes.
    p = loads(cli.reference_text(name))
    for db in (0, 6, 12, 18, 24, 30, 36):
        q = patch.softer(p, db)
        v = 10 ** (-db / 20)
        for a, b in zip(sources(p), sources(q), strict=True):
            assert b.level_db == (a.level_db if a.level_db == -math.inf else pytest.approx(a.level_db - db, abs=DB))
            for (a0, _), (a1, _) in zip(
                patch.rises([g for _, g in a.amp]),
                patch.rises([g for _, g in b.amp]),
                strict=True,
            ):
                assert (a0, _) == (a1, _)
            for i, j in patch.rises([g for _, g in a.amp]):
                before = a.amp[j][0] - a.amp[i][0]
                after = b.amp[j][0] - b.amp[i][0]
                assert after == pytest.approx(before * v**-0.2, abs=ROUND_US)
        for (m0, r0), (m1, r1) in zip(p.body.pitch, q.body.pitch, strict=True):
            assert m0 == m1
            assert r1 - 1 == pytest.approx((r0 - 1) * v**2, abs=GLIDE)
        for a, b in zip(p.noise, q.noise, strict=True):
            assert b.band_hz[0] == a.band_hz[0]
            assert b.band_hz[1] == pytest.approx(a.band_hz[1] * v**0.2, abs=BAND_HZ)
        if db:
            close(patch.softer(patch.softer(p, db / 2), db / 2), q)


def test_numbers_too_large_for_a_float_are_named() -> None:
    """Name an integer too large for a float, wherever a float is read from it."""
    huge = 10**400
    assert "body.tune_hz: is too large" in reject(**{"body.tune_hz": huge})
    assert "body.modes[2].ratio: is too large" in reject(**{"body.modes[2].ratio": huge})


def test_wrongly_typed_tables_are_named(example_dict: dict[str, Any]) -> None:
    """Name a table, an array of tables or a pair that is the wrong kind of value."""
    assert problems(from_dict, [example_dict]) == ["expected a table"]
    for path, value, message in (
        ("drive", 5, "drive: expected a table, found int"),
        ("body.amp", "x", "body.amp: expected an array of { ms, gain } tables, found a string"),
        ("body.modes", [1, 2], "body.modes[1]: expected a { ratio, level_db } table, found int"),
        ("body.pitch", [{"ms": 0.0}], "body.pitch[1].ratio: missing"),
        ("noise[1].amp", {"ms": 0}, "noise[1].amp: expected an array of { ms, gain } tables, found a table"),
    ):
        assert message in problems(from_dict, edit(copy.deepcopy(example_dict), path, value))


def test_programmatic_parts_must_have_their_types() -> None:
    """Name a part built in code with the wrong type: sections, layers, knots and modes."""
    p = loads(EXAMPLE)
    body = p.body
    knots = cast("Any", [(0.0, 0.0)])  # deliberately wrong: a list, not a tuple of pairs
    single = cast("Any", ((1.0,),))  # deliberately wrong: a mode of one number
    cases: list[tuple[dict[str, object], str]] = [
        ({"body": None}, "body: expected a body section"),
        ({"noise": [p.noise[0]]}, "noise: expected noise layers"),
        ({"drive": None}, "drive: expected a drive section"),
        ({"output": None}, "output: expected an output section"),
        ({"body": dataclasses.replace(body, amp=knots)}, "body.amp: expected (ms, gain) knots"),
        ({"body": dataclasses.replace(body, modes=single)}, "body.modes: expected (ratio, level_db) modes"),
    ]
    for fields, message in cases:
        assert message in problems(dataclasses.replace, p, **fields)


def test_softer_keeps_the_knots_inside_a_rise_in_their_place() -> None:
    """Lengthen a rise of several knots in proportion, every knot inside it keeping its place."""
    p = make(**{"noise[1].amp": amp((0, 0), (0.05, 0.5), (0.1, 1), (3, 0.1), (8, 0))})
    rise = patch.softer(p, 12, glide=0).noise[0].amp
    factor = 10 ** (12 / 20 * 0.2)
    assert rise[2][0] == pytest.approx(0.1 * factor, abs=1e-3)
    assert rise[1][0] == pytest.approx(rise[2][0] / 2, abs=1e-3)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"])
def test_patches_are_read_in_utf8_or_utf16(tmp_path: Path, encoding: str) -> None:
    """Read a patch written in UTF-8, with or without a BOM, or in UTF-16 with one in either byte order."""
    path = tmp_path / "snare.toml"
    bom = {"utf-16-le": codecs.BOM_UTF16_LE, "utf-16-be": codecs.BOM_UTF16_BE}.get(encoding, b"")
    path.write_bytes(bom + EXAMPLE.encode(encoding))
    assert patch.load(path) == loads(EXAMPLE)


def test_a_patch_that_is_not_text_is_refused(tmp_path: Path) -> None:
    """Refuse a file that is neither UTF-8 nor UTF-16."""
    path = tmp_path / "binary.toml"
    path.write_bytes(b"\xff\xfe\x00\xd8" + b"\xff" * 7)  # a UTF-16 BOM, then no valid text
    assert problems(patch.load, path) == ["not UTF-8 or UTF-16 text"]
    path.write_bytes(b"instrument = \xff\n")
    assert problems(patch.load, path) == ["not UTF-8 or UTF-16 text"]


# Rims.


def rim(*hz: tuple[float, float], level_db: float = -6.0) -> dict[str, Any]:
    """Write one [[rim]] table as TOML parses it."""
    return {
        "level_db": level_db,
        "damping": 1.0,
        "modes": [{"hz": f, "level_db": v} for f, v in hz],
        "amp": amp((0, 0), (0.15, 1), (1.2, 0.3), (8, 0)),
    }


RIM = rim((1630, 0), (4210, -8))
LATE_RIM_MS = 400.0  # where a long rim's shape ends


def test_rims_are_optional_and_at_most_two(example_dict: dict[str, Any]) -> None:
    """Read no [[rim]] as none, one or two as written, and refuse three or anything but tables."""
    assert from_dict(example_dict).rim == ()
    one = from_dict(dict(example_dict, rim=[RIM]))
    assert one.rim == (
        patch.Rim(-6.0, 1.0, ((1630.0, 0.0), (4210.0, -8.0)), ((0.0, 0.0), (0.15, 1.0), (1.2, 0.3), (8.0, 0.0))),
    )
    assert len(from_dict(dict(example_dict, rim=[RIM, RIM])).rim) == patch.MAX_RIMS
    assert "rim: at most 2 rims, found 3" in problems(from_dict, dict(example_dict, rim=[RIM] * 3))
    assert "rim: expected an array of tables ([[rim]]), found a table" in problems(
        from_dict,
        dict(example_dict, rim=RIM),
    )
    assert "rim[1].damping: missing" in problems(
        from_dict,
        dict(example_dict, rim=[{k: v for k, v in RIM.items() if k != "damping"}]),
    )
    assert "rim[1].amp: expected an array of { ms, gain } tables, found a string" in problems(
        from_dict,
        dict(example_dict, rim=[dict(RIM, amp="x")]),
    )
    assert "rim[2].modes[1]: expected a { hz, level_db } table, found int" in problems(
        from_dict,
        dict(example_dict, rim=[RIM, dict(RIM, modes=[1])]),
    )


def test_rim_modes(example_dict: dict[str, Any]) -> None:
    """Require 1 to 8 modes from 100 Hz to 20 kHz, strictly increasing, the loudest exactly 0 dB."""

    def found(r: dict[str, Any]) -> list[str]:
        return problems(from_dict, dict(example_dict, rim=[r]))

    from_dict(dict(example_dict, rim=[rim((100, 0)), rim(*[(1000 * k, -k + 1.0) for k in range(1, 9)])]))
    assert "rim[1].modes: needs 1 to 8 modes, found 0" in found(rim())
    assert "rim[1].modes: needs 1 to 8 modes, found 9" in found(rim(*[(1000 * k, 0.0) for k in range(1, 10)]))
    assert any(m.startswith("rim[1].modes[1].hz:") for m in found(rim((99, 0))))
    assert any(m.startswith("rim[1].modes[2].hz:") for m in found(rim((1000, 0), (20001, -3))))
    assert "rim[1].modes[2].hz: rim modes must strictly increase" in found(rim((1000, 0), (1000, -3)))
    assert any(m.startswith("rim[1].modes[2].level_db:") for m in found(rim((1000, 0), (2000, -61))))
    assert "rim[1].modes: the loudest mode must be exactly 0 dB, found -3.0" in found(rim((1000, -3), (2000, -6)))
    assert any(m.startswith("rim[1].level_db:") for m in found(rim((1000, 0), level_db=25)))
    assert from_dict(dict(example_dict, rim=[rim((1000, 0), level_db=-math.inf)])).rim[0].level_db == -math.inf
    p = from_dict(dict(example_dict, rim=[RIM]))
    single = cast("Any", ((1000.0,),))  # deliberately wrong: a mode of one number
    assert "rim: expected rims" in problems(dataclasses.replace, p, rim=[p.rim[0]])
    assert "rim[1].modes: expected (hz, level_db) modes" in problems(
        dataclasses.replace,
        p,
        rim=(dataclasses.replace(p.rim[0], modes=single),),
    )


def test_rims_are_written_after_the_layers_and_only_when_there_are_some(example_dict: dict[str, Any]) -> None:
    """Write each rim as a [[rim]] block after the layers, and none when there is none."""
    assert "[[rim]]" not in dumps(loads(EXAMPLE))
    p = from_dict(dict(example_dict, rim=[RIM, rim((700, 0), level_db=-math.inf)]))
    text = dumps(p)
    assert text.count("[[rim]]") == len(p.rim)
    assert text.index("[[noise]]") < text.index("[[rim]]") < text.index("[drive]")
    assert loads(text) == p
    assert patch.to_dict(p)["rim"][1]["level_db"] == -math.inf


def test_rims_count_in_the_length_and_take_their_edits(example_dict: dict[str, Any]) -> None:
    """Count an audible rim's shape in the length, stretch it by its path, and strike it softer."""
    late = dict(RIM, amp=amp((0, 0), (0.15, 1), (1.2, 0.3), (LATE_RIM_MS, 0)))
    p = from_dict(dict(example_dict, rim=[late]))
    assert p.length_ms == LATE_RIM_MS
    assert from_dict(dict(example_dict, rim=[dict(late, level_db=-math.inf)])).length_ms == WIRES_MS
    assert patch.curve_names(p)[-1] == "rim[1].amp"
    assert patch.stretch(p, "rim[1].amp", 0.5).rim[0].amp == ((0.0, 0.0), (0.15, 1.0), (0.675, 0.3), (200.075, 0.0))
    assert patch.stretch(p, "all", 0.5).rim[0].amp[-1] == (200.075, 0.0)
    assert problems(patch.stretch, p, "rim[2].amp", 2.0)[0].startswith("rim[2].amp: not a curve of this patch")
    q = patch.softer(p, 12)
    assert q.rim[0].level_db == p.rim[0].level_db - 12
    assert q.rim[0].amp[1] == (0.198, 1.0)  # the rise lengthens by v^(−1/5), as the others'
    assert (q.rim[0].modes, q.rim[0].damping) == (p.rim[0].modes, p.rim[0].damping)
    assert patch.softer(p, 12, contact=0).rim[0].amp == p.rim[0].amp


# Brushes.


type Sweep = tuple[tuple[float, float], ...]


def brush(sweep: Sweep = ((0, 1.0), (60, 1.5), (180, 0.9)), **fields: object) -> dict[str, Any]:
    """Write one [[brush]] table as TOML parses it."""
    return {
        "level_db": -3.0,
        "band_hz": [900.0, 7000.0],
        "rattle": 0.2,
        "sweep": [{"ms": m, "ratio": r} for m, r in sweep],
        "amp": amp((0, 0), (25, 1), (120, 0.5), (240, 0)),
        **fields,
    }


BRUSH_MS = 240.0  # where the brush ends


def test_a_brush_is_optional_and_alone(example_dict: dict[str, Any]) -> None:
    """Read no [[brush]] as none and one as written, and refuse two or anything but tables."""
    assert from_dict(example_dict).brush == ()
    b = from_dict(dict(example_dict, brush=[brush()])).brush[0]
    assert b == patch.Brush(
        -3.0,
        (900.0, 7000.0),
        0.2,
        ((0.0, 1.0), (60.0, 1.5), (180.0, 0.9)),
        ((0.0, 0.0), (25.0, 1.0), (120.0, 0.5), (240.0, 0.0)),
    )
    assert "brush: at most 1 brush, found 2" in problems(from_dict, dict(example_dict, brush=[brush()] * 2))
    found = problems(from_dict, dict(example_dict, brush=brush()))
    assert "brush: expected an array of tables ([[brush]]), found a table" in found
    no_sweep = {k: v for k, v in brush().items() if k != "sweep"}
    assert "brush[1].sweep: missing" in problems(from_dict, dict(example_dict, brush=[no_sweep]))
    assert "brush[1].rattle: missing" in problems(
        from_dict,
        dict(example_dict, brush=[{k: v for k, v in brush().items() if k != "rattle"}]),
    )
    assert "brush[1].sweep[1]: expected a { ms, ratio } table, found int" in problems(
        from_dict,
        dict(example_dict, brush=[dict(brush(), sweep=[1])]),
    )
    p = from_dict(dict(example_dict, brush=[brush()]))
    assert "brush: expected brushes" in problems(dataclasses.replace, p, brush=[p.brush[0]])
    narrow = cast("Any", (900.0,))  # deliberately wrong: a band of one edge
    found = problems(dataclasses.replace, p, brush=(dataclasses.replace(p.brush[0], band_hz=narrow),))
    assert found == ["brush[1].band_hz: expected [low, high] in Hz"]  # and the sweep's check of the band waits for it


def test_a_brush_sweeps_within_an_octave_and_the_audio_band(example_dict: dict[str, Any]) -> None:
    """Require sweep ratios within 0.5 … 2, an octave of travel at most, and the band within 20 Hz … 20 kHz."""

    def found(sweep: Sweep = ((0, 1.0), (60, 1.5)), **fields: object) -> list[str]:
        return problems(from_dict, dict(example_dict, brush=[brush(sweep, **fields)]))

    from_dict(dict(example_dict, brush=[brush(((0, 0.5), (100, 1.0)), band_hz=[40.0, 20000.0])]))
    from_dict(dict(example_dict, brush=[brush(((0, 1.0), (100, 2.0)), band_hz=[20.0, 10000.0])]))
    assert "brush[1].sweep[2].ratio: 2.1 is outside 0.5 … 2" in found(sweep=((0, 1), (50, 2.1)))
    assert "brush[1].sweep[1].ratio: 0.4 is outside 0.5 … 2" in found(sweep=((0, 0.4), (50, 1)))
    assert "brush[1].sweep: travels 3 times, more than an octave" in found(sweep=((0, 0.6), (50, 1.8)))
    low = found(sweep=((0, 0.5), (50, 1)), band_hz=[30.0, 1000.0])
    assert "brush[1].sweep: takes the low edge to 15 Hz, below 20 Hz" in low
    high = found(sweep=((0, 1), (50, 1.5)), band_hz=[900.0, 14000.0])
    assert "brush[1].sweep: takes the high edge to 21000 Hz, above 20 kHz" in high
    assert any(m.startswith("brush[1].sweep[2].ms") for m in found(sweep=((0, 1), (1001, 1))))
    assert "brush[1].sweep: needs 2 to 32 knots, found 1" in found(sweep=((0, 1),))
    assert any(m.startswith("brush[1].band_hz") for m in found(band_hz=[7000.0, 900.0]))
    assert any(m.startswith("brush[1].amp") for m in found(amp=amp((0, 0), (5, 0.5), (9, 0))))


def test_a_brush_is_written_after_the_rims(example_dict: dict[str, Any]) -> None:
    """Write a brush as a [[brush]] block after the rims, and none when there is none."""
    assert "[[brush]]" not in dumps(loads(EXAMPLE))
    p = from_dict(dict(example_dict, rim=[RIM], brush=[brush(level_db=-math.inf)]))
    text = dumps(p)
    assert text.index("[[rim]]") < text.index("[[brush]]") < text.index("[drive]")
    assert loads(text) == p
    assert patch.to_dict(p)["brush"][0]["sweep"][1] == {"ms": 60.0, "ratio": 1.5}


def test_a_brush_counts_in_the_length_and_takes_its_edits(example_dict: dict[str, Any]) -> None:
    """Count an audible brush in the length, stretch its sweep and shape by path, and strike it softer."""
    late = brush(amp=amp((0, 0), (25, 1), (120, 0.5), (400, 0)))
    p = from_dict(dict(example_dict, brush=[late]))
    assert p.length_ms == LATE_RIM_MS
    assert from_dict(dict(example_dict, brush=[dict(late, level_db=-math.inf)])).length_ms == WIRES_MS
    assert patch.curve_names(p)[-2:] == ("brush[1].sweep", "brush[1].amp")
    # A sweep stretches from 0 ms, like the pitch; `all` names the amplitude shapes only.
    assert patch.stretch(p, "brush[1].sweep", 0.5).brush[0].sweep == ((0.0, 1.0), (30.0, 1.5), (90.0, 0.9))
    assert patch.stretch(p, "all", 0.5).brush[0].sweep == p.brush[0].sweep
    assert patch.stretch(p, "all", 0.5).brush[0].amp[-1] == (212.5, 0.0)
    q = patch.softer(p, 12)
    assert q.brush[0].level_db == p.brush[0].level_db - 12
    assert q.brush[0].band_hz == (900.0, 5310.0)  # the upper edge falls as a layer's
    assert q.brush[0].amp[1] == (32.956, 1.0)  # the rise lengthens by v^(−1/5), as a layer's
    assert (q.brush[0].sweep, q.brush[0].rattle) == (p.brush[0].sweep, p.brush[0].rattle)
