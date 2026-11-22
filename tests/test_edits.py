# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""What every editing command writes, pinned: the same input gives the same patch in every version.

A patch an edit printed is kept and rendered later, and a take is drawn anew at
every render, so each is pinned by the SHA-256 of its canonical text. A change
fails here until it is made deliberately, as a new version (CONTRIBUTING.md).
"""

import dataclasses
import functools
import hashlib
import math
from collections.abc import Callable

import pytest

from halyard_synth import cli, patch, tuning
from halyard_synth.patch import Patch, PatchError

from .conftest import EXAMPLE


def ref(name: str) -> Patch:
    """Return a preset."""
    return patch.loads(cli.preset_text(name))


def takes_of(name: str, strength_db: float, timing_ms: float, seed: int) -> Patch:
    """Return one take of a preset given takes that differ as played."""
    return patch.played(dataclasses.replace(ref(name), seed=seed, takes=patch.Takes(strength_db, timing_ms)))


def two_rims() -> Patch:
    """Return a rimshot with a rim click stacked on it: two rims, the second silent."""
    two = patch.stack(ref("rimshot"), ref("rim-click"))
    return dataclasses.replace(two, rim=(two.rim[0], dataclasses.replace(two.rim[1], level_db=-math.inf)))


EDITS: dict[str, Callable[[], Patch]] = {
    "roll-build": lambda: patch.roll(patch.loads(EXAMPLE), patch.Roll(3, 60.0, 0.8, 6.0)),
    "roll-rimshot": lambda: patch.roll(ref("rimshot"), patch.Roll(2, 40.0), glide=1.0, contact=0.3),
    "strokes": lambda: patch.strokes(patch.loads(EXAMPLE), [(0.0, 6.0), (35.0, 0.0), (70.0, 3.0)], 1.5, 0.25),
    "flam-crack": lambda: patch.flam(ref("crack"), 25.0, 9.0, 1.0, 0.3),
    "flam-brush": lambda: patch.flam(ref("brush"), 30.0),
    "bursts": lambda: patch.bursts(ref("clap"), "noise[1]", patch.Train(4, 11.0, 2.0, 1.2, 0.2, 0.15, 0.9, 60.0)),
    "stack": lambda: patch.stack(ref("crack"), ref("short-clap"), 3.0),
    "stack-third": lambda: patch.stack(ref("crack"), ref("short-clap"), 1 / 3),
    "stack-rim": lambda: patch.stack(ref("brush"), ref("rim-click"), 3.0),
    "stack-brush": lambda: patch.stack(ref("rim-click"), ref("brush"), -2.0),
    "strokes-two-rims": lambda: patch.strokes(two_rims(), [(0.0, 0.0), (20.0, 4.0)]),
    "strokes-in-glide": lambda: patch.strokes(ref("zap"), [(0.0, 0.0), (10.0, 3.0), (20.0, 0.0)]),
    "stretch-rim": lambda: patch.stretch(ref("rimshot"), "rim[1].amp", 3.0),
    "stretch-body-long": lambda: patch.stretch(ref("long-snare"), "body.amp", 4.0),  # past the noise's 1 s
    "bursts-swell": lambda: patch.bursts(ref("clap"), "noise[1]", patch.Train(3, 15.0, taper=1.3, tail=0.4)),
    "softer": lambda: patch.softer(ref("rimshot"), 12.0),
    "tune": lambda: patch.tune(ref("crack"), "D3"),
    "stretch": lambda: patch.stretch(ref("long-snare"), "all", 1.5),
    **{f"take-{s}": functools.partial(takes_of, "crack", 3.0, 2.0, s) for s in (1, 2, 3, 4)},
    **{f"take-clap-{s}": functools.partial(takes_of, "clap", 2.0, 3.0, s) for s in (1, 2)},
}
PINNED = {
    "roll-build": "3bf0eaccf1445292f914ad128649fc61ee3b264e889f176c9068dc3ddb97e66e",
    "roll-rimshot": "f55486819d3be509e6ae77a69ddc6f7b61d163c30cd23961edb7a02906e8be97",
    "strokes": "eeb7827268870b44a919250acf75e4d40e3e537283972ffa7737b3db7dcfa6a1",
    "flam-crack": "7eec0c584f03e300710c18292dc34b1615451ccb6189c6d25e3931bd9eba3db0",
    "flam-brush": "88904ef5db9bb94675a5f6fbca8c4c573e6f91832f6716f00f878c25a29f7ae1",
    "bursts": "3de62ef54cd840264bbbc5ca0be239c19f18d8c92d71127fa174e51dc72d2198",
    "stack": "f4c9bfd09e0ac30cd650153be8cafe05ba5f2fc7e629e79c70fc581929292bff",
    "stack-third": "f9b64c9421624ac776108479292d74ad0a2922da40fe8e0f930643baa510c954",
    "stack-rim": "7dc6db9c0d85422a144cf4f1d2fd2d225a557e6dad93329bba9d5c3fc9344f31",
    "stack-brush": "c5f34389337530b4fa35f65b0a31b247d861817a11be830eec5ef29155e07f54",
    "strokes-two-rims": "4d578a653eeedfc6fd98f94da737266cbe336f4ca07100df85438699d690b4e5",
    "strokes-in-glide": "7117292294b26945f27ac6855feb896d2e6bddf2672e119a7e210b91d4aa18c7",
    "stretch-rim": "16a9450440c84a97891973ad454d757195172cbefd717a8dd69fe227b562cc9c",
    "stretch-body-long": "fd95200696991f6afb5d0a0a285da233b1ccdf3f89d9f2900dc9cafeccc8aca2",
    "bursts-swell": "826b8e748c90c3ddbcea1d35c39cbbda7bc82d4e406d248f2defeb63c5e130be",
    "softer": "cc282eba960ae3c8b4b42072aede8d4adadb31ce6cdc924a5bf126e6790342e7",
    "tune": "df064056a6142b8a10130e4b6261255d0bf05ae454bc4b3cc271fdc8aef8587a",
    "stretch": "1ffffd1da6e6fc4f9d253b871ac9613a0c0c1b14d03614be9784592cad510ba7",
    "take-1": "158ef483c6883dc84c5ee37b94a492c51d5253588a357e67f124f4b30efe8d07",
    "take-2": "46876759f06cc33e5a582592b3eaa50321150527befd127fa13218b41249a57d",
    "take-3": "da683f96e0cf0f4a7be4f1d884058f1a56880b1320641a1d9530a235a9be30b3",
    "take-4": "63ee5a0ddc99dfae04f2a958c298fbbfcd126256529d1172d212cd95385847e8",
    "take-clap-1": "8003ba712ce2d98fbf0491bd11ad21075d450289cb6d24dfe3eb786aa486e74a",
    "take-clap-2": "37d9a82758fc35d5879ad9f797fb45cee84466a575c8b87baeaaced1c8001eac",
}


@pytest.mark.parametrize("name", sorted(EDITS))
def test_every_edit_writes_the_patch_it_always_has(name: str) -> None:
    """Edit a preset, and find the canonical text of what the edit writes as pinned."""
    assert hashlib.sha256(patch.dumps(EDITS[name]()).encode()).hexdigest() == PINNED[name]


def test_every_pin_belongs_to_an_edit() -> None:
    """Find a pin for every edit, and no pin left from one removed."""
    assert sorted(PINNED) == sorted(EDITS)


@pytest.mark.parametrize(
    ("roll", "problem"),
    [
        (patch.Roll(1, 60.0), "--count: needs 2 to 16 strokes, found 1"),
        (patch.Roll(17, 60.0), "--count: needs 2 to 16 strokes, found 17"),
        (patch.Roll(4, 0.0), "--spacing: 0.0 is out of range"),
        (patch.Roll(4, 60.0, 0.0), "--growth: 0.0 is out of range"),
        (patch.Roll(4, 60.0, 1.0, -3.0), "--crescendo: -3.0 is out of range"),
    ],
)
def test_a_roll_names_the_option_out_of_range(roll: patch.Roll, problem: str) -> None:
    """Refuse a count past 2 … 16, a spacing or growth not above 0 and a crescendo below 0, saying which."""
    with pytest.raises(PatchError) as caught:
        patch.roll(patch.loads(EXAMPLE), roll)
    assert caught.value.problems == [problem]


def test_a_roll_takes_counts_from_2_to_16_and_no_crescendo() -> None:
    """Accept two strokes with no crescendo, and refuse 16 only for the knots a shape holds, not the count."""
    two = patch.roll(patch.loads(EXAMPLE), patch.Roll(2, 60.0, 1.0, 0.0))
    assert len(two.noise) == len(patch.loads(EXAMPLE).noise)
    with pytest.raises(PatchError) as caught:
        patch.roll(patch.loads(EXAMPLE), patch.Roll(16, 30.0))
    assert not any(m.startswith("--") for m in caught.value.problems)


def test_a_stroke_starts_at_least_10_us_after_the_last_struck() -> None:
    """Refuse a stroke at the moment the one before strikes, 0.05 ms in, and take one 10 µs later."""
    p = patch.loads(EXAMPLE)
    with pytest.raises(PatchError) as caught:
        patch.strokes(p, [(0.0, 0.0), (0.05, 0.0)])
    message = "stroke 2: starts 0.05 ms after stroke 1, which strikes 0.05 ms in; space the strokes further apart"
    assert message in caught.value.problems
    assert len(patch.strokes(p, [(0.0, 0.0), (0.06, 0.0)]).noise) == len(p.noise)


def test_a_roll_takes_a_spacing_under_a_millisecond() -> None:
    """Accept strokes half a millisecond apart: any spacing above 0 that the strokes can hold."""
    assert len(patch.roll(patch.loads(EXAMPLE), patch.Roll(2, 0.5)).noise) == len(patch.loads(EXAMPLE).noise)


@pytest.mark.parametrize(
    ("gap", "grace", "problem"),
    [
        (0.0, 12.0, "--gap: must be a positive number, found 0.0"),
        (25.0, 0.0, "--grace: must be a positive number, found 0.0"),
    ],
)
def test_a_flam_names_the_option_out_of_range(gap: float, grace: float, problem: str) -> None:
    """Refuse a gap or a grace that is not above 0, saying which."""
    with pytest.raises(PatchError) as caught:
        patch.flam(ref("crack"), gap, grace)
    assert caught.value.problems == [problem]


TAIL_LIMIT = 3 * (patch.SHOULDER_MS + float(patch.MIN_SPACING_MS))  # a tail must hold three shoulders


@pytest.mark.parametrize(
    ("train", "problem"),
    [
        (patch.Train(0, 11.0), "--count: needs 1 to 10 bursts, found 0"),
        (patch.Train(11, 11.0), "--count: needs 1 to 10 bursts, found 11"),
        (patch.Train(4, 11.0, tail=TAIL_LIMIT), f"--tail: {TAIL_LIMIT!r} is out of range"),
    ],
)
def test_a_train_names_the_option_out_of_range(train: patch.Train, problem: str) -> None:
    """Refuse a count past 1 … 10 and a tail no longer than three shoulders, saying which."""
    with pytest.raises(PatchError) as caught:
        patch.bursts(ref("clap"), "noise[1]", train)
    assert caught.value.problems == [problem]


def test_bursts_name_the_layers_a_patch_has() -> None:
    """Refuse a layer the patch lacks, listing every layer it has."""
    clap = ref("clap")
    with pytest.raises(PatchError) as caught:
        patch.bursts(clap, "noise[9]", patch.Train(4, 11.0))
    layers = ", ".join(f"noise[{j}]" for j in range(1, len(clap.noise) + 1))
    assert caught.value.problems == [f"noise[9]: not a layer of this patch; it has {layers}"]


def test_the_second_rims_curve_is_its_own() -> None:
    """Read the second rim's shape by its path, the rim click's in the stack."""
    assert patch.curve_knots(two_rims(), "rim[2].amp") == ref("rim-click").rim[0].amp


def test_a_note_just_flat_reads_as_plus_zero_cents() -> None:
    """Describe a pitch 0.04 cents flat of A4 as on it, with no minus sign before the zero."""
    assert tuning.describe_tuning(440.0 * 2 ** (-0.0004 / 12)) == "A4 +0.0 cents"
