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
from collections.abc import Callable

import pytest

from halyard_synth import cli, patch
from halyard_synth.patch import Patch, PatchError

from .conftest import EXAMPLE


def ref(name: str) -> Patch:
    """Return a reference."""
    return patch.loads(cli.reference_text(name))


def takes_of(name: str, strength_db: float, timing_ms: float, seed: int) -> Patch:
    """Return one take of a reference given takes that differ as played."""
    return patch.played(dataclasses.replace(ref(name), seed=seed, takes=patch.Takes(strength_db, timing_ms)))


EDITS: dict[str, Callable[[], Patch]] = {
    "roll-build": lambda: patch.roll(patch.loads(EXAMPLE), patch.Roll(3, 60.0, 0.8, 6.0)),
    "roll-rimshot": lambda: patch.roll(ref("rimshot"), patch.Roll(2, 40.0), glide=1.0, contact=0.3),
    "strokes": lambda: patch.strokes(patch.loads(EXAMPLE), [(0.0, 6.0), (35.0, 0.0), (70.0, 3.0)], 1.5, 0.25),
    "flam-crack": lambda: patch.flam(ref("crack"), 25.0, 9.0, 1.0, 0.3),
    "flam-brush": lambda: patch.flam(ref("brush"), 30.0),
    "bursts": lambda: patch.bursts(ref("clap"), "noise[1]", patch.Train(4, 11.0, 2.0, 1.2, 0.2, 0.15, 0.9, 60.0)),
    "stack": lambda: patch.stack(ref("crack"), ref("short-clap"), 3.0),
    "softer": lambda: patch.softer(ref("rimshot"), 12.0),
    "tune": lambda: patch.tune(ref("crack"), "D3"),
    "stretch": lambda: patch.stretch(ref("long-snare"), "all", 1.5),
    **{f"take-{s}": functools.partial(takes_of, "crack", 3.0, 2.0, s) for s in (1, 2, 3, 4)},
    **{f"take-clap-{s}": functools.partial(takes_of, "clap", 2.0, 3.0, s) for s in (1, 2)},
}
PINNED = {
    "roll-build": "3bf0eaccf1445292f914ad128649fc61ee3b264e889f176c9068dc3ddb97e66e",
    "roll-rimshot": "2ac20906aecece4e374d4e3f65444ca3380f7228a6d321644e82334e818cb1ad",
    "strokes": "eeb7827268870b44a919250acf75e4d40e3e537283972ffa7737b3db7dcfa6a1",
    "flam-crack": "7eec0c584f03e300710c18292dc34b1615451ccb6189c6d25e3931bd9eba3db0",
    "flam-brush": "88904ef5db9bb94675a5f6fbca8c4c573e6f91832f6716f00f878c25a29f7ae1",
    "bursts": "3de62ef54cd840264bbbc5ca0be239c19f18d8c92d71127fa174e51dc72d2198",
    "stack": "f4c9bfd09e0ac30cd650153be8cafe05ba5f2fc7e629e79c70fc581929292bff",
    "softer": "482e84e1e2c1bfb49aa03b257ec2856b0f9d05659b764952410156117953c489",
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
    """Edit a reference, and find the canonical text of what the edit writes as pinned."""
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
