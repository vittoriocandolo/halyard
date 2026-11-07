# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""What must never change: the names a file or a patch carries, and the streams every render draws from.

Changing any of these changes every render or orphans every file written so
far, so each is pinned here, and a change fails until it is made deliberately,
with a new key, a new schema or a new version (CONTRIBUTING.md).
"""

import tomllib
from pathlib import Path

from halyard_synth import cli, noise, patch, render, voice, wav

ROOT = Path(__file__).resolve().parent.parent


def test_the_names_a_patch_and_a_file_carry() -> None:
    """Find the instrument, the schema, the WAV chunk and the noise and dither keys as the first files had them."""
    assert patch.INSTRUMENT == "snare"
    assert patch.SCHEMA == 1
    assert wav.PRIVATE == b"snar"
    assert noise.KEY == b"snare/noise/v1"
    assert noise.DITHER_KEY == b"snare/dither/v1"


def test_the_package_command_and_references_keep_their_names() -> None:
    """Find the package, its command and every reference under the names people install, type and keep."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "halyard-synth"
    assert project["scripts"] == {"halyard": "halyard_synth.cli:main"}
    assert cli.REFERENCES[:6] == ("tone", "crack", "fat", "driven", "clap", "snap")
    for name in cli.REFERENCES:
        assert patch.loads(cli.reference_text(name)).seed == 1


def test_the_streams_every_render_draws_from() -> None:
    """Find the first noise and dither words of seed 1 as pinned: integer arithmetic, the same on every platform."""
    a, b = noise.uniforms(1)
    assert [int(x) for x in a[:3]] == [7722249428135383, 2233965327905589, 1986645927032010]
    assert [int(x) for x in b[:3]] == [7559645094285810, 779071378434091, 8356828306956370]
    d = noise.dither(1, 1)
    assert d[0] == (2955912619 - 2683841532) * 2.0**-32


FROZEN = {"components": 20000, "K": 32, "block": 2**16, "steps": 3}  # as the evidence in docs/ was made


def test_the_rendering_constants() -> None:
    """Find the noise's components, the internal rate's factor, the families, the block and the brush's steps."""
    found = {"components": noise.COMPONENTS, "K": render.K, "block": voice.BLOCK, "steps": voice.STEPS}
    assert found == FROZEN
    assert render.FAMILY == {44100: 44100, 88200: 44100, 176400: 44100, 48000: 48000, 96000: 48000, 192000: 48000}
