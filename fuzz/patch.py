# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fuzz the patch reader: any bytes are read as a patch file or refused with the reason.

The file is read as the command line reads one, and a patch read comes back unchanged from its canonical text.

    uv run --group fuzz python fuzz/patch.py [-max_total_time=SECONDS]

Linux on x86-64 only, where Atheris has wheels. The shipped presets seed the corpus.
"""

import sys
import tempfile
from importlib import resources
from pathlib import Path

import atheris

with atheris.instrument_imports():
    from halyard_synth.patch import PatchError, dumps, load, loads

FILE = Path(tempfile.mkdtemp()) / "patch.toml"


def test_one(data: bytes) -> None:
    """Read one input as a patch file."""
    FILE.write_bytes(data)
    try:
        patch = load(FILE)
    except PatchError:  # refused, with the reason the command line prints
        return
    assert loads(dumps(patch)) == patch


def main() -> None:
    """Fuzz from the presets, with libFuzzer's own options from the command line."""
    corpus = Path(tempfile.mkdtemp())
    for preset in resources.files("halyard_synth").joinpath("data", "presets").iterdir():
        if preset.name.endswith(".toml"):
            (corpus / preset.name).write_bytes(preset.read_bytes())
    atheris.Setup([*sys.argv, str(corpus)], test_one)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
