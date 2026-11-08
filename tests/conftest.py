# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The tests' shared patch: the plan's example, and ways to edit it by path."""

import copy
import re
import tomllib
from typing import Any

import pytest
from hypothesis import settings

from halyard_synth.patch import Patch, from_dict

# The illustrative patch from the plan (§4).
# Fuzzing derives its examples the same way on every run and keeps none between runs, so a run is reproducible
# from the code alone; a deadline would make the outcome depend on the machine.
settings.register_profile("reproducible", derandomize=True, database=None, deadline=None)
settings.load_profile("reproducible")

EXAMPLE = """\
instrument = "snare"
schema = 1
seed = 1

[body]
tune_hz  = 185.0
level_db = -4.0
damping  = 1.0              # mode i decays ratio^damping times faster
modes = [                   # loudest mode exactly 0 dB
  { ratio = 1.0,   level_db = 0.0 },
  { ratio = 1.593, level_db = -5.0 },
  { ratio = 2.136, level_db = -11.0 },
]
pitch = [                   # ratio to tune_hz; starts at 0 ms, ends at 1
  { ms = 0.0,  ratio = 1.08 },
  { ms = 15.0, ratio = 1.0 },
]
amp = [                     # shape: 0 at both ends, highest point exactly 1
  { ms = 0.0,   gain = 0.0 },
  { ms = 0.4,   gain = 1.0 },
  { ms = 25.0,  gain = 0.45 },
  { ms = 90.0,  gain = 0.1 },
  { ms = 180.0, gain = 0.0 },
]

[[noise]]                   # noise[1], the stick: short and wide
level_db = -10.0
band_hz  = [1500.0, 16000.0]
rattle   = 0.0
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.05, gain = 1.0 },
  { ms = 0.15, gain = 0.7 },  # a shoulder knot keeps the crack sharp
  { ms = 3.0,  gain = 0.1 },
  { ms = 8.0,  gain = 0.0 },
]

[[noise]]                   # noise[2], the wires
level_db = -6.0
band_hz  = [2500.0, 9000.0]
rattle   = 0.5              # 0 … 1, gated at the body's fundamental
amp = [
  { ms = 0.0,   gain = 0.0 },
  { ms = 1.5,   gain = 1.0 },
  { ms = 40.0,  gain = 0.4 },
  { ms = 140.0, gain = 0.08 },
  { ms = 260.0, gain = 0.0 },
]

[drive]
db = 6.0                    # small-signal gain; 0 = bypass

[output]
sample_rate = 48000
format      = "float32"     # "float32" | "pcm24" | "pcm16"
mode        = "gain"        # "gain" | "peak"
db          = -6.0          # gain in dB, or peak target in dBFS
polarity    = 1
"""


@pytest.fixture
def example_dict() -> dict[str, Any]:
    """Return the example patch's table, a fresh copy for each test."""
    return copy.deepcopy(tomllib.loads(EXAMPLE))


def edit(d: dict[str, Any], path: str, value: object) -> dict[str, Any]:
    """Set a field by its patch path, counting from 1, such as "noise[2].amp[3].ms", and return the table."""
    node: Any = d
    parts = path.split(".")
    for i, part in enumerate(parts):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        assert m is not None, path
        name, index = m.group(1), m.group(2)
        last = i == len(parts) - 1
        if index is None:
            if last:
                node[name] = value
            else:
                node = node[name]
        elif last:
            node[name][int(index) - 1] = value
        else:
            node = node[name][int(index) - 1]
    return d


def make(**edits: object) -> Patch:
    """Return the example patch with edits by path, such as **{"noise[2].rattle": 0.3}."""
    d = tomllib.loads(EXAMPLE)
    for path, value in edits.items():
        edit(d, path, value)
    return from_dict(d)
