# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Golden renders of the six references, two takes each (plan §10 regression contract).

Across machines and library versions the output agrees within floating-point
rounding, so the comparison uses a tolerance rather than a hash (plan §6).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from halyard_synth import wav
from halyard_synth.cli import REFERENCES, reference_text
from halyard_synth.patch import loads
from halyard_synth.render import render_takes

GOLDEN = Path(__file__).parent / "golden"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from golden import TAKES  # noqa: E402


@pytest.mark.parametrize("name", REFERENCES)
def test_golden(name):
    p = loads(reference_text(name))
    audio, report = render_takes(p, TAKES)
    for a, take in zip(audio, report.takes):
        golden = wav.read(GOLDEN / f"{name}_rr{take.number}.wav")
        assert loads(golden.meta["patch"]).seed == take.seed
        assert loads(golden.meta["patch"]) == loads(
            reference_text(name).replace(f"seed = {p.seed}", f"seed = {take.seed}", 1)
        )
        assert golden.rate == report.sample_rate
        assert a.size == golden.samples.size
        assert np.max(np.abs(a - golden.samples)) <= 1e-6
