# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Golden renders of the six references, two takes each (plan §10 regression contract).

Across machines and library versions the output agrees within floating-point
rounding, so the comparison uses a tolerance rather than a hash (plan §6).
"""

from pathlib import Path

import numpy as np
import pytest

from halyard_synth import wav
from halyard_synth.cli import REFERENCES, reference_text
from halyard_synth.patch import loads
from halyard_synth.render import render_takes
from tools.golden import TAKES

GOLDEN = Path(__file__).parent / "golden"
ROUNDING = 1e-6  # float32 output, rounded on another platform or library version


@pytest.mark.parametrize("name", REFERENCES)
def test_golden(name: str) -> None:
    """Render each reference's two takes and find them as recorded, their patches and rates included."""
    p = loads(reference_text(name))
    audio, report = render_takes(p, TAKES)
    for a, take in zip(audio, report.takes, strict=True):
        golden = wav.read(GOLDEN / f"{name}_rr{take.number}.wav")
        assert golden.meta is not None
        assert loads(golden.meta["patch"]).seed == take.seed
        assert loads(golden.meta["patch"]) == loads(
            reference_text(name).replace(f"seed = {p.seed}", f"seed = {take.seed}", 1),
        )
        assert golden.rate == report.sample_rate
        assert a.size == golden.samples.size
        assert np.max(np.abs(a - golden.samples)) <= ROUNDING
