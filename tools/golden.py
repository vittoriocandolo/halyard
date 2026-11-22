# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write the golden renders that tests/test_golden.py compares against (docs/design.md: Reproducibility).

Golden files begin once the rendering policy is frozen (milestone 2). Only
regenerate them for a deliberate change to the sound, and say so in the commit.

    uv run python -m tools.golden
"""

from __future__ import annotations

import sys
from pathlib import Path

from halyard_synth.cli import PRESETS, preset_text
from halyard_synth.patch import loads
from halyard_synth.render import export, render_takes
from tools import say

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden"
TAKES = 2  # two takes each, so the seeds of a set are pinned too


def main() -> int:
    """Render and write every preset's golden takes."""
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name in PRESETS:
        p = loads(preset_text(name))
        audio, report = render_takes(p, TAKES)
        for a, take in zip(audio, report.takes, strict=True):
            export(GOLDEN / f"{name}_rr{take.number}.wav", a, p, report, take)
        say(f"{name}: {audio[0].size} samples × {TAKES} takes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
