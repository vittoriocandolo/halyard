"""Write the golden renders that tests/test_golden.py compares against (plan §10).

Golden files begin once the rendering policy is frozen (milestone 2). Only
regenerate them for a deliberate change to the sound, and say so in the commit.

    python tools/golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from snare.cli import REFERENCES, reference_text
from snare.patch import loads
from snare.render import export, render_takes

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden"
TAKES = 2  # two takes each, so the seeds of a set are pinned too


def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name in REFERENCES:
        p = loads(reference_text(name))
        audio, report = render_takes(p, TAKES)
        for a, take in zip(audio, report.takes):
            export(GOLDEN / f"{name}_rr{take.number}.wav", a, p, report, take)
        print(f"{name}: {audio[0].size} samples × {TAKES} takes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
