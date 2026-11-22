# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tools that design the frozen filters, run the grid and measure the presets; run each with python -m."""

import sys


def say(text: str) -> None:
    """Write a line of a tool's report."""
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()  # a long run's progress shows as it happens, in a log too
