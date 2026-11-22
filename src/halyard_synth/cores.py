# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""How many threads a process may use: every core, or its share of them when a pack renders in parallel."""

import os

_SHARE: list[int] = []  # this process's share of the cores, once a pack has given it one


def available() -> int:
    """Return the threads this process may use."""
    return _SHARE[0] if _SHARE else os.cpu_count() or 1


def share(processes: int) -> None:
    """Take this process's share of the cores, as one of `processes` working at once."""
    _SHARE[:] = [max(1, (os.cpu_count() or 1) // processes)]
