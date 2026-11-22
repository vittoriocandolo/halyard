# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The cores a process may use: every one, or a pack worker's share."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from halyard_synth import cores


def test_a_process_uses_every_core_unless_it_has_a_share() -> None:
    """Use every core in a process of its own, and a pack worker's share of them in each worker."""
    everything = os.cpu_count() or 1
    assert cores.available() == everything
    for processes in (2, 4 * everything):
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(1, mp_context=context, initializer=cores.share, initargs=(processes,)) as pool:
            assert pool.submit(cores.available).result() == max(1, everything // processes)
    try:
        cores.share(3)
        assert cores.available() == max(1, everything // 3)
    finally:
        cores.share(1)  # the whole machine again, as a process of its own has
    assert cores.available() == everything
