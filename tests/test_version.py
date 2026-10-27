# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The version: declared once for the package, and the same everywhere it is said."""

from importlib.metadata import version

import halyard_synth


def test_the_version_is_declared_once() -> None:
    """Find the installed package's version the one the code says."""
    assert halyard_synth.__version__ == version("halyard-synth")
