# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the halyard command: python -m halyard_synth."""

import sys

from .cli import main

sys.exit(main())
