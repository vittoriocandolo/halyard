# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The frozen validated-domain rule against the recorded adversarial grid (docs/validation.md: The adversarial grid).

The grid itself takes a while (uv run python -m tools.grid run); this re-checks
the rule's constant against its committed results at the shipping K, so
changing the rule needs new evidence, and a fixture that has changed since its
row was recorded fails here until the grid is run again.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from halyard_synth.render import QUIET, SATURATION, K
from tools import grid, headroom

type Rows = list[dict[str, Any]]

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = {
    "body",
    "noise",
    "layers",
    "rattle",
    "clicks",
    "bursts",
    "snaps",
    "rims",
    "brushes",
    "takes",
    "dynamics",
    "seeds",
    "rates",
    "random",
    "longest",
}
FIXTURES = 900  # the grid's fixtures, at least
RANDOM = 200  # of them random, at least
SEEDS = 15  # the seeds family's seeds, 2 to 16
INSIDE = 500  # fixtures inside the domain, at least
INSIDE_DB = -105.0  # the worst error inside, with room under the −100 dB criterion
UNDRIVEN = 250  # undriven fixtures that test the kink bound, at least
DRIVEN = 250  # driven ones where the kinks dominate, at least
ROUNDING = 1e-15  # of the peak: an error the kink bound need not cover, rounding alone
DEEP_CHECKS = 80  # fixtures checked at 512 × base, at least
DEEP_FLOOR_DB = -200.0  # below which two references agree to rounding
EDGES = 300  # fixtures whose status changes within 0–48 dB of drive, at least
BISECTION_DB = 1e-4  # the brute force's resolution
CAUTIOUS = 3  # fixtures where the report places an edge that full renders never find, at most
DENSE = 100  # fixtures scanned every 0.05 dB, at least
DENSE_STEP_DB = 0.05


@pytest.fixture(scope="module")
def rows() -> Rows:
    """Return the grid's recorded results at the shipping K."""
    return grid.load(ROOT / "docs" / "grid" / f"k{K}.csv")


def _same(a: object, b: object) -> bool:
    """Return whether two fixtures' edits agree: numbers to 1e-12, everything else exactly.

    The random fixtures are drawn through powers of ten, which each platform's maths library may round differently in
    the last place, so the fixtures drawn on one platform differ from the recorded ones by that much on another.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12)
    return type(a) is type(b) and a == b


def test_the_grid_is_the_current_fixtures(rows: Rows) -> None:
    """Find every recorded row made from the fixture the grid makes today, to the last places of its numbers."""
    current = {(fam, name): edits for fam, name, edits in grid.fixtures()}
    recorded = {(r["family"], r["name"]): json.loads(r["edits"]) for r in rows}
    assert recorded.keys() == current.keys()
    assert [key for key in current if not _same(recorded[key], json.loads(json.dumps(current[key])))] == []


def test_the_grid_covers_the_corners(rows: Rows) -> None:
    """Find every family recorded, at least 900 fixtures, 200 random ones, and 15 more seeds."""
    assert {r["family"] for r in rows} >= FAMILIES
    assert len(rows) == len(grid.fixtures()) >= FIXTURES
    assert sum(r["family"] == "random" for r in rows) >= RANDOM
    assert len({r["seed"] for r in rows if r["family"] == "seeds"}) == SEEDS


def test_every_fixture_inside_the_validated_domain_passes(rows: Rows) -> None:
    """Find every fixture inside the domain passing, its reference converged, at −105 dB or better."""
    inside = [r for r in rows if r["inside"]]
    assert len(inside) > INSIDE
    assert all(r["pass"] and r["converged"] for r in inside)
    assert max(r["e_max"] for r in inside if r["peak"] >= QUIET) <= INSIDE_DB


def test_the_kink_bound_holds_without_drive(rows: Rows) -> None:
    """Find the kink bound above every undriven fixture's measured error."""
    linear = [r for r in rows if r["alpha"] == 0.0 and r["converged"]]
    assert len(linear) > UNDRIVEN
    assert all(r["max_abs"] <= r["kink"] or r["max_abs"] <= ROUNDING * r["peak"] for r in linear)


def test_the_kink_bound_holds_under_drive(rows: Rows) -> None:
    """Find the kink bound above every driven fixture's error, where the kinks dominate."""
    driven = [r for r in rows if r["alpha"] > 0.0 and r["converged"] and r["kink"] >= 2 * SATURATION * r["saturation"]]
    assert len(driven) > DRIVEN
    assert all(r["max_abs"] <= r["kink"] or r["max_abs"] <= ROUNDING * r["peak"] for r in driven)


def test_the_saturation_coefficient_has_margin(rows: Rows) -> None:
    """Refit the rule's constant to the grid, and find the frozen one covers it, 6 dB of margin included."""
    need, _ = grid.fit_saturation(rows)
    assert need <= SATURATION  # the fit already asks for 6 dB of margin over the worst fixture


def test_references_converge(rows: Rows) -> None:
    """Find every 512 × base check agreeing at least as well as the 128/256 × base pair."""
    checks = [r for r in rows if not math.isnan(r["c3_max"])]
    assert len(checks) >= DEEP_CHECKS
    assert all(r["c3_max"] <= r["c_max"] + 1e-9 or r["c3_max"] <= DEEP_FLOOR_DB for r in checks)


@pytest.fixture(scope="module")
def headroom_rows() -> Rows:
    """Return the recorded comparison of the reported drive headroom with full renders."""
    return headroom.load(ROOT / "docs" / "grid" / "headroom.csv")


def test_the_reported_drive_headroom_is_never_optimistic(headroom_rows: Rows) -> None:
    """Find the reported headroom never beyond the brute force's edge, and never claiming a way back in."""
    rows = headroom_rows
    assert len(rows) == len(grid.fixtures())
    edges = [r for r in rows if math.isfinite(r["reference"])]
    assert len(edges) > EDGES
    assert max(r["error"] for r in edges) <= BISECTION_DB
    # Where full renders never change status within 0–48 dB, the report says so too, or is cautious: it may
    # place an edge where the status in fact stays inside, but never claims a way back in that is not there.
    _, cautious, wrong = headroom.split_infinite([r for r in rows if not math.isfinite(r["reference"])])
    assert not wrong
    assert len(cautious) <= CAUTIOUS


def test_no_change_of_status_hides_between_the_candidates(headroom_rows: Rows) -> None:
    """Find no change of status between the report's candidates in any dense 0.05 dB scan."""
    dense = [r for r in headroom_rows if not math.isnan(r["dense"])]
    assert len(dense) >= DENSE
    assert all(abs(r["dense"] - r["reference"]) <= DENSE_STEP_DB + 1e-6 for r in dense)
