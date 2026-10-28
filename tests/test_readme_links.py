# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The package's README keeps its links on PyPI: hatch_build.py makes them absolute."""

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def hook_module() -> ModuleType:
    """Load hatch_build.py, which the build backend loads by path."""
    spec = importlib.util.spec_from_file_location("hatch_build", ROOT / "hatch_build.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_links_lead_to_the_repository() -> None:
    """Make pages link to the repository's view and images to the raw file, leaving absolute links and anchors."""
    hb = hook_module()
    text = "[guide](docs/GUIDE.md) ![fig](docs/figures/a.svg) [web](https://example.org) [here](#use)"
    assert hb.absolute(text, "https://github.com/someone/halyard", "master") == (
        "[guide](https://github.com/someone/halyard/blob/master/docs/GUIDE.md) "
        "![fig](https://raw.githubusercontent.com/someone/halyard/master/docs/figures/a.svg) "
        "[web](https://example.org) [here](#use)"
    )
    assert hb.branch_of({"Documentation": "https://github.com/someone/halyard/tree/main/docs"}) == "main"
    with pytest.raises(ValueError, match="must name a branch"):
        hb.branch_of({"Documentation": "https://someone.example/docs"})


def test_the_package_readme_has_no_relative_link_left() -> None:
    """Build the long description as the backend does, and find every link absolute, each to a file that exists."""
    hb = hook_module()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata: dict[str, Any] = {"urls": project["urls"]}
    hb.ReadmeHook(str(ROOT), {}).update(metadata)
    assert metadata["readme"]["content-type"] == "text/markdown"
    branch = hb.branch_of(project["urls"])
    for target in re.findall(r"\]\(([^)\s]+)\)", metadata["readme"]["text"]):
        assert target.startswith(("https://", "#")), target
        for prefix in (f"/blob/{branch}/", f"/{branch}/"):
            if target.startswith("https://") and prefix in target and "github" in target:
                assert (ROOT / target.split(prefix, 1)[1].partition("#")[0]).exists(), target
                break
