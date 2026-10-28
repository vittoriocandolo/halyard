# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The documentation tells the truth: generated parts current, links whole, examples as the code prints them."""

import re
import shlex
from pathlib import Path

import pytest

from docs import build
from halyard_synth import cli
from halyard_synth.patch import loads

from .conftest import EXAMPLE

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "CHANGELOG.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
QUICK_START = 10  # commands in the README's quick start, at least
D3_HZ = 146.832
REPORT = re.compile(r"```\n(tuning .*?)```", re.DOTALL)


def test_the_generated_docs_are_current(tmp_path: Path) -> None:
    """Build the figures, sounds and gallery afresh, and find every one as recorded in docs/."""
    stale = [
        path.relative_to(tmp_path)
        for path in build.build(tmp_path)
        if not build.current(path, build.DOCS / path.relative_to(tmp_path))
    ]
    assert not stale, f"run `uv run python -m docs.build`: {stale}"


def test_the_gallery_describes_every_reference_and_no_other() -> None:
    """Find the user guide's table of references naming exactly the shipped references, in their order."""
    assert list(build.descriptions()) == list(cli.REFERENCES)


def slug(heading: str) -> str:
    """Return a heading's anchor, as GitHub makes it."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(page: Path) -> set[str]:
    """Return a page's headings' anchors."""
    text = page.read_text(encoding="utf-8")
    return {slug(m[1]) for m in re.finditer(r"^#+ (.+)$", text, re.MULTILINE)}


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_link_leads_somewhere(page: Path) -> None:
    """Find every relative link of every page naming a file that exists, and every anchor a heading there."""
    for target in LINK.findall(page.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        found = (page.parent / path).resolve() if path else page
        assert found.exists(), f"{page.name}: {target}"
        if anchor:
            assert anchor in anchors(found), f"{page.name}: {target}"


def test_every_figure_and_sound_is_shown() -> None:
    """Find every generated figure and sound linked from some page, so none is left behind."""
    linked = {
        (page.parent / target.partition("#")[0]).resolve()
        for page in PAGES
        for target in LINK.findall(page.read_text(encoding="utf-8"))
    }
    for folder in (build.FIGURES, build.AUDIO):
        for path in folder.iterdir():
            assert path.resolve() in linked, f"{path.relative_to(ROOT)} is shown nowhere"
    assert {path.stem for path in build.AUDIO.iterdir()} == set(cli.REFERENCES)


def example() -> str:
    """Return the README's example patch."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    found = re.search(r"```toml\n(.*?)```", text, re.DOTALL)
    assert found is not None
    return found[1]


@pytest.mark.parametrize("page", [ROOT / "README.md", ROOT / "docs" / "GUIDE.md"], ids=["README", "GUIDE"])
def test_the_example_report_is_what_the_code_prints(page: Path) -> None:
    """Find the report shown in the README and the user guide the report of the README's patch, but its time."""
    shown = REPORT.findall(page.read_text(encoding="utf-8"))
    assert shown, page.name
    for block in shown:
        lines = [line for line in block.rstrip("\n").splitlines() if not line.startswith("render ")]
        assert "\n".join(lines) == build.report_text(loads(example()))


def test_the_readme_example_is_the_plan_example() -> None:
    """Find the README's patch the one the tests and the plan use."""
    assert loads(example()) == loads(EXAMPLE)


def quick_start() -> list[str]:
    """Return the README's quick start, one command a line, without the comments."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = text[text.index("## Use") :]
    start = block.index("```\n") + len("```\n")
    block = block[start : block.index("\n```", start)]
    return [line.split("  #")[0].strip() for line in block.splitlines()]


def test_the_quick_start_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every command of the README's quick start but audition, in order, each one succeeding."""
    monkeypatch.chdir(tmp_path)
    commands = quick_start()
    assert len(commands) >= QUICK_START
    assert all(c.startswith("halyard ") for c in commands)
    for command in commands:
        words = shlex.split(command)[1:]
        if words[0] == "audition":  # it needs a sound device and your loop
            continue
        out = None
        if ">" in words:
            words, out = words[: words.index(">")], words[words.index(">") + 1]
        assert cli.main(words) == 0, command
        printed = capsys.readouterr().out
        if out is not None:
            (tmp_path / out).write_text(printed, encoding="utf-8")
    assert loads((tmp_path / "snare.toml").read_text(encoding="utf-8")).body.tune_hz == D3_HZ  # as it says
    assert any((tmp_path / "kit").glob("*.sfz"))
