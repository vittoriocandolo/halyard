# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Give the package the README with its relative links made absolute, so they work on PyPI too.

On the repository the README links to `docs/GUIDE.md` and shows
`docs/figures/crack.svg`; on PyPI those would lead nowhere. At build time
links point to the repository's page for the file, images to the file itself,
both on the branch the project's URLs name, taken from `[project.urls]`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, override

from hatchling.metadata.plugin.interface import MetadataHookInterface

LINK = re.compile(r"(?P<bang>!?)\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)\)")


def absolute(text: str, repository: str, branch: str) -> str:
    """Return markdown with every relative link made absolute: pages to the repository, images to the raw file."""
    owner_repo = repository.removeprefix("https://github.com/").rstrip("/")

    def link(m: re.Match[str]) -> str:
        target = m["target"]
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return m[0]
        if m["bang"]:
            url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{target}"
        else:
            url = f"https://github.com/{owner_repo}/blob/{branch}/{target}"
        return f"{m['bang']}[{m['text']}]({url})"

    return LINK.sub(link, text)


def branch_of(urls: dict[str, str]) -> str:
    """Return the branch the project's documentation URL names."""
    found = re.search(r"/(?:tree|blob)/([^/]+)/", urls["Documentation"])
    if found is None:
        msg = "[project.urls] Documentation must name a branch, as .../tree/<branch>/docs"
        raise ValueError(msg)
    return found[1]


class ReadmeHook(MetadataHookInterface):
    """Hatchling's metadata hook: the README, its links absolute."""

    PLUGIN_NAME = "custom"

    @override
    def update(self, metadata: dict[str, Any]) -> None:
        """Set the package's long description."""
        urls = metadata["urls"]
        text = (Path(self.root) / "README.md").read_text(encoding="utf-8")
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": absolute(text, urls["Repository"], branch_of(urls)),
        }
