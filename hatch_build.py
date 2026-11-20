# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Give the package the README with its relative links made absolute, so they work on PyPI too.

On the repository the README links to `docs/GUIDE.md` and shows
`docs/figures/crack.svg`; on PyPI those would lead nowhere. At build time
links point to the repository's page for the file, images to the file itself,
both at the release's tag, v<version>, so a release's page on PyPI shows
that release's documentation even after later ones change it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, override

from hatchling.metadata.plugin.interface import MetadataHookInterface

LINK = re.compile(r"(?P<bang>!?)\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)\)")


def absolute(text: str, repository: str, ref: str) -> str:
    """Return markdown with every relative link made absolute: pages to the repository, images to the raw file."""
    owner_repo = repository.removeprefix("https://github.com/").rstrip("/")

    def link(m: re.Match[str]) -> str:
        target = m["target"]
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return m[0]
        if m["bang"]:
            url = f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{target}"
        else:
            url = f"https://github.com/{owner_repo}/blob/{ref}/{target}"
        return f"{m['bang']}[{m['text']}]({url})"

    return LINK.sub(link, text)


class ReadmeHook(MetadataHookInterface):
    """Hatchling's metadata hook: the README, its links absolute."""

    PLUGIN_NAME = "custom"

    @override
    def update(self, metadata: dict[str, Any]) -> None:
        """Set the package's long description, its links at the tag v<version>."""
        text = (Path(self.root) / "README.md").read_text(encoding="utf-8")
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": absolute(text, metadata["urls"]["Repository"], f"v{metadata['version']}"),
        }
