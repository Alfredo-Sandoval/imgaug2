#!/usr/bin/env python3
"""Check that local asset links in docs resolve to existing files.

This is a lightweight sanity check to avoid broken images in MkDocs output.
It intentionally does not attempt to validate external URLs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".pdf"}


def _is_external(link: str) -> bool:
    return link.startswith(("http://", "https://", "mailto:", "data:"))


def _strip_suffixes(link: str) -> str:
    # Drop anchors and query strings.
    link = link.split("#", 1)[0]
    link = link.split("?", 1)[0]
    return link


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    if not docs_dir.exists():
        print(f"ERROR: docs directory not found at {docs_dir}", file=sys.stderr)
        return 2

    missing: list[tuple[Path, str]] = []
    md_files = sorted(docs_dir.rglob("*.md"))

    # Capture markdown image links: ![alt](path)
    image_link_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for raw_link in image_link_re.findall(text):
            link = _strip_suffixes(raw_link.strip())
            if not link or link.startswith("#") or _is_external(link):
                continue

            if Path(link).suffix.lower() not in _ASSET_EXTS:
                continue

            target = (md_file.parent / link).resolve()
            if not target.exists():
                missing.append((md_file, raw_link))

    if missing:
        print("ERROR: Missing local assets referenced in docs:")
        for md_file, raw_link in missing:
            rel = md_file.relative_to(repo_root)
            print(f"- {rel}: {raw_link}")
        return 1

    print(f"OK: Checked {len(md_files)} markdown files; all local image assets exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
