#!/usr/bin/env python3
"""Validate that mkdocs.yml nav entries point to existing Markdown files.

This is a lightweight check that avoids requiring MkDocs/PyYAML at runtime.
It parses the `nav:` section heuristically and verifies that referenced `.md`
files exist under `docs/`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_NAV_MD_RE = re.compile(r"^\s*-\s+[^:]+:\s*(?P<path>[^#]+?\.md)\s*(?:#.*)?$")


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        return text[1:-1]
    return text


def _extract_nav_md_paths(mkdocs_yml: Path) -> list[str]:
    lines = mkdocs_yml.read_text(encoding="utf-8").splitlines()

    in_nav = False
    results: list[str] = []
    for line in lines:
        if line.strip() == "nav:":
            in_nav = True
            continue
        if not in_nav:
            continue

        match = _NAV_MD_RE.match(line)
        if match is None:
            continue
        path = _strip_quotes(match.group("path"))
        results.append(path)

    return results


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    mkdocs_yml = repo_root / "mkdocs.yml"
    docs_root = repo_root / "docs"

    if not mkdocs_yml.exists():
        print("ERROR: mkdocs.yml not found", file=sys.stderr)
        return 2
    if not docs_root.exists():
        print("ERROR: docs/ not found", file=sys.stderr)
        return 2

    nav_paths = _extract_nav_md_paths(mkdocs_yml)
    if not nav_paths:
        print("ERROR: No .md paths found in mkdocs.yml nav", file=sys.stderr)
        return 2

    missing: list[str] = []
    for rel in nav_paths:
        rel_path = Path(rel)
        # MkDocs nav paths are relative to docs root.
        abs_path = docs_root / rel_path
        if not abs_path.exists():
            missing.append(str(rel_path))

    if missing:
        print("ERROR: Missing docs pages referenced in mkdocs.yml nav:", file=sys.stderr)
        for rel in missing:
            print(f"- {rel}", file=sys.stderr)
        return 1

    print(f"OK: mkdocs nav references {len(nav_paths)} markdown files and all exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
