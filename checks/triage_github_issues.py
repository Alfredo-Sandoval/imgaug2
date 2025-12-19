#!/usr/bin/env python3
"""Bulk-triage GitHub issues for imgaug2.

Safe by default:
- Fetches open issues
- Computes proposed actions
- Writes a markdown report
- Does NOT modify issues unless flags are passed

Requires:
- GitHub CLI (gh)
- Authenticated `gh auth login`

Usage examples:

  # Dry-run report only
  python checks/triage_github_issues.py --config checks/issue_triage_config.json

  # Label issues according to config (still no close)
  python checks/triage_github_issues.py --config checks/issue_triage_config.json --apply-labels

  # Close explicitly-listed issues (and comment)
  python checks/triage_github_issues.py --config checks/issue_triage_config.json --close

  # Enable pattern rules too (more aggressive)
  python checks/triage_github_issues.py --config checks/issue_triage_config.json --enable-pattern-rules
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    url: str
    created_at: datetime


@dataclass(frozen=True)
class Decision:
    action: str  # close|review|keep|untriaged
    reason: str
    comment: str | None


def _run_gh(args: list[str]) -> str:
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh command failed: gh {' '.join(args)}\n{proc.stderr.strip()}")
    return proc.stdout


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_datetime(dt_str: str) -> datetime:
    # GitHub returns ISO 8601 like 2020-01-02T03:04:05Z
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_open_issues(repo: str, limit: int) -> list[Issue]:
    # GraphQL for stable pagination + minimal fields.
    query = """
    query($owner: String!, $name: String!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        issues(first: 100, after: $cursor, states: OPEN, orderBy: {field: CREATED_AT, direction: ASC}) {
          pageInfo { hasNextPage endCursor }
          nodes { number title url createdAt }
        }
      }
    }
    """.strip()

    owner, name = repo.split("/", 1)
    issues: list[Issue] = []
    cursor: str | None = None

    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
        ]
        if cursor is not None:
            args.extend(["-f", f"cursor={cursor}"])
        out = _run_gh(args)
        data = json.loads(out)
        nodes = data["data"]["repository"]["issues"]["nodes"]
        for node in nodes:
            issues.append(
                Issue(
                    number=int(node["number"]),
                    title=str(node["title"]),
                    url=str(node["url"]),
                    created_at=_parse_datetime(str(node["createdAt"])),
                )
            )
            if len(issues) >= limit:
                return issues
        page = data["data"]["repository"]["issues"]["pageInfo"]
        if not page["hasNextPage"]:
            return issues
        cursor = page["endCursor"]


def decide(
    issue: Issue,
    config: dict[str, Any],
    enable_pattern_rules: bool,
) -> Decision:
    explicit = config.get("explicit", {})
    close_set = set(explicit.get("close", []))
    review_set = set(explicit.get("review", []))
    keep_set = set(explicit.get("keep", []))

    duplicates: list[dict[str, Any]] = config.get("duplicates", [])
    for d in duplicates:
        if int(d.get("dupe")) == issue.number:
            canonical = int(d.get("canonical"))
            tmpl = str(config.get("comments", {}).get("duplicate", "Closing as duplicate of #{canonical}."))
            return Decision(
                action="close",
                reason=f"explicit-duplicate->{canonical}",
                comment=tmpl.format(canonical=canonical),
            )

    if issue.number in keep_set:
        return Decision(action="keep", reason="explicit-keep", comment=None)
    if issue.number in review_set:
        comment = str(config.get("comments", {}).get("review", "")) or None
        return Decision(action="review", reason="explicit-review", comment=comment)
    if issue.number in close_set:
        comment = str(config.get("comments", {}).get("close", "")) or None
        return Decision(action="close", reason="explicit-close", comment=comment)

    if not enable_pattern_rules:
        return Decision(action="untriaged", reason="no-rule", comment=None)

    rules = config.get("pattern_rules", {})
    how_to_re = re.compile(str(rules.get("how_to_title_regex", "")), re.IGNORECASE)
    install_re = re.compile(str(rules.get("install_title_regex", "")), re.IGNORECASE)
    feature_re = re.compile(str(rules.get("feature_request_title_regex", "")), re.IGNORECASE)
    old_cutoff_year = int(rules.get("old_install_cutoff_year", 2020))

    title = issue.title.strip()

    if how_to_re.search(title):
        comment = str(config.get("comments", {}).get("close", "")) or None
        return Decision(action="close", reason="pattern-how-to", comment=comment)

    if install_re.search(title) and issue.created_at.year < old_cutoff_year:
        comment = str(config.get("comments", {}).get("close", "")) or None
        return Decision(action="close", reason="pattern-old-install", comment=comment)

    if feature_re.search(title):
        # Don't auto-close feature requests without explicit list by default; mark review.
        comment = str(config.get("comments", {}).get("review", "")) or None
        return Decision(action="review", reason="pattern-feature-request", comment=comment)

    return Decision(action="untriaged", reason="no-match", comment=None)


def ensure_label(repo: str, label: str) -> None:
    # Create label if missing. `gh label create` exits non-zero if it exists.
    proc = subprocess.run(
        ["gh", "label", "create", label, "--repo", repo, "--color", "0E8A16", "--description", "Automated triage label"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return
    if "already exists" in (proc.stderr or "").lower():
        return


def label_issue(repo: str, number: int, label: str) -> None:
    _run_gh(["issue", "edit", str(number), "--repo", repo, "--add-label", label])


def comment_issue(repo: str, number: int, body: str) -> None:
    _run_gh(["issue", "comment", str(number), "--repo", repo, "--body", body])


def close_issue(repo: str, number: int, reason: str) -> None:
    # Allowed reasons: completed | not planned
    close_reason = "not planned" if reason != "completed" else "completed"
    _run_gh(["issue", "close", str(number), "--repo", repo, "--reason", close_reason])


def write_report(path: Path, repo: str, decisions: list[tuple[Issue, Decision]]) -> None:
    counts: dict[str, int] = {}
    for _, d in decisions:
        counts[d.action] = counts.get(d.action, 0) + 1

    lines: list[str] = []
    lines.append(f"# Issue triage report: {repo}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in ["close", "review", "keep", "untriaged"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")

    def section(action: str) -> None:
        lines.append(f"## {action.capitalize()}")
        lines.append("")
        for issue, d in decisions:
            if d.action != action:
                continue
            lines.append(f"- #{issue.number} - {issue.title} ({d.reason})")
            lines.append(f"  - {issue.url}")
        lines.append("")

    for action in ["close", "review", "keep", "untriaged"]:
        section(action)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to triage config JSON")
    parser.add_argument("--repo", default=None, help="Override repo owner/name (e.g. Alfredo-Sandoval/imgaug2)")
    parser.add_argument("--limit", type=int, default=2000, help="Max open issues to fetch")
    parser.add_argument("--report", default=None, help="Report output path (markdown)")

    parser.add_argument("--enable-pattern-rules", action="store_true", help="Enable heuristic triage rules")

    parser.add_argument("--apply-labels", action="store_true", help="Apply triage labels to issues")
    parser.add_argument("--close", action="store_true", help="Close issues marked close")
    parser.add_argument("--comment", action="store_true", help="Leave comments (for review/close) when applying")

    parser.add_argument("--sleep-secs", type=float, default=0.2, help="Delay between write operations")

    args = parser.parse_args()

    config_path = Path(args.config)
    config = _read_json(config_path)
    repo = str(args.repo or config.get("repo") or os.environ.get("GITHUB_REPO") or "").strip()
    if not repo or "/" not in repo:
        raise SystemExit("Repo not set. Use --repo or set config.repo")

    pattern_enabled = bool(args.enable_pattern_rules)
    if not args.enable_pattern_rules:
        # respect config default
        rules = config.get("pattern_rules", {})
        if bool(rules.get("enabled_by_default", False)):
            pattern_enabled = True

    issues = fetch_open_issues(repo=repo, limit=args.limit)
    decisions: list[tuple[Issue, Decision]] = [(iss, decide(iss, config, pattern_enabled)) for iss in issues]

    report_path = Path(args.report) if args.report else Path("checks/reports") / f"issue-triage-{repo.replace('/', '_')}.md"
    write_report(report_path, repo=repo, decisions=decisions)
    print(f"Wrote report: {report_path}")

    if not (args.apply_labels or args.close or args.comment):
        print("Dry-run only (no labels/comments/close applied).")
        return 0

    labels_cfg: dict[str, str] = config.get("labels", {})
    for label in labels_cfg.values():
        ensure_label(repo, label)

    for issue, d in decisions:
        if d.action not in {"close", "review", "keep", "untriaged"}:
            continue

        target_label = labels_cfg.get(d.action) or labels_cfg.get("untriaged")

        if args.apply_labels and target_label:
            label_issue(repo, issue.number, target_label)
            time.sleep(args.sleep_secs)

        if args.comment and d.comment:
            comment_issue(repo, issue.number, d.comment)
            time.sleep(args.sleep_secs)

        if args.close and d.action == "close":
            close_issue(repo, issue.number, reason="not planned")
            time.sleep(args.sleep_secs)

    print("Applied requested changes.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
