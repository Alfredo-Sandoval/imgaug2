#!/usr/bin/env python3
"""Generate a Markdown report from benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ResultFile:
    path: Path
    platform: str
    label: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_results(results_dir: Path) -> list[ResultFile]:
    files: list[ResultFile] = []
    for path in sorted(results_dir.glob("*.json")):
        # Keep the report stable and avoid pulling in random local artifacts.
        # These are the canonical outputs produced by the benchmark entrypoints.
        allowed_prefixes = (
            "cpu_",
            "apple_",
            "nvidia_",
            "legacy_cpu_",
            "ops_",
            "third_party_",
        )
        if not path.name.startswith(allowed_prefixes):
            continue

        try:
            data = _load_json(path)
        except Exception:
            continue
        platform = str(data.get("platform") or "unknown")
        label = str(data.get("label") or path.stem)
        files.append(ResultFile(path=path, platform=platform, label=label))
    return files


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _render_summary_table(rows: list[tuple[str, str, str, float | None, str]]) -> str:
    lines = [
        "| Platform | Label | Benchmark | Throughput/sec | Unit |",
        "|----------|-------|-----------|----------------|------|",
    ]
    for platform, label, aug, throughput, unit in rows:
        ips_str = f"{throughput:.2f}" if throughput is not None else "n/a"
        lines.append(
            f"| {_md_escape(platform)} | {_md_escape(label)} | {_md_escape(aug)} | {ips_str} | {_md_escape(unit)} |"
        )
    return "\n".join(lines)


def generate_report(results_dir: Path, output_path: Path) -> None:
    result_files = _discover_results(results_dir)
    if not result_files:
        raise SystemExit(f"No readable JSON files found in {results_dir}")

    summary_rows: list[tuple[str, str, str, float | None, str]] = []
    for rf in result_files:
        data = _load_json(rf.path)
        benches: dict[str, Any] = data.get("benchmarks") or {}
        for aug_name, configs in benches.items():
            # Pick the most "standard" config if present; otherwise pick the first.
            cfg_key: str | None = None
            if isinstance(configs, dict):
                if "16x256x256x3" in configs:
                    cfg_key = "16x256x256x3"
                elif configs:
                    cfg_key = sorted(configs.keys())[0]
            if cfg_key is None:
                continue
            metrics = configs.get(cfg_key, {})
            throughput_val = metrics.get("throughput_per_sec", metrics.get("images_per_sec"))
            throughput: float | None = (
                float(throughput_val) if isinstance(throughput_val, (int, float)) else None
            )
            unit = str(metrics.get("throughput_unit") or "images")
            summary_rows.append((rf.platform, rf.label, f"{aug_name} ({cfg_key})", throughput, unit))

    md = "\n".join(
        [
            "# imgaug2 Benchmarks Report",
            "",
            f"Results directory: `{results_dir}`",
            "",
            "## Summary (throughput/sec)",
            "",
            _render_summary_table(summary_rows),
            "",
            "## Notes",
            "",
            "- MLX fast-paths run only when inputs are already `mlx.core.array` (B1: no implicit NumPy↔MLX transfers).",
            "- `benchmarks/ops.py` reports both `mlx_device` (kept on device) and `mlx_roundtrip` (np↔mlx per call).",
            "- The third-party baseline is single-image oriented; our runner loops over batches.",
            "- If you want per-config tables, extend `benchmarks/reports/generate_report.py` to emit config-specific sections.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/reports/out/benchmark_report.md"),
        help="Path to write the Markdown report to.",
    )
    args = parser.parse_args()
    generate_report(args.results_dir, args.output)
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
