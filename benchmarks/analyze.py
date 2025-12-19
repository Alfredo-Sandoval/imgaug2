#!/usr/bin/env python3
"""Benchmark analysis tool for CPU vs MLX comparisons.

This module provides utilities for:
1. Parsing benchmark JSON files
2. Detecting break-even points between CPU and MLX
3. Flagging outliers and anomalies
4. Generating routing recommendations

Usage:
    python -m benchmarks.analyze benchmarks/results/ops_*.json
    python -m benchmarks.analyze --markdown benchmarks/results/ops_*.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Single benchmark result for an op/config combination."""

    op: str
    backend: str  # "cpu_cv2", "mlx_device", "mlx_roundtrip"
    config: str  # e.g., "16x256x256x3"
    batch: int
    height: int
    width: int
    channels: int
    images_per_sec: float
    avg_time_s: float
    std_time_s: float
    p95_time_s: float
    raw: dict[str, Any] = field(repr=False)

    @property
    def total_pixels(self) -> int:
        return self.batch * self.height * self.width

    @property
    def pixels_per_image(self) -> int:
        return self.height * self.width


@dataclass
class ComparisonResult:
    """CPU vs MLX comparison for a single op/config."""

    op: str
    config: str
    batch: int
    height: int
    width: int
    cpu_ips: float
    mlx_device_ips: float | None
    mlx_roundtrip_ips: float | None
    speedup_device: float | None  # MLX_device / CPU, >1 means MLX wins
    speedup_roundtrip: float | None

    @property
    def mlx_wins_device(self) -> bool:
        return self.speedup_device is not None and self.speedup_device > 1.0

    @property
    def mlx_wins_roundtrip(self) -> bool:
        return self.speedup_roundtrip is not None and self.speedup_roundtrip > 1.0


@dataclass
class BreakEvenPoint:
    """Describes where MLX becomes faster than CPU for an op."""

    op: str
    mode: str  # "device" or "roundtrip"
    min_batch_for_size: dict[tuple[int, int], int]  # (H, W) -> min batch
    min_pixels_batch1: int | None  # H*W at batch=1
    recommendation: str


@dataclass
class Outlier:
    """An anomalous data point that doesn't fit expected patterns."""

    op: str
    config: str
    description: str
    expected_speedup: float | None
    actual_speedup: float | None
    severity: str  # "info", "warning", "error"


def parse_config(config_str: str) -> tuple[int, int, int, int]:
    """Parse config string like '16x256x256x3' into (batch, h, w, c)."""
    parts = config_str.split("x")
    if len(parts) != 4:
        raise ValueError(f"Invalid config: {config_str}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def parse_benchmark_file(path: Path) -> list[BenchmarkResult]:
    """Parse a benchmark JSON file into structured results."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    benchmarks = data.get("benchmarks", {})

    for op_key, configs in benchmarks.items():
        # op_key is like "gaussian_blur/cpu_cv2" or "affine_transform/mlx_device"
        if "/" not in op_key:
            continue

        op_parts = op_key.rsplit("/", 1)
        op = op_parts[0]
        backend = op_parts[1]

        for config_str, metrics in configs.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                try:
                    b, h, w, c = parse_config(config_str)
                    results.append(
                        BenchmarkResult(
                            op=op,
                            backend=backend,
                            config=config_str,
                            batch=b,
                            height=h,
                            width=w,
                            channels=c,
                            images_per_sec=float(metrics.get("images_per_sec", 0)),
                            avg_time_s=float(metrics.get("avg_time_s", 0)),
                            std_time_s=float(metrics.get("std_time_s", 0)),
                            p95_time_s=float(metrics.get("p95_time_s", 0)),
                            raw=metrics,
                        )
                    )
                except (ValueError, KeyError):
                    continue

    return results


def compare_backends(results: list[BenchmarkResult]) -> list[ComparisonResult]:
    """Compare CPU vs MLX performance for each op/config."""
    # Group by (op, config)
    grouped: dict[tuple[str, str], dict[str, BenchmarkResult]] = defaultdict(dict)

    for r in results:
        grouped[(r.op, r.config)][r.backend] = r

    comparisons = []
    for (op, config), backends in grouped.items():
        cpu_result = backends.get("cpu_cv2")
        mlx_device = backends.get("mlx_device")
        mlx_roundtrip = backends.get("mlx_roundtrip")

        if cpu_result is None:
            continue

        b, h, w, c = parse_config(config)

        speedup_device = None
        speedup_roundtrip = None

        if mlx_device and cpu_result.images_per_sec > 0:
            speedup_device = mlx_device.images_per_sec / cpu_result.images_per_sec

        if mlx_roundtrip and cpu_result.images_per_sec > 0:
            speedup_roundtrip = mlx_roundtrip.images_per_sec / cpu_result.images_per_sec

        comparisons.append(
            ComparisonResult(
                op=op,
                config=config,
                batch=b,
                height=h,
                width=w,
                cpu_ips=cpu_result.images_per_sec,
                mlx_device_ips=mlx_device.images_per_sec if mlx_device else None,
                mlx_roundtrip_ips=mlx_roundtrip.images_per_sec if mlx_roundtrip else None,
                speedup_device=speedup_device,
                speedup_roundtrip=speedup_roundtrip,
            )
        )

    return comparisons


def find_break_even_points(comparisons: list[ComparisonResult]) -> list[BreakEvenPoint]:
    """Find the break-even points where MLX becomes faster than CPU."""
    # Group by op
    by_op: dict[str, list[ComparisonResult]] = defaultdict(list)
    for c in comparisons:
        by_op[c.op].append(c)

    break_evens = []

    for op, comps in by_op.items():
        for mode in ["device", "roundtrip"]:
            speedup_attr = f"speedup_{mode}"
            min_batch_for_size: dict[tuple[int, int], int] = {}
            min_pixels_batch1: int | None = None

            # Group by (H, W)
            by_size: dict[tuple[int, int], list[ComparisonResult]] = defaultdict(list)
            for c in comps:
                by_size[(c.height, c.width)].append(c)

            for (h, w), size_comps in by_size.items():
                # Sort by batch
                size_comps_sorted = sorted(size_comps, key=lambda x: x.batch)

                for c in size_comps_sorted:
                    speedup = getattr(c, speedup_attr)
                    if speedup is not None and speedup >= 1.0:
                        if (h, w) not in min_batch_for_size:
                            min_batch_for_size[(h, w)] = c.batch
                        if c.batch == 1 and min_pixels_batch1 is None:
                            min_pixels_batch1 = h * w
                        break

            if not min_batch_for_size:
                continue

            # Generate recommendation
            rec_parts = []
            sizes_at_batch1 = [
                (h, w) for (h, w), b in min_batch_for_size.items() if b == 1
            ]
            sizes_at_batch2_4 = [
                (h, w) for (h, w), b in min_batch_for_size.items() if 2 <= b <= 4
            ]
            sizes_at_batch8_plus = [
                (h, w) for (h, w), b in min_batch_for_size.items() if b >= 8
            ]

            if sizes_at_batch1:
                min_hw = min(h * w for h, w in sizes_at_batch1)
                rec_parts.append(f"Use MLX at batch>=1 when H*W>={min_hw:,}")

            if sizes_at_batch2_4:
                min_hw = min(h * w for h, w in sizes_at_batch2_4)
                if not sizes_at_batch1 or min_hw < min(
                    h * w for h, w in sizes_at_batch1
                ):
                    rec_parts.append(f"Use MLX at batch>=2 when H*W>={min_hw:,}")

            if sizes_at_batch8_plus:
                min_hw = min(h * w for h, w in sizes_at_batch8_plus)
                rec_parts.append(f"Use MLX at batch>=8 when H*W>={min_hw:,}")

            recommendation = "; ".join(rec_parts) if rec_parts else "No clear break-even"

            break_evens.append(
                BreakEvenPoint(
                    op=op,
                    mode=mode,
                    min_batch_for_size=min_batch_for_size,
                    min_pixels_batch1=min_pixels_batch1,
                    recommendation=recommendation,
                )
            )

    return break_evens


def detect_outliers(comparisons: list[ComparisonResult]) -> list[Outlier]:
    """Detect anomalous data points that don't fit expected patterns."""
    outliers = []

    # Group by op
    by_op: dict[str, list[ComparisonResult]] = defaultdict(list)
    for c in comparisons:
        by_op[c.op].append(c)

    for op, comps in by_op.items():
        # Sort by total pixels
        sorted_comps = sorted(comps, key=lambda x: (x.height * x.width, x.batch))

        prev_speedups: list[float] = []
        for c in sorted_comps:
            if c.speedup_device is None:
                continue

            # Check for non-monotonic speedup (larger should be faster for MLX)
            if prev_speedups:
                avg_prev = sum(prev_speedups) / len(prev_speedups)

                # If current is significantly worse than smaller sizes
                if c.speedup_device < avg_prev * 0.5 and avg_prev > 0.5:
                    outliers.append(
                        Outlier(
                            op=op,
                            config=c.config,
                            description=(
                                f"MLX speedup dropped unexpectedly: {c.speedup_device:.2f}x "
                                f"(expected ~{avg_prev:.2f}x based on smaller sizes)"
                            ),
                            expected_speedup=avg_prev,
                            actual_speedup=c.speedup_device,
                            severity="warning",
                        )
                    )

                # Non-monotonic pattern in the middle of data
                if (
                    len(prev_speedups) >= 2
                    and c.speedup_device < prev_speedups[-1] * 0.7
                    and prev_speedups[-1] > prev_speedups[-2]
                ):
                    outliers.append(
                        Outlier(
                            op=op,
                            config=c.config,
                            description=(
                                f"Non-monotonic speedup pattern: "
                                f"{c.speedup_device:.2f}x after {prev_speedups[-1]:.2f}x"
                            ),
                            expected_speedup=prev_speedups[-1],
                            actual_speedup=c.speedup_device,
                            severity="info",
                        )
                    )

            prev_speedups.append(c.speedup_device)

            # Keep only last 3 for moving average
            if len(prev_speedups) > 3:
                prev_speedups.pop(0)

    return outliers


def generate_routing_rules(
    break_evens: list[BreakEvenPoint],
) -> dict[str, dict[str, Any]]:
    """Generate routing rules for the backend router."""
    rules: dict[str, dict[str, Any]] = {}

    for be in break_evens:
        if be.mode != "device":  # Focus on device mode for routing
            continue

        op = be.op
        if op not in rules:
            rules[op] = {
                "thresholds": [],
                "recommendation": be.recommendation,
            }

        # Convert break-even data to thresholds
        for (h, w), min_batch in sorted(
            be.min_batch_for_size.items(), key=lambda x: x[0][0] * x[0][1]
        ):
            rules[op]["thresholds"].append(
                {
                    "min_hw": h * w,
                    "min_batch": min_batch,
                }
            )

    return rules


def format_report_text(
    comparisons: list[ComparisonResult],
    break_evens: list[BreakEvenPoint],
    outliers: list[Outlier],
) -> str:
    """Format analysis results as plain text."""
    lines = []
    lines.append("=" * 70)
    lines.append("CPU vs MLX Benchmark Analysis")
    lines.append("=" * 70)
    lines.append("")

    # Summary table
    lines.append("## Comparison Summary")
    lines.append("")
    lines.append(f"{'Op':<30} {'Config':<18} {'CPU':<10} {'MLX Dev':<10} {'Speedup':<8}")
    lines.append("-" * 78)

    for c in sorted(comparisons, key=lambda x: (x.op, x.height * x.width, x.batch)):
        speedup_str = f"{c.speedup_device:.2f}x" if c.speedup_device else "N/A"
        mlx_str = f"{c.mlx_device_ips:.0f}" if c.mlx_device_ips else "N/A"
        lines.append(
            f"{c.op:<30} {c.config:<18} {c.cpu_ips:<10.0f} {mlx_str:<10} {speedup_str:<8}"
        )

    lines.append("")
    lines.append("## Break-Even Points")
    lines.append("")

    for be in break_evens:
        if be.mode == "device":
            lines.append(f"### {be.op}")
            lines.append(f"   Recommendation: {be.recommendation}")
            lines.append("")

    if outliers:
        lines.append("## Outliers Detected")
        lines.append("")
        for o in outliers:
            lines.append(f"- [{o.severity.upper()}] {o.op} @ {o.config}")
            lines.append(f"  {o.description}")
            lines.append("")

    return "\n".join(lines)


def format_report_markdown(
    comparisons: list[ComparisonResult],
    break_evens: list[BreakEvenPoint],
    outliers: list[Outlier],
    routing_rules: dict[str, dict[str, Any]],
) -> str:
    """Format analysis results as Markdown."""
    lines = []
    lines.append("# CPU vs MLX Benchmark Analysis")
    lines.append("")

    # Summary statistics
    mlx_wins = sum(1 for c in comparisons if c.mlx_wins_device)
    cpu_wins = sum(1 for c in comparisons if c.speedup_device and c.speedup_device < 1)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total comparisons: {len(comparisons)}")
    lines.append(f"- MLX faster (device): {mlx_wins}")
    lines.append(f"- CPU faster: {cpu_wins}")
    lines.append("")

    # Break-even table
    lines.append("## Break-Even Points")
    lines.append("")
    lines.append("| Op | Mode | Recommendation |")
    lines.append("|:---|:-----|:---------------|")

    for be in sorted(break_evens, key=lambda x: (x.op, x.mode)):
        lines.append(f"| {be.op} | {be.mode} | {be.recommendation} |")

    lines.append("")

    # Detailed comparison table
    lines.append("## Detailed Comparisons")
    lines.append("")
    lines.append("| Op | Config | Batch | H×W | CPU img/s | MLX img/s | Speedup |")
    lines.append("|:---|:-------|------:|----:|----------:|----------:|--------:|")

    for c in sorted(comparisons, key=lambda x: (x.op, x.height * x.width, x.batch)):
        speedup_str = f"{c.speedup_device:.2f}×" if c.speedup_device else "—"
        mlx_str = f"{c.mlx_device_ips:,.0f}" if c.mlx_device_ips else "—"
        hw = c.height * c.width
        lines.append(
            f"| {c.op} | {c.config} | {c.batch} | {hw:,} | "
            f"{c.cpu_ips:,.0f} | {mlx_str} | {speedup_str} |"
        )

    lines.append("")

    # Outliers
    if outliers:
        lines.append("## Outliers")
        lines.append("")
        for o in outliers:
            emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(o.severity, "•")
            lines.append(f"- {emoji} **{o.op}** @ `{o.config}`: {o.description}")
        lines.append("")

    # Routing rules
    lines.append("## Routing Rules")
    lines.append("")
    lines.append("```python")
    lines.append("# Auto-generated routing thresholds")
    lines.append("BACKEND_ROUTING_RULES = {")
    for op, rule in sorted(routing_rules.items()):
        lines.append(f'    "{op}": {{')
        lines.append(f'        "thresholds": [')
        for t in rule["thresholds"]:
            lines.append(f'            {{"min_hw": {t["min_hw"]}, "min_batch": {t["min_batch"]}}},')
        lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Benchmark JSON files to analyze",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as Markdown",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    # Parse all files
    all_results: list[BenchmarkResult] = []
    for path in args.files:
        if path.exists():
            all_results.extend(parse_benchmark_file(path))
        else:
            print(f"Warning: {path} not found", file=sys.stderr)

    if not all_results:
        print("No benchmark results found", file=sys.stderr)
        sys.exit(1)

    # Analyze
    comparisons = compare_backends(all_results)
    break_evens = find_break_even_points(comparisons)
    outliers = detect_outliers(comparisons)
    routing_rules = generate_routing_rules(break_evens)

    # Format output
    if args.json:
        output = json.dumps(
            {
                "comparisons": [
                    {
                        "op": c.op,
                        "config": c.config,
                        "batch": c.batch,
                        "height": c.height,
                        "width": c.width,
                        "cpu_ips": c.cpu_ips,
                        "mlx_device_ips": c.mlx_device_ips,
                        "speedup_device": c.speedup_device,
                    }
                    for c in comparisons
                ],
                "break_evens": [
                    {
                        "op": be.op,
                        "mode": be.mode,
                        "min_batch_for_size": {
                            f"{h}x{w}": b for (h, w), b in be.min_batch_for_size.items()
                        },
                        "min_pixels_batch1": be.min_pixels_batch1,
                        "recommendation": be.recommendation,
                    }
                    for be in break_evens
                ],
                "outliers": [
                    {
                        "op": o.op,
                        "config": o.config,
                        "description": o.description,
                        "severity": o.severity,
                    }
                    for o in outliers
                ],
                "routing_rules": routing_rules,
            },
            indent=2,
        )
    elif args.markdown:
        output = format_report_markdown(comparisons, break_evens, outliers, routing_rules)
    else:
        output = format_report_text(comparisons, break_evens, outliers)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()