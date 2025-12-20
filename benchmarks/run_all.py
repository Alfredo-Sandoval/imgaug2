#!/usr/bin/env python3
"""Run the full imgaug2 benchmark suite (one entrypoint).

This replaces ad-hoc shell scripts with a single Python entrypoint that:
- runs the CPU augmenter-level benchmark suite
- runs the legacy-style suite (parity with checks/check_performance.py)
- runs the ops-level CPU vs MLX suite (includes mlx_device vs mlx_roundtrip)
- optionally runs a third-party CPU baseline (if installed)
- generates a Markdown summary report

Run from the repository root:
    python -m benchmarks.run_all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _print_header() -> None:
    line = "=" * 42
    print(line)
    print("imgaug2 Full Benchmark Suite")
    print(line)
    print("")


def run_all(
    *,
    output_dir: Path,
    report_path: Path,
    seed: int,
    iterations: int,
    warmup: int,
    legacy_warmup: int,
    label: str | None,
    fail_fast: bool,
    skip_legacy: bool,
    skip_ops: bool,
    skip_third_party: bool,
) -> dict[str, Any]:
    _print_header()

    results: dict[str, Any] = {"suites": {}}

    from benchmarks.runner import run_benchmarks

    print("[1/4] CPU augmenter-level suite")
    results["suites"]["runner_cpu"] = run_benchmarks(
        platform_name="cpu",
        output_dir=output_dir,
        iterations=iterations,
        warmup=warmup,
        seed=seed,
        augmenter_filter=None,
        config_filter=None,
        fail_fast=fail_fast,
        label=label,
    )

    if not skip_legacy:
        from benchmarks.legacy import (
            run_legacy_benchmark,
        )

        print("\n[2/4] Legacy suite (parity with checks/check_performance.py)")
        results["suites"]["legacy_cpu"] = run_legacy_benchmark(
            output_dir=output_dir,
            iterations=iterations,
            warmup=legacy_warmup,
            seed=seed,
            batch_size=16,
            include_images=True,
            include_keypoints=True,
            label=label,
        )
    else:
        print("\n[2/4] Legacy suite: skipped")

    if not skip_ops:
        from benchmarks.ops import run_ops_benchmarks

        print("\n[3/4] Ops suite (CPU vs MLX device/roundtrip)")
        results["suites"]["ops"] = run_ops_benchmarks(
            output_dir=output_dir,
            iterations=iterations,
            warmup=warmup,
            seed=seed,
            label=label,
        )
    else:
        print("\n[3/4] Ops suite: skipped")

    if skip_third_party:
        print("\n[4/4] Third-party baseline: skipped")
    else:
        try:
            from benchmarks import (
                third_party_baseline as _tpb,
            )
        except Exception:
            print("\n[4/4] Third-party baseline: unavailable; skipping")
        else:
            if not _tpb.third_party_available():
                print("\n[4/4] Third-party baseline: dependency not installed; skipping")
            else:
                print("\n[4/4] Third-party baseline (CPU)")
                results["suites"]["third_party_baseline"] = _tpb.run_third_party_baseline_benchmarks(
                    output_dir=output_dir,
                    iterations=iterations,
                    warmup=warmup,
                    seed=seed,
                    augmenter_filter=None,
                    config_filter=None,
                    fail_fast=fail_fast,
                    label=label,
                )

    from benchmarks.reports.generate_report import (
        generate_report,
    )

    print("\nGenerating report...")
    generate_report(output_dir, report_path)
    print(f"Wrote report to {report_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/reports/out/benchmark_report.md"),
        help="Path to write the Markdown report to.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--legacy-warmup",
        type=int,
        default=0,
        help="Legacy suite warmup iterations (default matches benchmarks.legacy).",
    )
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-ops", action="store_true")
    parser.add_argument("--skip-third-party", action="store_true")
    args = parser.parse_args()

    run_all(
        output_dir=args.output,
        report_path=args.report,
        seed=int(args.seed),
        iterations=int(args.iterations),
        warmup=int(args.warmup),
        legacy_warmup=int(args.legacy_warmup),
        label=(args.label.strip() or None),
        fail_fast=bool(args.fail_fast),
        skip_legacy=bool(args.skip_legacy),
        skip_ops=bool(args.skip_ops),
        skip_third_party=bool(args.skip_third_party),
    )


if __name__ == "__main__":
    main()
