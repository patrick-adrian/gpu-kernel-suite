#!/usr/bin/env python3
"""Analyze existing CSV/JSON benchmark results without rerunning GPU work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def summarize(df: pd.DataFrame) -> str:
    lines = ["# Benchmark Analysis Report", ""]
    if df.empty:
        lines.append("No rows found.")
        return "\n".join(lines) + "\n"

    if "gpu_available" in df.columns and not df["gpu_available"].astype(str).str.lower().eq(
        "true"
    ).any():
        lines.append(
            "**Status: UNBENCHMARKED on AMD GPU** (no rows with `gpu_available=true`)."
        )
        lines.append("")

    lines.append(f"Rows: {len(df)}")
    if "status" in df.columns:
        lines.append("Status counts:")
        for status, count in df["status"].value_counts(dropna=False).items():
            lines.append(f"- {status}: {count}")
        lines.append("")

    ok = df
    if "correctness_passed" in df.columns:
        ok = df[df["correctness_passed"].astype(str).str.lower().isin(["true", "1"])]
        lines.append(f"Correctness-passed rows: {len(ok)} / {len(df)}")
        lines.append("")

    for op, group in ok.groupby("operation"):
        lines.append(f"## {op}")
        if group.empty:
            lines.append("No passing rows.")
            lines.append("")
            continue

        # Fastest measured configuration per workload key.
        keys = [c for c in ["size", "m", "n", "k", "tile"] if c in group.columns]
        metric_col = "median_ms"
        if metric_col not in group.columns:
            lines.append("Missing median_ms.")
            lines.append("")
            continue

        # Baseline for speedup: hip_baseline within same op/workload when present.
        for workload, wdf in group.groupby(keys, dropna=False):
            wdf = wdf.sort_values(metric_col)
            best = wdf.iloc[0]
            base = wdf[wdf["variant"].astype(str).str.contains("baseline", case=False)]
            speedup = None
            if not base.empty and float(base.iloc[0][metric_col]) > 0:
                speedup = float(base.iloc[0][metric_col]) / float(best[metric_col])
            lines.append(
                f"- workload={workload}: best variant=`{best['variant']}` "
                f"median_ms={best[metric_col]} "
                f"metric={best.get('metric_value', 'n/a')} {best.get('metric_unit', '')}"
                + (f" speedup_vs_baseline={speedup:.3f}" if speedup is not None else "")
            )
        lines.append("")

    # Variability flags
    if {"stddev_ms", "mean_ms"}.issubset(ok.columns):
        noisy = ok.copy()
        noisy["cv"] = noisy["stddev_ms"] / noisy["mean_ms"].replace(0, pd.NA)
        flagged = noisy[noisy["cv"] > 0.15]
        lines.append("## Variability")
        lines.append(f"Rows with CV>0.15: {len(flagged)}")
        lines.append("")

    lines.append(
        "Notes: Effective GB/s and GFLOP/s are derived metrics, not hardware-counter "
        "DRAM/compute peaks. Do not compare different workloads as equivalent."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Results directory or CSV file")
    ap.add_argument("--output", default=None, help="Markdown report path")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        csv_path = inp / "benchmarks.csv"
        if not csv_path.exists():
            # Prefer newest nested results dir if pointing at results/
            candidates = sorted(inp.glob("**/benchmarks.csv"))
            if not candidates:
                print(f"error: no benchmarks.csv under {inp}", file=sys.stderr)
                return 1
            csv_path = candidates[-1]
    else:
        csv_path = inp

    try:
        df = load_csv(csv_path)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    report = summarize(df)
    out = Path(args.output) if args.output else csv_path.parent / "analysis_report.md"
    out.write_text(report)
    print(report)
    print(f"Wrote {out}")

    # Machine-readable summary
    summary_json = {
        "source_csv": str(csv_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    (out.parent / "analysis_summary.json").write_text(json.dumps(summary_json, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
