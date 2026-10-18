#!/usr/bin/env python3
"""Generate plots from stored benchmark CSV without rerunning workloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_df(inp: Path) -> pd.DataFrame:
    if inp.is_dir():
        csv_path = inp / "benchmarks.csv"
        if not csv_path.exists():
            cands = sorted(inp.glob("**/benchmarks.csv"))
            if not cands:
                raise FileNotFoundError(f"no benchmarks.csv under {inp}")
            csv_path = cands[-1]
    else:
        csv_path = inp
    return pd.read_csv(csv_path), csv_path


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_latency_vs_size(df: pd.DataFrame, out_dir: Path) -> None:
    for op, g in df.groupby("operation"):
        if "size" not in g.columns or g["size"].isna().all():
            continue
        plt.figure()
        for variant, vg in g.groupby("variant"):
            vg = vg.sort_values("size")
            plt.plot(vg["size"], vg["median_ms"], marker="o", label=str(variant))
        plt.xlabel("size")
        plt.ylabel("median latency (ms)")
        plt.title(f"{op}: latency vs input size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig(out_dir / f"{op}_latency_vs_size.png")


def plot_throughput(df: pd.DataFrame, out_dir: Path) -> None:
    if "metric_value" not in df.columns:
        return
    for op, g in df.groupby("operation"):
        plt.figure()
        for variant, vg in g.groupby("variant"):
            xcol = "size" if "size" in vg and vg["size"].notna().any() else None
            if xcol is None:
                continue
            vg = vg.sort_values(xcol)
            unit = str(vg["metric_unit"].iloc[0]) if "metric_unit" in vg else ""
            plt.plot(vg[xcol], vg["metric_value"], marker="o", label=str(variant))
            ylabel = f"effective metric ({unit})"
        if plt.gca().has_data():
            plt.xlabel("size")
            plt.ylabel(ylabel)
            plt.title(f"{op}: effective throughput vs size")
            plt.legend()
            plt.grid(True, alpha=0.3)
            savefig(out_dir / f"{op}_throughput_vs_size.png")
        else:
            plt.close()


def plot_matmul_tiles(df: pd.DataFrame, out_dir: Path) -> None:
    g = df[df["operation"] == "matmul"]
    if g.empty or "tile" not in g.columns:
        return
    tiled = g[g["variant"].astype(str).str.contains("tiled", case=False)]
    if tiled.empty:
        return
    plt.figure()
    for (m, n, k), tg in tiled.groupby(["m", "n", "k"]):
        tg = tg.sort_values("tile")
        plt.plot(tg["tile"], tg["median_ms"], marker="o", label=f"{m}x{n}x{k}")
    plt.xlabel("tile")
    plt.ylabel("median latency (ms)")
    plt.title("matmul: latency vs tile configuration")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    savefig(out_dir / "matmul_latency_vs_tile.png")


def plot_variability(df: pd.DataFrame, out_dir: Path) -> None:
    if not {"mean_ms", "stddev_ms"}.issubset(df.columns):
        return
    plt.figure()
    plt.errorbar(
        range(len(df)),
        df["mean_ms"],
        yerr=df["stddev_ms"],
        fmt="o",
        alpha=0.7,
    )
    plt.xlabel("run index")
    plt.ylabel("mean latency ± stddev (ms)")
    plt.title("Timing variability across recorded runs")
    plt.grid(True, alpha=0.3)
    savefig(out_dir / "timing_variability.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    try:
        df, csv_path = load_df(Path(args.input))
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if df.empty:
        print("No data to plot.")
        return 0

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer correctness-passing rows when column exists.
    if "correctness_passed" in df.columns:
        passed = df[df["correctness_passed"].astype(str).str.lower().isin(["true", "1"])]
        if not passed.empty:
            df = passed

    plot_latency_vs_size(df, out_dir)
    plot_throughput(df, out_dir)
    plot_matmul_tiles(df, out_dir)
    plot_variability(df, out_dir)

    note = out_dir / "PLOT_NOTES.txt"
    note.write_text(
        "Plots generated from stored results only.\n"
        "Effective bandwidth/GFLOP/s are calculated metrics, not profiler DRAM/compute peaks.\n"
        "If gpu_available was false, treat figures as host-only / non-AMD-GPU data.\n"
    )
    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
