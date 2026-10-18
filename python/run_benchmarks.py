#!/usr/bin/env python3
"""Enumerate workload sweeps, invoke gpu_suite, preserve raw CSV/JSON results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def suite_configs(suite: str) -> list[dict]:
    """Return benchmark configurations. quick: small; full: broader sweeps."""
    configs: list[dict] = []

    if suite == "quick":
        vec_sizes = [1 << 16, 1 << 20]
        mats = [(128, 128, 128), (256, 256, 256), (127, 255, 63)]
        tiles = [16]
        red_sizes = [10_000, 1_000_000]
        blocks = [256]
        transpose_shapes = [(256, 256), (257, 129)]
        trials, warmup = 10, 3
    else:
        vec_sizes = [1 << 14, 1 << 18, 1 << 20, 1 << 22, 1 << 24]
        mats = []
        for s in [64, 128, 256, 512, 1024]:
            mats.append((s, s, s))
        mats.extend([(127, 255, 63), (512, 256, 128), (768, 512, 256)])
        tiles = [8, 16, 32]
        red_sizes = [1, 1023, 65536, 1_000_003, 1 << 22]
        blocks = [128, 256, 512]
        transpose_shapes = [(128, 128), (256, 256), (512, 512), (1024, 1024), (513, 257)]
        trials, warmup = 20, 5

    for n in vec_sizes:
        configs.append(
            {
                "kernel": "vector_add",
                "variant": "hip_baseline",
                "args": ["--size", str(n)],
                "trials": trials,
                "warmup": warmup,
            }
        )

    for m, n, k in mats:
        configs.append(
            {
                "kernel": "matmul",
                "variant": "hip_baseline",
                "args": ["--m", str(m), "--n", str(n), "--k", str(k)],
                "trials": trials,
                "warmup": warmup,
            }
        )
        for tile in tiles:
            configs.append(
                {
                    "kernel": "matmul",
                    "variant": "hip_tiled",
                    "args": ["--m", str(m), "--n", str(n), "--k", str(k), "--tile", str(tile)],
                    "trials": trials,
                    "warmup": warmup,
                }
            )

    for n in red_sizes:
        for variant in ("hip_baseline", "hip_hierarchical"):
            for block in blocks:
                configs.append(
                    {
                        "kernel": "reduction_sum",
                        "variant": variant,
                        "args": ["--size", str(n), "--block", str(block)],
                        "trials": trials,
                        "warmup": warmup,
                    }
                )

    for rows, cols in transpose_shapes:
        for variant in ("hip_baseline", "hip_tiled", "hip_tiled_opt"):
            configs.append(
                {
                    "kernel": "transpose",
                    "variant": variant,
                    "args": ["--m", str(rows), "--n", str(cols), "--tile", "16"],
                    "trials": trials,
                    "warmup": warmup,
                }
            )

    return configs


def find_binary(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    here = Path(__file__).resolve().parent.parent
    candidates = [
        here / "build" / "gpu_suite",
        here / "build" / "Release" / "gpu_suite",
        here / "build" / "Debug" / "gpu_suite",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "gpu_suite not found. Build first (scripts/build.sh) or pass --binary."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU kernel suite benchmark orchestrator")
    ap.add_argument("--suite", choices=("quick", "full"), default="quick")
    ap.add_argument("--binary", default=None, help="Path to gpu_suite")
    ap.add_argument("--output-dir", default=None, help="Results directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--host-only",
        action="store_true",
        help="Force CPU variants (for environments without HIP)",
    )
    ap.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Continue when configurations are unsupported/failed",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else root / "results" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.suite
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmarks.csv"
    manifest = {
        "suite": args.suite,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host_only": args.host_only,
        "runs": [],
        "note": "UNBENCHMARKED on AMD GPU until executed under ROCm/HIP with a device.",
    }

    try:
        binary = find_binary(args.binary)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    configs = suite_configs(args.suite)
    if args.host_only:
        for c in configs:
            c["variant"] = "cpu_ref"

    failures = 0
    for i, cfg in enumerate(configs):
        json_path = out_dir / f"run_{i:04d}_{cfg['kernel']}_{cfg['variant']}.json"
        cmd = [
            str(binary),
            "--kernel",
            cfg["kernel"],
            "--variant",
            cfg["variant"],
            "--seed",
            str(args.seed),
            "--warmup",
            str(cfg["warmup"]),
            "--trials",
            str(cfg["trials"]),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            *cfg["args"],
        ]
        if args.host_only:
            cmd.append("--host-only")

        print("RUN:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        entry = {
            "index": i,
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "json": str(json_path) if json_path.exists() else None,
        }
        manifest["runs"].append(entry)
        if proc.returncode != 0:
            failures += 1
            print(f"FAIL ({proc.returncode}): {proc.stderr or proc.stdout}")
            if not args.allow_unsupported:
                # Still write manifest so partial results are preserved.
                break
        else:
            print(proc.stdout)

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote results under {out_dir}")
    if failures and not args.allow_unsupported:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
