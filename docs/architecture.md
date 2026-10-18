# Architecture

## Goals

Unified C++/HIP binary for kernel variants, correctness checking, device-event timing, and metadata export; Python layer for sweeps, analysis, and plots.

## Layout

```
include/common/     types, compare, stats, metrics, CLI, JSON/CSV I/O
include/kernels/    CPU API + HIP API declarations
include/benchmark/  device query + runner
src/hip/            HIP kernels (kernels.hip) or stubs (kernels_stub.cpp)
src/benchmark/      main, runner, device metadata
python/             run_benchmarks, analyze_results, plot_results
docs/               methodology and optimization notes
results/            machine-readable outputs (not fabricated)
```

## Backend boundary

`gks::hip_api::*` is the GPU backend surface. Today it is HIP-only. A future CUDA backend would implement the same function signatures behind a compile-time or runtime switch without rewriting the runner.

CPU references live in `gks::cpu::*` and are never replaced by GPU results during validation.

## Execution flow

1. CLI parse (`--kernel`, `--variant`, sizes, trials, outputs).
2. Correctness: CPU oracle vs GPU (or host-only CPU self-check).
3. Timing: HIP events around device work (kernel-only for single-launch ops; multi-launch for hierarchical reduction).
4. Metrics: effective GFLOP/s or GB/s from documented arithmetic/byte conventions.
5. Export: JSON (one record) + CSV append (raw samples preserved).

## Design constraints

- Prefer straightforward functions over deep class hierarchies.
- Fail loudly on missing devices / HIP errors; never treat “not run” as PASS.
- Do not hardcode Instinct architecture until probed.
