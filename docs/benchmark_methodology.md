# Benchmark Methodology

## Phases

1. **Correctness** — GPU output vs independent CPU reference with absolute/relative tolerances (exact equality for transpose).
2. **Performance** — Only configurations that pass correctness (or explicitly recorded failures) are kept in sweeps.

## Timing categories

| Category | What is included | How |
|----------|------------------|-----|
| Kernel-only | Device kernel(s) for a single launch op | HIP events around launch; H2D/D2H/alloc outside |
| Device multi-launch | Hierarchical reduction passes | HIP events around the full device reduce sequence |
| End-to-end | Optional host wall time including setup | Host steady_clock when requested |

Warm-up iterations run before timed trials. Default: warmup=5, trials=20 (CLI-overridable).

## Statistics

Per configuration: raw `samples_ms`, median, mean, sample stddev, min, max. Coefficient of variation > 0.15 is flagged as noisy in analysis.

## Metrics (effective, not peak)

| Op | Formula | Label |
|----|---------|-------|
| Vector add | bytes = `3 * N * 4`; GB/s = bytes/time | effective bandwidth |
| Matmul | FLOPs = `2*M*N*K`; GFLOP/s = FLOPs/time | effective GFLOP/s |
| Reduction | bytes = `N * 4` | effective input bandwidth |
| Transpose | bytes = `2 * rows * cols * 4` | effective bandwidth |

These are **not** claimed DRAM bandwidth or hardware peak compute unless supported by profiler counters on the target Instinct GPU.

## Speedup

Speedup is relative to the **naive/baseline GPU variant of the same workload** unless a report explicitly states CPU comparison. CPU timings are single-threaded in this suite.

## Reproducibility

- Fixed RNG seed (default 42).
- Record device name/arch, HIP runtime, build mode, launch params, timestamp.
- Unavailable metadata is stored as `"unavailable"`, never invented.

## Suites

- `quick` — small set for iteration
- `full` — broader size/tile/block sweeps

Orchestrator: `python/run_benchmarks.py`.
