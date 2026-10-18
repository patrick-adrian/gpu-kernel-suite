# Optimization Notes

**Status: hypotheses documented; AMD GPU measurements not yet collected on the development host.**

Use this file as a living log. For each change on AMD Developer Cloud, record: hypothesis, code change, conditions, measured result, evidence, trade-offs.

## Vector addition

- **Hypothesis:** Memory-bound; performance tracks effective bandwidth; work-group size has mild effect once occupancy is sufficient.
- **Baseline:** One element per work-item, contiguous addressing (coalesced on AMD when consecutive work-items hit consecutive addresses).
- **Measured:** *pending AMD GPU run*.

## Matrix multiplication

- **Naive:** Each work-item streams a row of A and column of B; poor temporal reuse; high global memory traffic.
- **Tiled (LDS):** Load `TILE×TILE` blocks of A and B into LDS; reuse within the work-group; `__syncthreads()` between tile phases. Boundary-safe with zero-fill for partial tiles.
- **Hypothesis:** Tiled beats naive for large enough geometries when LDS bandwidth + reuse outweigh sync and extra moves. Small matrices may favor naive due to launch/sync overhead.
- **Tile sizes:** Compile-time 8/16/32. Unsupported request falls back to 16 with a note.
- **Measured:** *pending*.

## Reduction

- **Baseline:** Intentionally serial single-thread GPU loop — lower bound / pedagogical contrast, not a production pattern.
- **Hierarchical:** Work-group shared-memory tree + multi-level launches for arbitrary `N` (including non-powers-of-two). Grid uses grid-stride style double-load per work-item.
- **Numerical notes:** FP32 sum association differs from CPU double-accumulator reference; tolerances scale with `N`. Max reduction is exact for finite inputs; NaN/Inf handled per IEEE `fmax` behavior.
- **Measured:** *pending*.

## Transpose

- **Naive:** Often coalesced reads, strided writes (or vice versa) → poor write coalescing.
- **Tiled LDS:** Coalesced read into LDS, transposed write; classic bandwidth optimization.
- **Tiled + bank-conflict padding:** LDS array `TILE×(TILE+1)` to reduce bank conflicts on AMD LDS during transposed shared reads.
- **Hypothesis:** Padded tiled ≥ tiled ≥ naive for large 2D sizes; verify with effective bandwidth and, when available, ROCm profiler memory metrics.
- **Measured:** *pending*.

## Profiling checklist (cloud)

1. Run `scripts/env_probe.sh` and note `rocprof` / `rocprofv3` / Omniperf availability.
2. Collect kernel time, memory unit busy, LDS bank conflicts if counters exist for the installed tool version.
3. Distinguish inference from counter evidence in this log.
