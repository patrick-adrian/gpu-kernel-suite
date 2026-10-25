# Benchmark Analysis Report

Rows: 10
Status counts:
- ok: 9
- correctness_failed: 1

Correctness-passed rows: 9 / 10

## matmul
- workload=(127, 127, 255, 63, 16): best variant=`hip_tiled` median_ms=0.005813499912619591 metric=701.9024789425519 GFLOP/s speedup_vs_baseline=1.872
- workload=(128, 128, 128, 128, 16): best variant=`hip_tiled` median_ms=0.0073369997553527355 metric=571.6647321597674 GFLOP/s speedup_vs_baseline=2.527
- workload=(256, 256, 256, 256, 16): best variant=`hip_tiled` median_ms=0.01108550000935793 metric=3026.8758262301844 GFLOP/s speedup_vs_baseline=2.960

## reduction_sum
- workload=(10000, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.34641149640083313 metric=0.11546960887728726 GB/s speedup_vs_baseline=1.000

## vector_add
- workload=(65536, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.005873000016435981 metric=133.9063507235004 GB/s speedup_vs_baseline=1.000
- workload=(1048576, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.006173999980092049 metric=2038.0485974365681 GB/s speedup_vs_baseline=1.000

## Variability
Rows with CV>0.15: 0

Notes: Effective GB/s and GFLOP/s are derived metrics, not hardware-counter DRAM/compute peaks. Do not compare different workloads as equivalent.
