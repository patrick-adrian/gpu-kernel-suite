# Benchmark Analysis Report

Rows: 47
Status counts:
- ok: 46
- correctness_failed: 1

Correctness-passed rows: 46 / 47

## matmul
- workload=(64, 64, 64, 64, 8): best variant=`hip_tiled` median_ms=0.006254500010982156 metric=83.82572533046812 GFLOP/s
- workload=(64, 64, 64, 64, 16): best variant=`hip_tiled` median_ms=0.005812999792397022 metric=90.19233076280689 GFLOP/s speedup_vs_baseline=1.893
- workload=(64, 64, 64, 64, 32): best variant=`hip_tiled` median_ms=0.006494499975815415 metric=80.72800091652525 GFLOP/s
- workload=(127, 127, 255, 63, 8): best variant=`hip_tiled` median_ms=0.006454499904066324 metric=632.1961516227284 GFLOP/s
- workload=(127, 127, 255, 63, 16): best variant=`hip_tiled` median_ms=0.005812999792397022 metric=701.962866975672 GFLOP/s speedup_vs_baseline=1.873
- workload=(127, 127, 255, 63, 32): best variant=`hip_tiled` median_ms=0.006435000104829669 metric=634.1118777818588 GFLOP/s
- workload=(128, 128, 128, 128, 8): best variant=`hip_tiled` median_ms=0.008900499902665615 metric=471.24364315130725 GFLOP/s
- workload=(128, 128, 128, 128, 16): best variant=`hip_tiled` median_ms=0.007216999772936106 metric=581.1700335267745 GFLOP/s speedup_vs_baseline=2.569
- workload=(128, 128, 128, 128, 32): best variant=`hip_tiled` median_ms=0.009380999952554703 metric=447.10628090961416 GFLOP/s
- workload=(256, 256, 256, 256, 8): best variant=`hip_tiled` median_ms=0.014953999780118465 metric=2243.843285634593 GFLOP/s
- workload=(256, 256, 256, 256, 16): best variant=`hip_tiled` median_ms=0.011086000129580498 metric=3026.739275463974 GFLOP/s speedup_vs_baseline=2.964
- workload=(256, 256, 256, 256, 32): best variant=`hip_tiled` median_ms=0.014894000254571438 metric=2252.8824645146015 GFLOP/s
- workload=(512, 512, 256, 128, 8): best variant=`hip_tiled` median_ms=0.010223000310361385 metric=3282.2489466219968 GFLOP/s
- workload=(512, 512, 256, 128, 16): best variant=`hip_tiled` median_ms=0.007737999781966209 metric=4336.318550719046 GFLOP/s speedup_vs_baseline=2.373
- workload=(512, 512, 256, 128, 32): best variant=`hip_tiled` median_ms=0.009480999782681465 metric=3539.1238022484126 GFLOP/s
- workload=(512, 512, 512, 512, 8): best variant=`hip_tiled` median_ms=0.0436599999666214 metric=6148.315533788873 GFLOP/s
- workload=(512, 512, 512, 512, 16): best variant=`hip_tiled` median_ms=0.028906000778079033 metric=9286.49584080718 GFLOP/s speedup_vs_baseline=2.198
- workload=(512, 512, 512, 512, 32): best variant=`hip_tiled` median_ms=0.027101999148726463 metric=9904.636721701541 GFLOP/s
- workload=(768, 768, 512, 256, 8): best variant=`hip_tiled` median_ms=0.02630000002682209 metric=7655.003490291894 GFLOP/s
- workload=(768, 768, 512, 256, 16): best variant=`hip_tiled` median_ms=0.021889999508857727 metric=9197.194907132536 GFLOP/s speedup_vs_baseline=1.591
- workload=(768, 768, 512, 256, 32): best variant=`hip_tiled` median_ms=0.023773999884724617 metric=8468.351685715172 GFLOP/s
- workload=(1024, 1024, 1024, 1024, 8): best variant=`hip_tiled` median_ms=0.23846399784088135 metric=9005.483710094219 GFLOP/s
- workload=(1024, 1024, 1024, 1024, 16): best variant=`hip_tiled` median_ms=0.18287599831819534 metric=11742.840327594455 GFLOP/s speedup_vs_baseline=1.816
- workload=(1024, 1024, 1024, 1024, 32): best variant=`hip_tiled` median_ms=0.177263505756855 metric=12114.640511203781 GFLOP/s

## reduction_sum
- workload=(1, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.005652999971061945 metric=0.0007075888944766048 GB/s speedup_vs_baseline=1.000
- workload=(1023, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.03848699852824211 metric=0.10632161915659007 GB/s speedup_vs_baseline=1.000

## vector_add
- workload=(16384, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.005812999792397022 metric=33.822124036052585 GB/s speedup_vs_baseline=1.000
- workload=(262144, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.005894000176340342 metric=533.7169843712529 GB/s speedup_vs_baseline=1.000
- workload=(1048576, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.006173999980092049 metric=2038.0485974365681 GB/s speedup_vs_baseline=1.000
- workload=(4194304, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.014012000057846308 metric=3592.038809036099 GB/s speedup_vs_baseline=1.000
- workload=(16777216, 0, 0, 0, 16): best variant=`hip_baseline` median_ms=0.04945199936628342 metric=4071.1517143888286 GB/s speedup_vs_baseline=1.000

## Variability
Rows with CV>0.15: 1

Notes: Effective GB/s and GFLOP/s are derived metrics, not hardware-counter DRAM/compute peaks. Do not compare different workloads as equivalent.
