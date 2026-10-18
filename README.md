# GPU Kernel Optimization and Characterization Suite

Framework for implementing, validating, benchmarking, and characterizing HIP kernels on **AMD Instinct** accelerators (ROCm/HIP), with a clear path to a future CUDA backend.

> **Benchmark status: UNBENCHMARKED on AMD GPU.**  
> This repository was bootstrapped on an Apple M1 host without ROCm/HIP. Host-only paths are verified. GPU execution, profiling, and performance conclusions require AMD Developer Cloud (or another ROCm system). Do not treat host `cpu_ref` timings as Instinct results.

## Motivation

GPU performance engineering needs more than kernel demos: independent correctness oracles, repeatable timing, statistical summaries, sweep-based characterization, and honest reporting when hardware or tools are missing.

## Features

- Kernels: **vector add**, **matmul** (naive + LDS tiled), **reduction** (sum/max; baseline + hierarchical), **transpose** (naive + tiled + bank-conflict-aware tiled)
- Independent **CPU references** and tolerance-based validation
- HIP **event timing**, raw samples, median/mean/stddev/min/max
- CSV/JSON export with device/build metadata (`unavailable` when unknown)
- Python **quick/full** sweeps, analysis, and plots from stored data
- CMake with HIP auto-detect and host-only fallback

## Architecture

See [docs/architecture.md](docs/architecture.md). High level:

- `gpu_suite` — C++ CLI (correctness + single-config benchmarks)
- `python/run_benchmarks.py` — configuration sweeps
- `python/analyze_results.py` / `plot_results.py` — offline reporting

## Requirements

### AMD GPU node (primary)

- ROCm with `hipcc`, HIP runtime, visible Instinct GPU
- CMake ≥ 3.21, Ninja or Make, C++17
- Python 3.9+ with deps in `python/requirements.txt`

### Laptop without ROCm (development)

- CMake + C++17 compiler
- Project venv for Python tools
- Use `--host-only` / `cpu_ref` only; mark results as non-GPU

## Build

```bash
# Probe tools (safe on any host)
./scripts/env_probe.sh

# Configure + build (auto host-only if hipcc missing)
./scripts/build.sh

# On AMD cloud, optionally pin architecture after rocminfo:
# GKS_GPU_TARGETS=gfx90a ./scripts/build.sh
```

Manual CMake:

```bash
cmake -S . -B build -DGKS_ENABLE_HIP=ON -DGKS_GPU_TARGETS=<gfxXXX>
cmake --build build -j
```

## CLI

```bash
./build/gpu_suite --list
./build/gpu_suite --test --host-only
./build/gpu_suite --kernel vector_add --size 1048576 --variant cpu_ref --host-only \
  --output-json results/va.json --output-csv results/va.csv

# On AMD GPU node:
./build/gpu_suite --test
./build/gpu_suite --kernel matmul --variant tiled --m 256 --n 256 --k 256 --tile 16
./build/gpu_suite --kernel reduction --variant hierarchical --size 1000000
./build/gpu_suite --kernel transpose --variant tiled_opt --m 1024 --n 1024
```

## Benchmarks and reports

```bash
# Create venv once
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt

# Host-path smoke (not AMD GPU data)
HOST_ONLY=1 ./scripts/run_all.sh quick

# On AMD GPU:
./scripts/run_all.sh quick
./scripts/run_all.sh full

.venv/bin/python python/analyze_results.py --input results/<run_dir>
.venv/bin/python python/plot_results.py --input results/<run_dir>
```

## Correctness

`gpu_suite --test` covers empty/odd/non-power-of-two sizes, non-tile-aligned matrices, randomized inputs (seeded), and special values where relevant. Failures print `FAIL` with configuration detail and nonzero exit status.

## Result schema (JSON excerpt)

```json
{
  "operation": "vector_add",
  "variant": "hip_baseline",
  "kernel_timing": {
    "samples_ms": [0.01, 0.012],
    "median_ms": 0.011,
    "mean_ms": 0.011,
    "stddev_ms": 0.001,
    "min_ms": 0.01,
    "max_ms": 0.012
  },
  "correctness_passed": true,
  "metric_name": "effective_bandwidth",
  "metric_unit": "GB/s",
  "device": {
    "name": "unavailable",
    "arch": "unavailable",
    "gpu_available": false
  },
  "status": "ok",
  "notes": "bytes=3*N*sizeof(f32); effective not DRAM-peak"
}
```

The numeric values above are **illustrative of schema only**. Real measurements must come from executed runs.

## Documentation


| Doc                                                              | Content                        |
| ---------------------------------------------------------------- | ------------------------------ |
| [docs/benchmark_methodology.md](docs/benchmark_methodology.md)   | Timing, stats, metrics         |
| [docs/optimization_notes.md](docs/optimization_notes.md)         | Hypotheses / evidence log      |
| [docs/hardware_compatibility.md](docs/hardware_compatibility.md) | Host vs cloud inventory        |
| [docs/architecture.md](docs/architecture.md)                     | Structure and extension points |


## Known limitations

- No AMD GPU execution from the bootstrap host
- Hierarchical reduction timed path includes multi-launch device work + scalar D2H
- Matmul tile sizes limited to 8/16/32 templates
- No CUDA backend yet
- Profiler-driven conclusions deferred until tools exist on the cloud image

## Future CUDA extension

1. Add `src/cuda/kernels.cu` implementing the same operations.
2. Mirror `gks::hip_api` as `gks::cuda_api` or a thin `backend` enum in CMake.
3. Keep CPU oracles and Python analysis unchanged.

## Reproduction checklist (AMD cloud)

1. `./scripts/env_probe.sh` → record GPU/ROCm in `docs/hardware_compatibility.md`
2. `./scripts/build.sh` (set `GKS_GPU_TARGETS` if required)
3. `./build/gpu_suite --test`
4. `./scripts/run_all.sh quick` then `full`
5. Fill measured outcomes into `docs/optimization_notes.md`
6. Commit only real `results/` you intend to publish; never invent numbers

## License

Educational portfolio project; add a license of your choice before public distribution.