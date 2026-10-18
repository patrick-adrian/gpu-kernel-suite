# Hardware and Software Compatibility

## Discovery status

| Item | Value |
|------|-------|
| Probe date | 2026-10-17 |
| Development host | Apple M1 MacBook Air (Darwin arm64) |
| Local AMD GPU | **None** |
| Local ROCm/HIP | **Not installed** (`hipcc`, `rocminfo`, `/opt/rocm` absent) |
| CMake | 4.1.2 |
| Host C++ | Apple Clang 16.0.0 |
| Python | 3.13.5 (+ project venv: pandas, matplotlib, pytest, numpy) |
| Target platform | AMD Instinct via AMD Developer Cloud (model TBD) |
| AMDGPU target (`gfx*`) | **Not selected** — set `-DGKS_GPU_TARGETS=` only after `rocminfo` on the cloud node |
| Profiling tools | Unknown until cloud probe (`scripts/env_probe.sh`) |

## Build modes

1. **Host-only** (current laptop): CMake falls back when `hipcc` is missing. HIP APIs are stubbed. Use `--host-only` / `cpu_ref` for correctness of CPU references and I/O paths.
2. **HIP/ROCm** (AMD cloud): With `hipcc` on `PATH` and a visible device, CMake compiles `src/hip/kernels.hip` and defines `GKS_WITH_HIP=1`.

## Compatibility risks

- Instinct models differ in compute units, memory hierarchy, LDS banks, and matrix engines. Tiled kernels use portable LDS tiling; they are **not** tuned to a specific CDNA generation until measured.
- Do not assume NVIDIA warp/occupancy terminology applies. Use wavefront/work-group/LDS language on AMD.
- Short kernels may be launch-overhead dominated; interpret small absolute deltas cautiously.
- Hierarchical reduction timing includes multiple device launches (and a scalar D2H in the timed path). Compare only like-for-like.

## Re-probe on AMD Developer Cloud

```bash
./scripts/env_probe.sh
rocminfo | less
# Then configure with an explicit target if required, e.g.:
# GKS_GPU_TARGETS=gfx90a ./scripts/build.sh
```

Update this file with GPU name, `gcnArchName`, ROCm version, `hipcc --version`, and profiler availability after the first successful cloud session.

## Benchmark status

**UNBENCHMARKED on AMD GPU.** No Instinct performance numbers have been collected in this repository from this development host. Any CSV/JSON produced with `gpu_available=false` is host-path data only.
