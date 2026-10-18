#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
ENABLE_HIP="${GKS_ENABLE_HIP:-ON}"
GPU_TARGETS="${GKS_GPU_TARGETS:-}"

mkdir -p "$BUILD_DIR"

CMAKE_ARGS=(
  -S "$ROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  -DGKS_ENABLE_HIP="${ENABLE_HIP}"
)

if [[ -n "$GPU_TARGETS" ]]; then
  CMAKE_ARGS+=(-DGKS_GPU_TARGETS="${GPU_TARGETS}")
fi

if command -v ninja >/dev/null 2>&1; then
  CMAKE_ARGS+=(-G Ninja)
fi

echo "Configuring: ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "Built: $BUILD_DIR/gpu_suite"
"$BUILD_DIR/gpu_suite" --list || true
