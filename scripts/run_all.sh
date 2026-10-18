#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SUITE="${1:-quick}"
HOST_ONLY="${HOST_ONLY:-0}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

if [[ ! -x "$BUILD_DIR/gpu_suite" ]]; then
  echo "Building..."
  "$ROOT/scripts/build.sh"
fi

echo "=== Correctness ==="
if [[ "$HOST_ONLY" == "1" ]]; then
  "$BUILD_DIR/gpu_suite" --test --host-only
else
  "$BUILD_DIR/gpu_suite" --test || "$BUILD_DIR/gpu_suite" --test --host-only
fi

echo "=== Benchmarks ($SUITE) ==="
EXTRA=()
if [[ "$HOST_ONLY" == "1" ]]; then
  EXTRA+=(--host-only --allow-unsupported)
fi
"$PYTHON" "$ROOT/python/run_benchmarks.py" --suite "$SUITE" --binary "$BUILD_DIR/gpu_suite" "${EXTRA[@]}"

LATEST="$(ls -td "$ROOT/results"/*/ 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST" ]]; then
  echo "=== Analyze $LATEST ==="
  "$PYTHON" "$ROOT/python/analyze_results.py" --input "$LATEST"
  "$PYTHON" "$ROOT/python/plot_results.py" --input "$LATEST"
fi

echo "Done."
