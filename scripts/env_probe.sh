#!/usr/bin/env bash
# Probe AMD/ROCm/HIP availability. Safe to run on non-AMD hosts.
set -u

echo "=== Host ==="
uname -a || true

probe() {
  local cmd="$1"
  echo "--- $cmd ---"
  if command -v "$cmd" >/dev/null 2>&1; then
    "$@" 2>&1 | head -80 || true
  else
    echo "NOT FOUND"
  fi
}

probe rocminfo
probe amd-smi
probe rocm-smi
probe hipcc --version
probe hipconfig --full
probe cmake --version
probe rocprof --help
probe rocprofv3 --help

echo "--- /opt/rocm ---"
ls /opt/rocm 2>/dev/null || echo "not present"

echo "--- env ---"
env | grep -iE 'ROCM|HIP|AMDGPU' | sort || true
