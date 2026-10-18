#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gks {

using index_t = std::size_t;
using f32 = float;

enum class KernelOp {
  VectorAdd,
  Matmul,
  ReductionSum,
  ReductionMax,
  Transpose,
};

enum class Variant {
  CpuRef,
  HipBaseline,
  HipTiled,
  HipTiledOptimized,
  HipHierarchical,
};

inline const char* to_string(KernelOp op) {
  switch (op) {
    case KernelOp::VectorAdd: return "vector_add";
    case KernelOp::Matmul: return "matmul";
    case KernelOp::ReductionSum: return "reduction_sum";
    case KernelOp::ReductionMax: return "reduction_max";
    case KernelOp::Transpose: return "transpose";
  }
  return "unknown";
}

inline const char* to_string(Variant v) {
  switch (v) {
    case Variant::CpuRef: return "cpu_ref";
    case Variant::HipBaseline: return "hip_baseline";
    case Variant::HipTiled: return "hip_tiled";
    case Variant::HipTiledOptimized: return "hip_tiled_opt";
    case Variant::HipHierarchical: return "hip_hierarchical";
  }
  return "unknown";
}

struct Shape2D {
  index_t rows = 0;
  index_t cols = 0;
};

struct LaunchConfig {
  index_t block_x = 256;
  index_t block_y = 1;
  index_t tile = 16;
};

struct Tolerance {
  double abs_tol = 1e-5;
  double rel_tol = 1e-5;
  bool require_exact = false;
};

struct CompareResult {
  bool passed = false;
  index_t first_mismatch = static_cast<index_t>(-1);
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  std::string message;
};

struct TimingStats {
  std::vector<double> samples_ms;
  double median_ms = 0.0;
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

struct DeviceInfo {
  std::string name = "unavailable";
  std::string arch = "unavailable";
  int device_count = 0;
  std::size_t global_mem_bytes = 0;
  std::string hip_runtime = "unavailable";
  std::string hip_compiler = "unavailable";
  std::string rocm_version = "unavailable";
  bool gpu_available = false;
};

struct BenchmarkRecord {
  std::string timestamp_utc;
  KernelOp operation = KernelOp::VectorAdd;
  Variant variant = Variant::HipBaseline;
  std::string dtype = "f32";
  index_t size = 0;
  index_t m = 0;
  index_t n = 0;
  index_t k = 0;
  LaunchConfig launch{};
  int warmup = 0;
  int trials = 0;
  TimingStats kernel_timing{};
  TimingStats e2e_timing{};
  bool correctness_passed = false;
  std::string correctness_message;
  double metric_value = 0.0;
  std::string metric_name;
  std::string metric_unit;
  double speedup_vs_baseline = 0.0;
  DeviceInfo device{};
  std::string build_config = "unavailable";
  std::string compiler_flags = "unavailable";
  std::string status = "ok";
  std::string notes;
};

}  // namespace gks
