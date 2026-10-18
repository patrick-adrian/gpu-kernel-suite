#pragma once

#include "common/types.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace gks {

struct CliOptions {
  bool show_help = false;
  bool list_kernels = false;
  bool run_tests = false;
  bool host_only = false;
  std::optional<std::string> kernel;
  std::optional<std::string> variant;
  std::optional<index_t> size;
  std::optional<index_t> m;
  std::optional<index_t> n;
  std::optional<index_t> k;
  index_t tile = 16;
  index_t block = 256;
  int warmup = 5;
  int trials = 20;
  std::uint64_t seed = 42;
  Tolerance tol{};
  std::string output_json;
  std::string output_csv;
  std::string timing_mode = "kernel";  // kernel | e2e | both
};

inline void print_help(const char* argv0) {
  std::cout
      << "GPU Kernel Optimization and Characterization Suite\n\n"
      << "Usage:\n"
      << "  " << argv0 << " --list\n"
      << "  " << argv0 << " --test [--host-only]\n"
      << "  " << argv0 << " --kernel <name> [options]\n\n"
      << "Kernels: vector_add, matmul, reduction_sum, reduction_max, transpose\n"
      << "Variants: hip_baseline, hip_tiled, hip_tiled_opt, hip_hierarchical, cpu_ref\n\n"
      << "Options:\n"
      << "  --size N              Vector/reduction length or square matrix dim\n"
      << "  --m/--n/--k N         Rectangular matrix dimensions\n"
      << "  --variant NAME        Implementation variant\n"
      << "  --tile N              Tile size (matmul/transpose)\n"
      << "  --block N             Work-group size (1D)\n"
      << "  --warmup N            Warm-up iterations (default 5)\n"
      << "  --trials N            Timed trials (default 20)\n"
      << "  --seed N              RNG seed (default 42)\n"
      << "  --abs-tol X           Absolute FP tolerance\n"
      << "  --rel-tol X           Relative FP tolerance\n"
      << "  --timing MODE         kernel | e2e | both\n"
      << "  --output-json PATH    Write JSON result\n"
      << "  --output-csv PATH     Append CSV result\n"
      << "  --host-only           Run CPU-only paths (no HIP device required)\n"
      << "  --help                Show this help\n";
}

inline bool parse_cli(int argc, char** argv, CliOptions& opt, std::string& err) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        err = std::string("missing value for ") + name;
        return {};
      }
      return argv[++i];
    };

    if (a == "--help" || a == "-h") {
      opt.show_help = true;
    } else if (a == "--list") {
      opt.list_kernels = true;
    } else if (a == "--test") {
      opt.run_tests = true;
    } else if (a == "--host-only") {
      opt.host_only = true;
    } else if (a == "--kernel") {
      opt.kernel = need("--kernel");
    } else if (a == "--variant") {
      opt.variant = need("--variant");
    } else if (a == "--size") {
      opt.size = static_cast<index_t>(std::stoull(need("--size")));
    } else if (a == "--m") {
      opt.m = static_cast<index_t>(std::stoull(need("--m")));
    } else if (a == "--n") {
      opt.n = static_cast<index_t>(std::stoull(need("--n")));
    } else if (a == "--k") {
      opt.k = static_cast<index_t>(std::stoull(need("--k")));
    } else if (a == "--tile") {
      opt.tile = static_cast<index_t>(std::stoull(need("--tile")));
    } else if (a == "--block") {
      opt.block = static_cast<index_t>(std::stoull(need("--block")));
    } else if (a == "--warmup") {
      opt.warmup = std::stoi(need("--warmup"));
    } else if (a == "--trials") {
      opt.trials = std::stoi(need("--trials"));
    } else if (a == "--seed") {
      opt.seed = std::stoull(need("--seed"));
    } else if (a == "--abs-tol") {
      opt.tol.abs_tol = std::stod(need("--abs-tol"));
    } else if (a == "--rel-tol") {
      opt.tol.rel_tol = std::stod(need("--rel-tol"));
    } else if (a == "--timing") {
      opt.timing_mode = need("--timing");
    } else if (a == "--output-json") {
      opt.output_json = need("--output-json");
    } else if (a == "--output-csv") {
      opt.output_csv = need("--output-csv");
    } else {
      err = "unknown argument: " + a;
      return false;
    }
    if (!err.empty()) {
      return false;
    }
  }
  return true;
}

inline std::optional<KernelOp> parse_kernel(const std::string& s) {
  if (s == "vector_add") return KernelOp::VectorAdd;
  if (s == "matmul") return KernelOp::Matmul;
  if (s == "reduction" || s == "reduction_sum") return KernelOp::ReductionSum;
  if (s == "reduction_max") return KernelOp::ReductionMax;
  if (s == "transpose") return KernelOp::Transpose;
  return std::nullopt;
}

inline std::optional<Variant> parse_variant(const std::string& s) {
  if (s == "cpu_ref") return Variant::CpuRef;
  if (s == "hip_baseline" || s == "naive" || s == "baseline") return Variant::HipBaseline;
  if (s == "hip_tiled" || s == "tiled") return Variant::HipTiled;
  if (s == "hip_tiled_opt" || s == "tiled_opt") return Variant::HipTiledOptimized;
  if (s == "hip_hierarchical" || s == "hierarchical") return Variant::HipHierarchical;
  return std::nullopt;
}

}  // namespace gks
