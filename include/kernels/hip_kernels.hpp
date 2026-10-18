#pragma once

#include "common/types.hpp"

#include <string>
#include <vector>

namespace gks::hip_api {

struct HipStatus {
  bool ok = false;
  std::string message;
};

// Vector addition: c = a + b
HipStatus vector_add(const std::vector<f32>& a, const std::vector<f32>& b, std::vector<f32>& c,
                     const LaunchConfig& launch, TimingStats* kernel_ms, int warmup, int trials);

// Matmul variants
HipStatus matmul_naive(const std::vector<f32>& a, const std::vector<f32>& b, std::vector<f32>& c,
                       index_t m, index_t n, index_t k, const LaunchConfig& launch,
                       TimingStats* kernel_ms, int warmup, int trials);

HipStatus matmul_tiled(const std::vector<f32>& a, const std::vector<f32>& b, std::vector<f32>& c,
                       index_t m, index_t n, index_t k, const LaunchConfig& launch,
                       TimingStats* kernel_ms, int warmup, int trials);

// Reductions
HipStatus reduction_sum_baseline(const std::vector<f32>& x, f32& out, const LaunchConfig& launch,
                                 TimingStats* kernel_ms, int warmup, int trials);

HipStatus reduction_sum_hierarchical(const std::vector<f32>& x, f32& out,
                                     const LaunchConfig& launch, TimingStats* kernel_ms,
                                     int warmup, int trials);

HipStatus reduction_max_baseline(const std::vector<f32>& x, f32& out, const LaunchConfig& launch,
                                 TimingStats* kernel_ms, int warmup, int trials);

HipStatus reduction_max_hierarchical(const std::vector<f32>& x, f32& out,
                                     const LaunchConfig& launch, TimingStats* kernel_ms,
                                     int warmup, int trials);

// Transpose variants
HipStatus transpose_baseline(const std::vector<f32>& in, std::vector<f32>& out, index_t rows,
                             index_t cols, const LaunchConfig& launch, TimingStats* kernel_ms,
                             int warmup, int trials);

HipStatus transpose_tiled(const std::vector<f32>& in, std::vector<f32>& out, index_t rows,
                          index_t cols, const LaunchConfig& launch, TimingStats* kernel_ms,
                          int warmup, int trials);

HipStatus transpose_tiled_opt(const std::vector<f32>& in, std::vector<f32>& out, index_t rows,
                              index_t cols, const LaunchConfig& launch, TimingStats* kernel_ms,
                              int warmup, int trials);

}  // namespace gks::hip_api
