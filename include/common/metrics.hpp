#pragma once

#include "common/types.hpp"

namespace gks {

// Vector add: read A, read B, write C => 3 * N * sizeof(f32) bytes.
inline double vector_add_bytes(index_t n) {
  return 3.0 * static_cast<double>(n) * static_cast<double>(sizeof(f32));
}

inline double vector_add_bandwidth_gbs(index_t n, double median_ms) {
  if (median_ms <= 0.0) return 0.0;
  return (vector_add_bytes(n) / 1e9) / (median_ms / 1e3);
}

// Matmul FLOPs: 2*M*N*K
inline double matmul_flops(index_t m, index_t n, index_t k) {
  return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

inline double matmul_gflops(index_t m, index_t n, index_t k, double median_ms) {
  if (median_ms <= 0.0) return 0.0;
  return (matmul_flops(m, n, k) / 1e9) / (median_ms / 1e3);
}

// Reduction: one read of N elements.
inline double reduction_bytes(index_t n) {
  return static_cast<double>(n) * static_cast<double>(sizeof(f32));
}

inline double reduction_throughput_gbs(index_t n, double median_ms) {
  if (median_ms <= 0.0) return 0.0;
  return (reduction_bytes(n) / 1e9) / (median_ms / 1e3);
}

// Transpose: read + write => 2 * rows * cols * sizeof(f32)
inline double transpose_bytes(index_t rows, index_t cols) {
  return 2.0 * static_cast<double>(rows) * static_cast<double>(cols) *
         static_cast<double>(sizeof(f32));
}

inline double transpose_bandwidth_gbs(index_t rows, index_t cols, double median_ms) {
  if (median_ms <= 0.0) return 0.0;
  return (transpose_bytes(rows, cols) / 1e9) / (median_ms / 1e3);
}

}  // namespace gks
