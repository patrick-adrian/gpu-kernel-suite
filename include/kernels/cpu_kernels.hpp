#pragma once

#include "common/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gks::cpu {

inline void vector_add(const std::vector<f32>& a, const std::vector<f32>& b,
                       std::vector<f32>& c) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_add: size mismatch");
  }
  c.resize(a.size());
  for (index_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
}

// Row-major C = A(m,k) * B(k,n)
inline void matmul(const std::vector<f32>& a, const std::vector<f32>& b, std::vector<f32>& c,
                   index_t m, index_t n, index_t k) {
  if (a.size() != m * k || b.size() != k * n) {
    throw std::invalid_argument("matmul: dimension/buffer mismatch");
  }
  c.assign(m * n, 0.0f);
  for (index_t i = 0; i < m; ++i) {
    for (index_t p = 0; p < k; ++p) {
      const f32 aik = a[i * k + p];
      for (index_t j = 0; j < n; ++j) {
        c[i * n + j] += aik * b[p * n + j];
      }
    }
  }
}

inline f32 reduction_sum(const std::vector<f32>& x) {
  double acc = 0.0;
  for (f32 v : x) {
    acc += static_cast<double>(v);
  }
  return static_cast<f32>(acc);
}

inline f32 reduction_max(const std::vector<f32>& x) {
  if (x.empty()) {
    throw std::invalid_argument("reduction_max: empty input");
  }
  f32 m = x[0];
  for (index_t i = 1; i < x.size(); ++i) {
    m = std::max(m, x[i]);
  }
  return m;
}

// Out is cols x rows (transposed of rows x cols input).
inline void transpose(const std::vector<f32>& in, std::vector<f32>& out, index_t rows,
                      index_t cols) {
  if (in.size() != rows * cols) {
    throw std::invalid_argument("transpose: dimension/buffer mismatch");
  }
  out.resize(rows * cols);
  for (index_t i = 0; i < rows; ++i) {
    for (index_t j = 0; j < cols; ++j) {
      out[j * rows + i] = in[i * cols + j];
    }
  }
}

}  // namespace gks::cpu
