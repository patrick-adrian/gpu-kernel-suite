#pragma once

#include "common/types.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace gks {

class Rng {
 public:
  explicit Rng(std::uint64_t seed = 42) : eng_(static_cast<std::mt19937_64::result_type>(seed)) {}

  std::vector<f32> uniform(index_t n, f32 low = -1.0f, f32 high = 1.0f) {
    std::uniform_real_distribution<f32> dist(low, high);
    std::vector<f32> out(n);
    for (index_t i = 0; i < n; ++i) {
      out[i] = dist(eng_);
    }
    return out;
  }

  std::vector<f32> uniform_matrix(index_t rows, index_t cols, f32 low = -1.0f, f32 high = 1.0f) {
    return uniform(rows * cols, low, high);
  }

 private:
  std::mt19937_64 eng_;
};

}  // namespace gks
