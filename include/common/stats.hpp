#pragma once

#include "common/types.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gks {

inline TimingStats compute_timing_stats(std::vector<double> samples_ms) {
  TimingStats stats;
  stats.samples_ms = std::move(samples_ms);
  if (stats.samples_ms.empty()) {
    return stats;
  }

  auto sorted = stats.samples_ms;
  std::sort(sorted.begin(), sorted.end());
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  stats.mean_ms =
      std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());

  const std::size_t n = sorted.size();
  if (n % 2 == 1) {
    stats.median_ms = sorted[n / 2];
  } else {
    stats.median_ms = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  }

  if (n > 1) {
    double acc = 0.0;
    for (double v : sorted) {
      const double d = v - stats.mean_ms;
      acc += d * d;
    }
    stats.stddev_ms = std::sqrt(acc / static_cast<double>(n - 1));
  }
  return stats;
}

inline bool is_noisy(const TimingStats& stats, double cv_threshold = 0.15) {
  if (stats.mean_ms <= 0.0) {
    return false;
  }
  return (stats.stddev_ms / stats.mean_ms) > cv_threshold;
}

}  // namespace gks
