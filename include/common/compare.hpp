#pragma once

#include "common/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace gks {

inline bool nearly_equal(f32 a, f32 b, const Tolerance& tol) {
  if (tol.require_exact) {
    return a == b;
  }
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::isinf(a) || std::isinf(b)) {
    return a == b;
  }
  const double abs_err = std::abs(static_cast<double>(a) - static_cast<double>(b));
  const double denom = std::max(std::abs(static_cast<double>(a)),
                                std::abs(static_cast<double>(b)));
  const double rel_err = (denom > 0.0) ? (abs_err / denom) : abs_err;
  return abs_err <= tol.abs_tol || rel_err <= tol.rel_tol;
}

inline CompareResult compare_arrays(const std::vector<f32>& expected,
                                    const std::vector<f32>& actual,
                                    const Tolerance& tol) {
  CompareResult result;
  if (expected.size() != actual.size()) {
    result.passed = false;
    std::ostringstream oss;
    oss << "size mismatch: expected " << expected.size() << " got " << actual.size();
    result.message = oss.str();
    return result;
  }

  double max_abs = 0.0;
  double max_rel = 0.0;
  index_t first = static_cast<index_t>(-1);

  for (index_t i = 0; i < expected.size(); ++i) {
    const double abs_err =
        std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    const double denom = std::max(std::abs(static_cast<double>(expected[i])),
                                  std::abs(static_cast<double>(actual[i])));
    const double rel_err = (denom > 0.0) ? (abs_err / denom) : abs_err;
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
    if (first == static_cast<index_t>(-1) && !nearly_equal(expected[i], actual[i], tol)) {
      first = i;
    }
  }

  result.max_abs_err = max_abs;
  result.max_rel_err = max_rel;
  result.first_mismatch = first;
  result.passed = (first == static_cast<index_t>(-1));
  if (result.passed) {
    result.message = "PASS";
  } else {
    std::ostringstream oss;
    oss << "FAIL at index " << first << " expected=" << expected[first]
        << " actual=" << actual[first] << " max_abs=" << max_abs
        << " max_rel=" << max_rel;
    result.message = oss.str();
  }
  return result;
}

inline CompareResult compare_scalars(f32 expected, f32 actual, const Tolerance& tol) {
  CompareResult result;
  const double abs_err =
      std::abs(static_cast<double>(expected) - static_cast<double>(actual));
  const double denom =
      std::max(std::abs(static_cast<double>(expected)), std::abs(static_cast<double>(actual)));
  const double rel_err = (denom > 0.0) ? (abs_err / denom) : abs_err;
  result.max_abs_err = abs_err;
  result.max_rel_err = rel_err;
  result.passed = nearly_equal(expected, actual, tol);
  if (result.passed) {
    result.message = "PASS";
  } else {
    std::ostringstream oss;
    oss << "FAIL expected=" << expected << " actual=" << actual
        << " abs=" << abs_err << " rel=" << rel_err;
    result.message = oss.str();
  }
  return result;
}

}  // namespace gks
