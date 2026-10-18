#pragma once

#include "common/cli.hpp"
#include "common/types.hpp"

#include <string>
#include <vector>

namespace gks {

struct TestCaseResult {
  std::string name;
  bool passed = false;
  std::string message;
};

std::vector<TestCaseResult> run_correctness_suite(bool host_only, std::uint64_t seed,
                                                  const Tolerance& tol);

BenchmarkRecord run_kernel_benchmark(const CliOptions& opt, std::string& err);

int run_tests_and_report(bool host_only, std::uint64_t seed, const Tolerance& tol);

}  // namespace gks
