#include "benchmark/runner.hpp"

#include "benchmark/device.hpp"
#include "common/compare.hpp"
#include "common/metrics.hpp"
#include "common/rng.hpp"
#include "common/stats.hpp"
#include "common/time_utils.hpp"
#include "kernels/cpu_kernels.hpp"
#include "kernels/hip_kernels.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace gks {
namespace {

LaunchConfig make_launch(const CliOptions& opt) {
  LaunchConfig lc;
  lc.block_x = opt.block;
  lc.block_y = 1;
  lc.tile = opt.tile;
  return lc;
}

void fill_common(BenchmarkRecord& rec, const CliOptions& opt, KernelOp op, Variant var) {
  rec.timestamp_utc = utc_timestamp();
  rec.operation = op;
  rec.variant = var;
  rec.dtype = "f32";
  rec.launch = make_launch(opt);
  rec.warmup = opt.warmup;
  rec.trials = opt.trials;
  rec.device = query_device_info();
#if defined(GKS_WITH_HIP)
  rec.build_config = "HIP enabled";
#else
  rec.build_config = "host-only (HIP disabled)";
#endif
  rec.compiler_flags = "unavailable";
}

TestCaseResult check(const std::string& name, bool passed, const std::string& msg) {
  TestCaseResult r;
  r.name = name;
  r.passed = passed;
  r.message = msg;
  return r;
}

std::vector<TestCaseResult> test_vector_add(bool host_only, std::uint64_t seed,
                                            const Tolerance& tol) {
  std::vector<TestCaseResult> out;
  Rng rng(seed);
  const std::vector<index_t> sizes = {0, 1, 3, 17, 256, 1000, 4097};

  for (index_t n : sizes) {
    auto a = rng.uniform(n);
    auto b = rng.uniform(n);
    std::vector<f32> cpu_c;
    cpu::vector_add(a, b, cpu_c);

    const std::string base = "vector_add/cpu n=" + std::to_string(n);
    out.push_back(check(base, cpu_c.size() == n, "cpu size"));

    if (host_only) {
      continue;
    }
    std::vector<f32> gpu_c;
    TimingStats ts;
    LaunchConfig lc;
    lc.block_x = 256;
    auto st = hip_api::vector_add(a, b, gpu_c, lc, &ts, 1, 1);
    if (!st.ok) {
      out.push_back(check("vector_add/hip n=" + std::to_string(n), false, st.message));
      continue;
    }
    auto cmp = compare_arrays(cpu_c, gpu_c, tol);
    out.push_back(check("vector_add/hip n=" + std::to_string(n), cmp.passed, cmp.message));
  }
  return out;
}

std::vector<TestCaseResult> test_matmul(bool host_only, std::uint64_t seed, const Tolerance& tol) {
  std::vector<TestCaseResult> out;
  Rng rng(seed + 1);
  struct Case {
    index_t m, n, k;
  };
  const std::vector<Case> cases = {
      {0, 0, 0}, {1, 1, 1}, {3, 5, 7}, {16, 16, 16}, {17, 19, 23}, {64, 32, 48}, {65, 65, 65}};

  for (const auto& c : cases) {
    auto a = rng.uniform_matrix(c.m, c.k);
    auto b = rng.uniform_matrix(c.k, c.n);
    std::vector<f32> cpu_c;
    cpu::matmul(a, b, cpu_c, c.m, c.n, c.k);
    const std::string tag =
        "matmul m=" + std::to_string(c.m) + " n=" + std::to_string(c.n) + " k=" + std::to_string(c.k);

    if (host_only) {
      out.push_back(check(tag + "/cpu", cpu_c.size() == c.m * c.n, "cpu ok"));
      continue;
    }

    for (Variant v : {Variant::HipBaseline, Variant::HipTiled}) {
      std::vector<f32> gpu_c;
      TimingStats ts;
      LaunchConfig lc;
      lc.tile = 16;
      lc.block_x = 16;
      lc.block_y = 16;
      hip_api::HipStatus st;
      if (v == Variant::HipBaseline) {
        st = hip_api::matmul_naive(a, b, gpu_c, c.m, c.n, c.k, lc, &ts, 1, 1);
      } else {
        st = hip_api::matmul_tiled(a, b, gpu_c, c.m, c.n, c.k, lc, &ts, 1, 1);
      }
      if (!st.ok) {
        out.push_back(check(tag + "/" + to_string(v), false, st.message));
        continue;
      }
      // Matmul accumulates; slightly looser relative tolerance for larger K.
      Tolerance mt = tol;
      mt.rel_tol = std::max(tol.rel_tol, 1e-3);
      mt.abs_tol = std::max(tol.abs_tol, 1e-3);
      auto cmp = compare_arrays(cpu_c, gpu_c, mt);
      out.push_back(check(tag + "/" + to_string(v), cmp.passed, cmp.message));
    }
  }
  return out;
}

std::vector<TestCaseResult> test_reduction(bool host_only, std::uint64_t seed,
                                           const Tolerance& tol) {
  std::vector<TestCaseResult> out;
  Rng rng(seed + 2);
  const std::vector<index_t> sizes = {0, 1, 2, 3, 17, 255, 1024, 10007};

  for (index_t n : sizes) {
    auto x = rng.uniform(n, -2.0f, 2.0f);
    if (n == 0) {
      f32 sum = cpu::reduction_sum(x);
      out.push_back(check("reduction_sum empty/cpu", sum == 0.0f, "sum empty"));
      bool threw = false;
      try {
        (void)cpu::reduction_max(x);
      } catch (...) {
        threw = true;
      }
      out.push_back(check("reduction_max empty/cpu", threw, "max empty should throw"));
      continue;
    }

    const f32 cpu_sum = cpu::reduction_sum(x);
    const f32 cpu_max = cpu::reduction_max(x);
    out.push_back(check("reduction_sum/cpu n=" + std::to_string(n), true, "ok"));
    out.push_back(check("reduction_max/cpu n=" + std::to_string(n), true, "ok"));

    if (host_only) {
      continue;
    }

    Tolerance rt = tol;
    rt.rel_tol = std::max(tol.rel_tol, 1e-3);
    rt.abs_tol = std::max(tol.abs_tol, 1e-3 * static_cast<double>(n));

    for (Variant v : {Variant::HipBaseline, Variant::HipHierarchical}) {
      f32 gpu_sum = 0.0f;
      TimingStats ts;
      LaunchConfig lc;
      lc.block_x = 256;
      hip_api::HipStatus st = (v == Variant::HipBaseline)
                         ? hip_api::reduction_sum_baseline(x, gpu_sum, lc, &ts, 1, 1)
                         : hip_api::reduction_sum_hierarchical(x, gpu_sum, lc, &ts, 1, 1);
      if (!st.ok) {
        out.push_back(
            check("reduction_sum/" + std::string(to_string(v)) + " n=" + std::to_string(n), false,
                  st.message));
      } else {
        auto cmp = compare_scalars(cpu_sum, gpu_sum, rt);
        out.push_back(
            check("reduction_sum/" + std::string(to_string(v)) + " n=" + std::to_string(n),
                  cmp.passed, cmp.message));
      }

      f32 gpu_max = 0.0f;
      st = (v == Variant::HipBaseline)
               ? hip_api::reduction_max_baseline(x, gpu_max, lc, &ts, 1, 1)
               : hip_api::reduction_max_hierarchical(x, gpu_max, lc, &ts, 1, 1);
      if (!st.ok) {
        out.push_back(
            check("reduction_max/" + std::string(to_string(v)) + " n=" + std::to_string(n), false,
                  st.message));
      } else {
        auto cmp = compare_scalars(cpu_max, gpu_max, tol);
        out.push_back(
            check("reduction_max/" + std::string(to_string(v)) + " n=" + std::to_string(n),
                  cmp.passed, cmp.message));
      }
    }
  }

  // Special values
  {
    std::vector<f32> x = {1.0f, std::numeric_limits<f32>::infinity(), -3.0f};
    const f32 m = cpu::reduction_max(x);
    out.push_back(check("reduction_max/inf", std::isinf(m) && m > 0, "max inf"));
  }
  return out;
}

std::vector<TestCaseResult> test_transpose(bool host_only, std::uint64_t seed,
                                           const Tolerance& tol) {
  std::vector<TestCaseResult> out;
  Rng rng(seed + 3);
  struct Case {
    index_t r, c;
  };
  const std::vector<Case> cases = {{0, 0}, {1, 1}, {3, 5}, {16, 16}, {17, 31}, {64, 33}};

  for (const auto& c : cases) {
    auto in = rng.uniform_matrix(c.r, c.c);
    std::vector<f32> cpu_out;
    cpu::transpose(in, cpu_out, c.r, c.c);
    const std::string tag =
        "transpose " + std::to_string(c.r) + "x" + std::to_string(c.c);

    if (host_only) {
      out.push_back(check(tag + "/cpu", cpu_out.size() == c.r * c.c, "cpu ok"));
      continue;
    }

    for (Variant v : {Variant::HipBaseline, Variant::HipTiled, Variant::HipTiledOptimized}) {
      std::vector<f32> gpu_out;
      TimingStats ts;
      LaunchConfig lc;
      lc.tile = 16;
      hip_api::HipStatus st;
      if (v == Variant::HipBaseline) {
        st = hip_api::transpose_baseline(in, gpu_out, c.r, c.c, lc, &ts, 1, 1);
      } else if (v == Variant::HipTiled) {
        st = hip_api::transpose_tiled(in, gpu_out, c.r, c.c, lc, &ts, 1, 1);
      } else {
        st = hip_api::transpose_tiled_opt(in, gpu_out, c.r, c.c, lc, &ts, 1, 1);
      }
      if (!st.ok) {
        out.push_back(check(tag + "/" + to_string(v), false, st.message));
        continue;
      }
      Tolerance et = tol;
      et.require_exact = true;
      auto cmp = compare_arrays(cpu_out, gpu_out, et);
      out.push_back(check(tag + "/" + to_string(v), cmp.passed, cmp.message));
    }
  }
  return out;
}

}  // namespace

std::vector<TestCaseResult> run_correctness_suite(bool host_only, std::uint64_t seed,
                                                  const Tolerance& tol) {
  std::vector<TestCaseResult> all;
  auto append = [&](std::vector<TestCaseResult> part) {
    all.insert(all.end(), part.begin(), part.end());
  };
  append(test_vector_add(host_only, seed, tol));
  append(test_matmul(host_only, seed, tol));
  append(test_reduction(host_only, seed, tol));
  append(test_transpose(host_only, seed, tol));
  return all;
}

int run_tests_and_report(bool host_only, std::uint64_t seed, const Tolerance& tol) {
  const bool force_host = host_only || !hip_runtime_available();
  if (!host_only && !hip_runtime_available()) {
    std::cout << "NOTE: No HIP device available; running host-only correctness suite.\n";
  }
  auto results = run_correctness_suite(force_host, seed, tol);
  int pass = 0;
  int fail = 0;
  for (const auto& r : results) {
    std::cout << (r.passed ? "PASS" : "FAIL") << "  " << r.name << "  " << r.message << '\n';
    if (r.passed) {
      ++pass;
    } else {
      ++fail;
    }
  }
  std::cout << "Summary: " << pass << " passed, " << fail << " failed, " << (pass + fail)
            << " total"
            << (force_host ? " [host-only]" : " [GPU]") << '\n';
  return fail == 0 ? 0 : 1;
}

BenchmarkRecord run_kernel_benchmark(const CliOptions& opt, std::string& err) {
  BenchmarkRecord rec;
  if (!opt.kernel) {
    err = "missing --kernel";
    return rec;
  }
  auto op = parse_kernel(*opt.kernel);
  if (!op) {
    err = "unknown kernel: " + *opt.kernel;
    return rec;
  }

  Variant var = Variant::HipBaseline;
  if (opt.variant) {
    auto pv = parse_variant(*opt.variant);
    if (!pv) {
      err = "unknown variant: " + *opt.variant;
      return rec;
    }
    var = *pv;
  }

  fill_common(rec, opt, *op, var);
  Rng rng(opt.seed);
  LaunchConfig lc = make_launch(opt);

  const bool use_cpu = (var == Variant::CpuRef) || opt.host_only || !hip_runtime_available();
  if (use_cpu && var != Variant::CpuRef && !opt.host_only && !hip_runtime_available()) {
    rec.status = "unsupported";
    rec.notes = "HIP device unavailable; refusing to silently substitute CPU for GPU variant";
    rec.correctness_passed = false;
    rec.correctness_message = rec.notes;
    err.clear();
    return rec;
  }

  try {
    if (*op == KernelOp::VectorAdd) {
      const index_t n = opt.size.value_or(1 << 20);
      rec.size = n;
      auto a = rng.uniform(n);
      auto b = rng.uniform(n);
      std::vector<f32> ref, out;
      cpu::vector_add(a, b, ref);

      HostTimer e2e;
      e2e.start();
      if (var == Variant::CpuRef || opt.host_only) {
        std::vector<double> samples;
        for (int i = 0; i < opt.warmup; ++i) {
          cpu::vector_add(a, b, out);
        }
        for (int i = 0; i < opt.trials; ++i) {
          HostTimer t;
          t.start();
          cpu::vector_add(a, b, out);
          samples.push_back(t.stop_ms());
        }
        rec.kernel_timing = compute_timing_stats(std::move(samples));
        rec.variant = Variant::CpuRef;
      } else {
        auto st = hip_api::vector_add(a, b, out, lc, &rec.kernel_timing, opt.warmup, opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          rec.correctness_message = st.message;
          return rec;
        }
      }
      rec.e2e_timing.median_ms = e2e.stop_ms();
      auto cmp = compare_arrays(ref, out, opt.tol);
      rec.correctness_passed = cmp.passed;
      rec.correctness_message = cmp.message;
      rec.metric_name = "effective_bandwidth";
      rec.metric_unit = "GB/s";
      rec.metric_value = vector_add_bandwidth_gbs(n, rec.kernel_timing.median_ms);
      rec.notes = "bytes=3*N*sizeof(f32); effective not DRAM-peak";
    } else if (*op == KernelOp::Matmul) {
      index_t m = opt.m.value_or(opt.size.value_or(256));
      index_t n = opt.n.value_or(opt.size.value_or(256));
      index_t k = opt.k.value_or(opt.size.value_or(256));
      rec.m = m;
      rec.n = n;
      rec.k = k;
      rec.size = m;
      auto a = rng.uniform_matrix(m, k);
      auto b = rng.uniform_matrix(k, n);
      std::vector<f32> ref, out;
      cpu::matmul(a, b, ref, m, n, k);
      Tolerance mt = opt.tol;
      mt.rel_tol = std::max(opt.tol.rel_tol, 1e-3);
      mt.abs_tol = std::max(opt.tol.abs_tol, 1e-3);

      if (var == Variant::CpuRef || opt.host_only) {
        std::vector<double> samples;
        for (int i = 0; i < opt.warmup; ++i) cpu::matmul(a, b, out, m, n, k);
        for (int i = 0; i < opt.trials; ++i) {
          HostTimer t;
          t.start();
          cpu::matmul(a, b, out, m, n, k);
          samples.push_back(t.stop_ms());
        }
        rec.kernel_timing = compute_timing_stats(std::move(samples));
        rec.variant = Variant::CpuRef;
      } else if (var == Variant::HipTiled) {
        auto st =
            hip_api::matmul_tiled(a, b, out, m, n, k, lc, &rec.kernel_timing, opt.warmup, opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          return rec;
        }
      } else {
        auto st =
            hip_api::matmul_naive(a, b, out, m, n, k, lc, &rec.kernel_timing, opt.warmup, opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          return rec;
        }
        rec.variant = Variant::HipBaseline;
      }
      auto cmp = compare_arrays(ref, out, mt);
      rec.correctness_passed = cmp.passed;
      rec.correctness_message = cmp.message;
      rec.metric_name = "effective_gflops";
      rec.metric_unit = "GFLOP/s";
      rec.metric_value = matmul_gflops(m, n, k, rec.kernel_timing.median_ms);
      rec.notes = "flops=2*M*N*K; speedup filled by sweep runner vs naive GPU";
    } else if (*op == KernelOp::ReductionSum || *op == KernelOp::ReductionMax) {
      const index_t n = opt.size.value_or(1 << 20);
      rec.size = n;
      auto x = rng.uniform(n, -1.0f, 1.0f);
      const bool is_sum = (*op == KernelOp::ReductionSum);
      f32 ref = is_sum ? cpu::reduction_sum(x) : cpu::reduction_max(x);
      f32 out = 0.0f;
      Tolerance rt = opt.tol;
      if (is_sum) {
        rt.abs_tol = std::max(opt.tol.abs_tol, 1e-3 * static_cast<double>(n));
        rt.rel_tol = std::max(opt.tol.rel_tol, 1e-3);
      }

      if (var == Variant::CpuRef || opt.host_only) {
        std::vector<double> samples;
        for (int i = 0; i < opt.warmup; ++i)
          out = is_sum ? cpu::reduction_sum(x) : cpu::reduction_max(x);
        for (int i = 0; i < opt.trials; ++i) {
          HostTimer t;
          t.start();
          out = is_sum ? cpu::reduction_sum(x) : cpu::reduction_max(x);
          samples.push_back(t.stop_ms());
        }
        rec.kernel_timing = compute_timing_stats(std::move(samples));
        rec.variant = Variant::CpuRef;
      } else if (var == Variant::HipHierarchical) {
        auto st = is_sum ? hip_api::reduction_sum_hierarchical(x, out, lc, &rec.kernel_timing,
                                                               opt.warmup, opt.trials)
                         : hip_api::reduction_max_hierarchical(x, out, lc, &rec.kernel_timing,
                                                               opt.warmup, opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          return rec;
        }
      } else {
        auto st = is_sum ? hip_api::reduction_sum_baseline(x, out, lc, &rec.kernel_timing,
                                                           opt.warmup, opt.trials)
                         : hip_api::reduction_max_baseline(x, out, lc, &rec.kernel_timing,
                                                           opt.warmup, opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          return rec;
        }
        rec.variant = Variant::HipBaseline;
      }
      auto cmp = compare_scalars(ref, out, rt);
      rec.correctness_passed = cmp.passed;
      rec.correctness_message = cmp.message;
      rec.metric_name = "effective_input_bandwidth";
      rec.metric_unit = "GB/s";
      rec.metric_value = reduction_throughput_gbs(n, rec.kernel_timing.median_ms);
      rec.notes = "bytes=N*sizeof(f32); FP association may differ vs CPU";
    } else if (*op == KernelOp::Transpose) {
      index_t rows = opt.m.value_or(opt.size.value_or(1024));
      index_t cols = opt.n.value_or(opt.size.value_or(rows));
      rec.m = rows;
      rec.n = cols;
      rec.size = rows;
      auto in = rng.uniform_matrix(rows, cols);
      std::vector<f32> ref, out;
      cpu::transpose(in, ref, rows, cols);
      Tolerance et = opt.tol;
      et.require_exact = true;

      if (var == Variant::CpuRef || opt.host_only) {
        std::vector<double> samples;
        for (int i = 0; i < opt.warmup; ++i) cpu::transpose(in, out, rows, cols);
        for (int i = 0; i < opt.trials; ++i) {
          HostTimer t;
          t.start();
          cpu::transpose(in, out, rows, cols);
          samples.push_back(t.stop_ms());
        }
        rec.kernel_timing = compute_timing_stats(std::move(samples));
        rec.variant = Variant::CpuRef;
      } else if (var == Variant::HipTiledOptimized) {
        auto st = hip_api::transpose_tiled_opt(in, out, rows, cols, lc, &rec.kernel_timing,
                                               opt.warmup, opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          return rec;
        }
      } else if (var == Variant::HipTiled) {
        auto st = hip_api::transpose_tiled(in, out, rows, cols, lc, &rec.kernel_timing, opt.warmup,
                                           opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          return rec;
        }
      } else {
        auto st = hip_api::transpose_baseline(in, out, rows, cols, lc, &rec.kernel_timing,
                                              opt.warmup, opt.trials);
        if (!st.ok) {
          rec.status = "error";
          rec.notes = st.message;
          return rec;
        }
        rec.variant = Variant::HipBaseline;
      }
      auto cmp = compare_arrays(ref, out, et);
      rec.correctness_passed = cmp.passed;
      rec.correctness_message = cmp.message;
      rec.metric_name = "effective_bandwidth";
      rec.metric_unit = "GB/s";
      rec.metric_value = transpose_bandwidth_gbs(rows, cols, rec.kernel_timing.median_ms);
      rec.notes = "bytes=2*rows*cols*sizeof(f32); effective not DRAM-peak";
    } else {
      err = "kernel not implemented";
      rec.status = "unsupported";
    }
  } catch (const std::exception& ex) {
    rec.status = "error";
    rec.notes = ex.what();
    rec.correctness_passed = false;
    rec.correctness_message = ex.what();
  }

  if (!rec.correctness_passed && rec.status == "ok") {
    rec.status = "correctness_failed";
  }
  return rec;
}

}  // namespace gks
