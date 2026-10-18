#include "benchmark/device.hpp"
#include "benchmark/runner.hpp"
#include "common/cli.hpp"
#include "common/result_io.hpp"

#include <iostream>

int main(int argc, char** argv) {
  gks::CliOptions opt;
  std::string err;
  if (!gks::parse_cli(argc, argv, opt, err)) {
    std::cerr << "error: " << err << "\n\n";
    gks::print_help(argv[0]);
    return 2;
  }
  if (opt.show_help || argc == 1) {
    gks::print_help(argv[0]);
    return argc == 1 ? 2 : 0;
  }
  if (opt.list_kernels) {
    std::cout << "Kernels:\n"
              << "  vector_add\n"
              << "  matmul\n"
              << "  reduction_sum (alias: reduction)\n"
              << "  reduction_max\n"
              << "  transpose\n"
              << "Variants:\n"
              << "  cpu_ref\n"
              << "  hip_baseline | naive | baseline\n"
              << "  hip_tiled | tiled\n"
              << "  hip_tiled_opt | tiled_opt\n"
              << "  hip_hierarchical | hierarchical\n";
    const auto dev = gks::query_device_info();
    std::cout << "Device: " << dev.name << " arch=" << dev.arch
              << " gpu_available=" << (dev.gpu_available ? "true" : "false") << '\n';
    return 0;
  }
  if (opt.run_tests) {
    return gks::run_tests_and_report(opt.host_only, opt.seed, opt.tol);
  }
  if (!opt.kernel) {
    std::cerr << "error: specify --kernel, --test, or --list\n";
    gks::print_help(argv[0]);
    return 2;
  }

  auto rec = gks::run_kernel_benchmark(opt, err);
  if (!err.empty()) {
    std::cerr << "error: " << err << '\n';
    return 2;
  }

  std::cout << "operation=" << gks::to_string(rec.operation)
            << " variant=" << gks::to_string(rec.variant) << " status=" << rec.status << '\n'
            << "correctness=" << (rec.correctness_passed ? "PASS" : "FAIL") << " "
            << rec.correctness_message << '\n'
            << "median_ms=" << rec.kernel_timing.median_ms
            << " mean_ms=" << rec.kernel_timing.mean_ms
            << " stddev_ms=" << rec.kernel_timing.stddev_ms << '\n'
            << "metric=" << rec.metric_name << " value=" << rec.metric_value << " "
            << rec.metric_unit << '\n'
            << "device=" << rec.device.name << " arch=" << rec.device.arch << '\n';

  if (!opt.output_json.empty()) {
    std::string ioerr;
    if (!gks::write_json(opt.output_json, rec, ioerr)) {
      std::cerr << "error: " << ioerr << '\n';
      return 1;
    }
    std::cout << "wrote " << opt.output_json << '\n';
  }
  if (!opt.output_csv.empty()) {
    std::string ioerr;
    if (!gks::append_csv(opt.output_csv, rec, ioerr)) {
      std::cerr << "error: " << ioerr << '\n';
      return 1;
    }
    std::cout << "wrote " << opt.output_csv << '\n';
  }

  if (rec.status != "ok" || !rec.correctness_passed) {
    return 1;
  }
  return 0;
}
