#pragma once

#include "common/types.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace gks {

inline std::string escape_json(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\n': o << "\\n"; break;
      default: o << c; break;
    }
  }
  return o.str();
}

inline std::string samples_to_json(const std::vector<double>& v) {
  std::ostringstream o;
  o << '[';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) o << ',';
    o << std::setprecision(17) << v[i];
  }
  o << ']';
  return o.str();
}

inline std::string timing_to_json(const TimingStats& t) {
  std::ostringstream o;
  o << '{'
    << "\"samples_ms\":" << samples_to_json(t.samples_ms) << ','
    << "\"median_ms\":" << t.median_ms << ','
    << "\"mean_ms\":" << t.mean_ms << ','
    << "\"stddev_ms\":" << t.stddev_ms << ','
    << "\"min_ms\":" << t.min_ms << ','
    << "\"max_ms\":" << t.max_ms
    << '}';
  return o.str();
}

inline std::string record_to_json(const BenchmarkRecord& r) {
  std::ostringstream o;
  o << std::setprecision(17);
  o << "{\n"
    << "  \"timestamp_utc\": \"" << escape_json(r.timestamp_utc) << "\",\n"
    << "  \"operation\": \"" << to_string(r.operation) << "\",\n"
    << "  \"variant\": \"" << to_string(r.variant) << "\",\n"
    << "  \"dtype\": \"" << escape_json(r.dtype) << "\",\n"
    << "  \"size\": " << r.size << ",\n"
    << "  \"m\": " << r.m << ",\n"
    << "  \"n\": " << r.n << ",\n"
    << "  \"k\": " << r.k << ",\n"
    << "  \"launch\": {\"block_x\": " << r.launch.block_x
    << ", \"block_y\": " << r.launch.block_y
    << ", \"tile\": " << r.launch.tile << "},\n"
    << "  \"warmup\": " << r.warmup << ",\n"
    << "  \"trials\": " << r.trials << ",\n"
    << "  \"kernel_timing\": " << timing_to_json(r.kernel_timing) << ",\n"
    << "  \"e2e_timing\": " << timing_to_json(r.e2e_timing) << ",\n"
    << "  \"correctness_passed\": " << (r.correctness_passed ? "true" : "false") << ",\n"
    << "  \"correctness_message\": \"" << escape_json(r.correctness_message) << "\",\n"
    << "  \"metric_value\": " << r.metric_value << ",\n"
    << "  \"metric_name\": \"" << escape_json(r.metric_name) << "\",\n"
    << "  \"metric_unit\": \"" << escape_json(r.metric_unit) << "\",\n"
    << "  \"speedup_vs_baseline\": " << r.speedup_vs_baseline << ",\n"
    << "  \"device\": {\n"
    << "    \"name\": \"" << escape_json(r.device.name) << "\",\n"
    << "    \"arch\": \"" << escape_json(r.device.arch) << "\",\n"
    << "    \"device_count\": " << r.device.device_count << ",\n"
    << "    \"global_mem_bytes\": " << r.device.global_mem_bytes << ",\n"
    << "    \"hip_runtime\": \"" << escape_json(r.device.hip_runtime) << "\",\n"
    << "    \"hip_compiler\": \"" << escape_json(r.device.hip_compiler) << "\",\n"
    << "    \"rocm_version\": \"" << escape_json(r.device.rocm_version) << "\",\n"
    << "    \"gpu_available\": " << (r.device.gpu_available ? "true" : "false") << "\n"
    << "  },\n"
    << "  \"build_config\": \"" << escape_json(r.build_config) << "\",\n"
    << "  \"compiler_flags\": \"" << escape_json(r.compiler_flags) << "\",\n"
    << "  \"status\": \"" << escape_json(r.status) << "\",\n"
    << "  \"notes\": \"" << escape_json(r.notes) << "\"\n"
    << "}\n";
  return o.str();
}

inline bool write_json(const std::string& path, const BenchmarkRecord& r, std::string& err) {
  std::ofstream out(path);
  if (!out) {
    err = "failed to open JSON output: " + path;
    return false;
  }
  out << record_to_json(r);
  return true;
}

inline std::string csv_header() {
  return "timestamp_utc,operation,variant,dtype,size,m,n,k,block_x,block_y,tile,"
         "warmup,trials,median_ms,mean_ms,stddev_ms,min_ms,max_ms,"
         "e2e_median_ms,correctness_passed,metric_name,metric_value,metric_unit,"
         "speedup_vs_baseline,device_name,device_arch,gpu_available,status,notes,"
         "raw_samples_ms";
}

inline std::string samples_to_csv_field(const std::vector<double>& v) {
  std::ostringstream o;
  o << '"';
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) o << ';';
    o << std::setprecision(17) << v[i];
  }
  o << '"';
  return o.str();
}

inline bool append_csv(const std::string& path, const BenchmarkRecord& r, std::string& err) {
  const bool write_header = [&]() {
    std::ifstream in(path);
    return !in.good() || in.peek() == std::ifstream::traits_type::eof();
  }();

  std::ofstream out(path, std::ios::app);
  if (!out) {
    err = "failed to open CSV output: " + path;
    return false;
  }
  if (write_header) {
    out << csv_header() << '\n';
  }
  out << std::setprecision(17)
      << r.timestamp_utc << ','
      << to_string(r.operation) << ','
      << to_string(r.variant) << ','
      << r.dtype << ','
      << r.size << ','
      << r.m << ','
      << r.n << ','
      << r.k << ','
      << r.launch.block_x << ','
      << r.launch.block_y << ','
      << r.launch.tile << ','
      << r.warmup << ','
      << r.trials << ','
      << r.kernel_timing.median_ms << ','
      << r.kernel_timing.mean_ms << ','
      << r.kernel_timing.stddev_ms << ','
      << r.kernel_timing.min_ms << ','
      << r.kernel_timing.max_ms << ','
      << r.e2e_timing.median_ms << ','
      << (r.correctness_passed ? "true" : "false") << ','
      << r.metric_name << ','
      << r.metric_value << ','
      << r.metric_unit << ','
      << r.speedup_vs_baseline << ','
      << '"' << r.device.name << '"' << ','
      << '"' << r.device.arch << '"' << ','
      << (r.device.gpu_available ? "true" : "false") << ','
      << r.status << ','
      << '"' << r.notes << '"' << ','
      << samples_to_csv_field(r.kernel_timing.samples_ms)
      << '\n';
  return true;
}

}  // namespace gks
