#include "kernels/hip_kernels.hpp"

namespace gks::hip_api {
namespace {

HipStatus unavailable(const char* name) {
  HipStatus st;
  st.ok = false;
  st.message = std::string(name) +
               ": HIP runtime not available in this build. "
               "Configure with ROCm/HIP on an AMD GPU host "
               "(-DGKS_ENABLE_HIP=ON) or use --host-only for CPU paths.";
  return st;
}

}  // namespace

HipStatus vector_add(const std::vector<f32>&, const std::vector<f32>&, std::vector<f32>&,
                     const LaunchConfig&, TimingStats*, int, int) {
  return unavailable("vector_add");
}

HipStatus matmul_naive(const std::vector<f32>&, const std::vector<f32>&, std::vector<f32>&,
                       index_t, index_t, index_t, const LaunchConfig&, TimingStats*, int, int) {
  return unavailable("matmul_naive");
}

HipStatus matmul_tiled(const std::vector<f32>&, const std::vector<f32>&, std::vector<f32>&,
                       index_t, index_t, index_t, const LaunchConfig&, TimingStats*, int, int) {
  return unavailable("matmul_tiled");
}

HipStatus reduction_sum_baseline(const std::vector<f32>&, f32&, const LaunchConfig&, TimingStats*,
                                 int, int) {
  return unavailable("reduction_sum_baseline");
}

HipStatus reduction_sum_hierarchical(const std::vector<f32>&, f32&, const LaunchConfig&,
                                     TimingStats*, int, int) {
  return unavailable("reduction_sum_hierarchical");
}

HipStatus reduction_max_baseline(const std::vector<f32>&, f32&, const LaunchConfig&, TimingStats*,
                                 int, int) {
  return unavailable("reduction_max_baseline");
}

HipStatus reduction_max_hierarchical(const std::vector<f32>&, f32&, const LaunchConfig&,
                                     TimingStats*, int, int) {
  return unavailable("reduction_max_hierarchical");
}

HipStatus transpose_baseline(const std::vector<f32>&, std::vector<f32>&, index_t, index_t,
                             const LaunchConfig&, TimingStats*, int, int) {
  return unavailable("transpose_baseline");
}

HipStatus transpose_tiled(const std::vector<f32>&, std::vector<f32>&, index_t, index_t,
                          const LaunchConfig&, TimingStats*, int, int) {
  return unavailable("transpose_tiled");
}

HipStatus transpose_tiled_opt(const std::vector<f32>&, std::vector<f32>&, index_t, index_t,
                              const LaunchConfig&, TimingStats*, int, int) {
  return unavailable("transpose_tiled_opt");
}

}  // namespace gks::hip_api
