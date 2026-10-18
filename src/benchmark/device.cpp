#include "benchmark/device.hpp"

#include <cstdlib>
#include <string>

#if defined(GKS_WITH_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gks {

bool hip_runtime_available() {
#if defined(GKS_WITH_HIP)
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    return false;
  }
  return count > 0;
#else
  return false;
#endif
}

DeviceInfo query_device_info() {
  DeviceInfo info;
#if defined(GKS_WITH_HIP)
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess || count <= 0) {
    info.name = "unavailable";
    info.arch = "unavailable";
    info.gpu_available = false;
    info.device_count = 0;
    info.hip_runtime = "HIP linked; no device";
    return info;
  }
  info.device_count = count;
  info.gpu_available = true;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
    info.name = prop.name ? prop.name : "unknown";
    info.global_mem_bytes = prop.totalGlobalMem;
    // Prefer gcnArchName when the ROCm headers expose it.
    info.arch = "unavailable";
#if defined(__HIP_PLATFORM_AMD__)
    if (prop.gcnArchName && prop.gcnArchName[0] != '\0') {
      info.arch = prop.gcnArchName;
    }
#endif
  }
  int runtime_ver = 0;
  if (hipRuntimeGetVersion(&runtime_ver) == hipSuccess) {
    info.hip_runtime = std::to_string(runtime_ver);
  }
  if (const char* rocm = std::getenv("ROCM_PATH")) {
    info.rocm_version = rocm;  // path recorded; version string may be unavailable
  }
  if (const char* hipcc = std::getenv("HIP_COMPILER")) {
    info.hip_compiler = hipcc;
  } else {
    info.hip_compiler = "hipcc (version unavailable)";
  }
#else
  info.name = "unavailable";
  info.arch = "unavailable";
  info.gpu_available = false;
  info.device_count = 0;
  info.hip_runtime = "unavailable (host-only build)";
  info.hip_compiler = "unavailable";
  info.rocm_version = "unavailable";
#endif
  return info;
}

}  // namespace gks
