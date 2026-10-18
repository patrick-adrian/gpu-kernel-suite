#pragma once

#if defined(GKS_WITH_HIP)

#include <hip/hip_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace gks::hip_detail {

inline void check(hipError_t err, const char* expr, const char* file, int line) {
  if (err != hipSuccess) {
    std::ostringstream oss;
    oss << "HIP error " << hipGetErrorString(err) << " at " << file << ":" << line
        << " in " << expr;
    throw std::runtime_error(oss.str());
  }
}

}  // namespace gks::hip_detail

#define GKS_HIP_CHECK(expr) \
  ::gks::hip_detail::check((expr), #expr, __FILE__, __LINE__)

#endif  // GKS_WITH_HIP
