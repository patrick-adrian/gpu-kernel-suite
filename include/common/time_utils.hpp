#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace gks {

inline std::string utc_timestamp() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const std::time_t t = clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  std::ostringstream o;
  o << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return o.str();
}

inline double host_now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

class HostTimer {
 public:
  void start() { t0_ = host_now_ms(); }
  double stop_ms() const { return host_now_ms() - t0_; }

 private:
  double t0_ = 0.0;
};

}  // namespace gks
