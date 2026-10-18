#pragma once

#include "common/types.hpp"

namespace gks {

// Populated from HIP when GKS_WITH_HIP=1; otherwise returns unavailable metadata.
DeviceInfo query_device_info();

bool hip_runtime_available();

}  // namespace gks
