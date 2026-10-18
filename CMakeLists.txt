cmake_minimum_required(VERSION 3.21)

project(gpu_kernel_suite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GKS_ENABLE_HIP "Build HIP GPU kernels (requires ROCm/HIP)" ON)
set(GKS_GPU_TARGETS "" CACHE STRING
    "Optional AMDGPU targets for hipcc, e.g. gfx90a;gfx942. Empty = toolchain default.")

set(GKS_COMMON_SOURCES
    src/benchmark/device.cpp
    src/benchmark/runner.cpp
    src/benchmark/main.cpp
)

add_library(gks_headers INTERFACE)
target_include_directories(gks_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hip
)

set(GKS_HIP_FOUND FALSE)
if(GKS_ENABLE_HIP)
  # Prefer ROCm CMake package when present.
  if(NOT DEFINED ROCM_PATH AND DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  endif()
  if(DEFINED ROCM_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
  endif()

  find_program(HIPCC_EXECUTABLE NAMES hipcc
               HINTS "${ROCM_PATH}/bin" ENV ROCM_PATH ENV HIP_PATH
               PATH_SUFFIXES bin)

  if(HIPCC_EXECUTABLE)
    set(GKS_HIP_FOUND TRUE)
    message(STATUS "Found hipcc: ${HIPCC_EXECUTABLE}")
  else()
    message(WARNING
      "GKS_ENABLE_HIP=ON but hipcc was not found. "
      "Building host-only binary with HIP stubs. "
      "On AMD Developer Cloud: module load rocm (or equivalent), then reconfigure.")
  endif()
endif()

add_executable(gpu_suite ${GKS_COMMON_SOURCES})
target_link_libraries(gpu_suite PRIVATE gks_headers)

if(GKS_HIP_FOUND)
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  set_source_files_properties(src/hip/kernels.hip PROPERTIES LANGUAGE HIP)
  target_sources(gpu_suite PRIVATE src/hip/kernels.hip)
  target_compile_definitions(gpu_suite PRIVATE GKS_WITH_HIP=1)
  find_package(hip QUIET CONFIG)
  if(hip_FOUND)
    target_link_libraries(gpu_suite PRIVATE hip::host)
  endif()
  if(GKS_GPU_TARGETS)
    set_property(TARGET gpu_suite PROPERTY HIP_ARCHITECTURES ${GKS_GPU_TARGETS})
    message(STATUS "HIP_ARCHITECTURES=${GKS_GPU_TARGETS}")
  else()
    message(STATUS "HIP_ARCHITECTURES not set; using compiler/toolchain default")
  endif()
else()
  target_sources(gpu_suite PRIVATE src/hip/kernels_stub.cpp)
  message(STATUS "Building gpu_suite in host-only mode (no HIP device code)")
endif()

install(TARGETS gpu_suite RUNTIME DESTINATION bin)

enable_testing()
add_test(NAME host_correctness
         COMMAND gpu_suite --test --host-only --seed 42)
