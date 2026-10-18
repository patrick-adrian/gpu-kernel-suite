#include "kernels/hip_kernels.hpp"

#include "common/stats.hpp"
#include "hip_check.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace gks::hip_api {
namespace {

template <typename F>
HipStatus run_timed(F&& launch_once, TimingStats* kernel_ms, int warmup, int trials) {
  HipStatus st;
  try {
    for (int i = 0; i < warmup; ++i) {
      launch_once();
    }
    GKS_HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(std::max(trials, 0)));
    for (int i = 0; i < trials; ++i) {
      hipEvent_t start{};
      hipEvent_t stop{};
      GKS_HIP_CHECK(hipEventCreate(&start));
      GKS_HIP_CHECK(hipEventCreate(&stop));
      GKS_HIP_CHECK(hipEventRecord(start, nullptr));
      launch_once();
      GKS_HIP_CHECK(hipEventRecord(stop, nullptr));
      GKS_HIP_CHECK(hipEventSynchronize(stop));
      float ms = 0.0f;
      GKS_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
      samples.push_back(static_cast<double>(ms));
      GKS_HIP_CHECK(hipEventDestroy(start));
      GKS_HIP_CHECK(hipEventDestroy(stop));
    }
    if (kernel_ms) {
      *kernel_ms = compute_timing_stats(std::move(samples));
    }
    st.ok = true;
    st.message = "ok";
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

__global__ void k_vector_add(const f32* a, const f32* b, f32* c, index_t n) {
  const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void k_matmul_naive(const f32* a, const f32* b, f32* c, index_t m, index_t n,
                               index_t k) {
  const index_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const index_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }
  f32 acc = 0.0f;
  for (index_t p = 0; p < k; ++p) {
    acc += a[row * k + p] * b[p * n + col];
  }
  c[row * n + col] = acc;
}

template <int TILE>
__global__ void k_matmul_tiled(const f32* a, const f32* b, f32* c, index_t m, index_t n,
                               index_t k) {
  __shared__ f32 As[TILE][TILE];
  __shared__ f32 Bs[TILE][TILE];

  const index_t row = blockIdx.y * TILE + threadIdx.y;
  const index_t col = blockIdx.x * TILE + threadIdx.x;

  f32 acc = 0.0f;
  const index_t tiles = (k + TILE - 1) / TILE;
  for (index_t t = 0; t < tiles; ++t) {
    const index_t a_col = t * TILE + threadIdx.x;
    const index_t b_row = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int p = 0; p < TILE; ++p) {
      acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

__global__ void k_reduce_sum_block(const f32* in, f32* partial, index_t n) {
  extern __shared__ f32 smem[];
  const index_t tid = threadIdx.x;
  index_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  f32 val = 0.0f;
  if (i < n) {
    val = in[i];
  }
  if (i + blockDim.x < n) {
    val += in[i + blockDim.x];
  }
  smem[tid] = val;
  __syncthreads();
  for (index_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial[blockIdx.x] = smem[0];
  }
}

__global__ void k_reduce_max_block(const f32* in, f32* partial, index_t n) {
  extern __shared__ f32 smem[];
  const index_t tid = threadIdx.x;
  index_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  f32 val = -std::numeric_limits<f32>::infinity();
  if (i < n) {
    val = in[i];
  }
  if (i + blockDim.x < n) {
    val = fmaxf(val, in[i + blockDim.x]);
  }
  smem[tid] = val;
  __syncthreads();
  for (index_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial[blockIdx.x] = smem[0];
  }
}

__global__ void k_reduce_sum_naive(const f32* in, f32* out, index_t n) {
  // Single-threaded baseline for comparison (intentionally serial on GPU).
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    f32 acc = 0.0f;
    for (index_t i = 0; i < n; ++i) {
      acc += in[i];
    }
    out[0] = acc;
  }
}

__global__ void k_reduce_max_naive(const f32* in, f32* out, index_t n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    f32 m = -std::numeric_limits<f32>::infinity();
    for (index_t i = 0; i < n; ++i) {
      m = fmaxf(m, in[i]);
    }
    out[0] = m;
  }
}

__global__ void k_transpose_naive(const f32* in, f32* out, index_t rows, index_t cols) {
  const index_t i = blockIdx.y * blockDim.y + threadIdx.y;
  const index_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    out[j * rows + i] = in[i * cols + j];
  }
}

template <int TILE>
__global__ void k_transpose_tiled(const f32* in, f32* out, index_t rows, index_t cols) {
  __shared__ f32 tile[TILE][TILE];
  const index_t x = blockIdx.x * TILE + threadIdx.x;
  const index_t y = blockIdx.y * TILE + threadIdx.y;
  if (y < rows && x < cols) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }
  __syncthreads();
  const index_t tx = blockIdx.y * TILE + threadIdx.x;
  const index_t ty = blockIdx.x * TILE + threadIdx.y;
  if (ty < cols && tx < rows) {
    out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
  }
}

// Padding to mitigate LDS bank conflicts on AMD.
template <int TILE>
__global__ void k_transpose_tiled_opt(const f32* in, f32* out, index_t rows, index_t cols) {
  __shared__ f32 tile[TILE][TILE + 1];
  const index_t x = blockIdx.x * TILE + threadIdx.x;
  const index_t y = blockIdx.y * TILE + threadIdx.y;
  if (y < rows && x < cols) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }
  __syncthreads();
  const index_t tx = blockIdx.y * TILE + threadIdx.x;
  const index_t ty = blockIdx.x * TILE + threadIdx.y;
  if (ty < cols && tx < rows) {
    out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
  }
}

HipStatus hierarchical_reduce(bool is_sum, const std::vector<f32>& x, f32& out,
                              const LaunchConfig& launch, TimingStats* kernel_ms, int warmup,
                              int trials) {
  HipStatus st;
  if (x.empty()) {
    if (is_sum) {
      out = 0.0f;
      st.ok = true;
      st.message = "ok empty sum";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    st.ok = false;
    st.message = "reduction_max: empty input";
    return st;
  }

  try {
    const index_t n = x.size();
    const index_t block = std::max<index_t>(1, launch.block_x);
    const std::size_t smem = block * sizeof(f32);

    // Ping-pong buffers sized to the original input (first level is largest).
    f32* d_a = nullptr;
    f32* d_b = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_a, n * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_b, n * sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_a, x.data(), n * sizeof(f32), hipMemcpyHostToDevice));

    auto launch_level = [&](f32* in, f32* partial, index_t len, index_t grid) {
      if (is_sum) {
        hipLaunchKernelGGL(k_reduce_sum_block, dim3(grid), dim3(block), smem, 0, in, partial,
                           len);
      } else {
        hipLaunchKernelGGL(k_reduce_max_block, dim3(grid), dim3(block), smem, 0, in, partial,
                           len);
      }
      GKS_HIP_CHECK(hipGetLastError());
    };

    // Timed region: all device reduction launches (multi-kernel hierarchy).
    // Allocations and the initial H2D are outside timing. Final scalar D2H is after timing.
    auto device_reduce = [&]() {
      index_t cur_n = n;
      f32* in_ptr = d_a;
      f32* out_ptr = d_b;
      while (cur_n > 1) {
        index_t g = (cur_n + block * 2 - 1) / (block * 2);
        if (g == 0) {
          g = 1;
        }
        launch_level(in_ptr, out_ptr, cur_n, g);
        std::swap(in_ptr, out_ptr);
        cur_n = g;
      }
      // Result lives in in_ptr[0] after the loop.
      GKS_HIP_CHECK(hipMemcpy(&out, in_ptr, sizeof(f32), hipMemcpyDeviceToHost));
    };

    st = run_timed(device_reduce, kernel_ms, warmup, trials);
    // Ensure `out` reflects a final correct pass (run_timed already ran trials).
    if (st.ok) {
      device_reduce();
    }
    GKS_HIP_CHECK(hipFree(d_a));
    GKS_HIP_CHECK(hipFree(d_b));
    if (st.ok) {
      st.message = "ok (timed path includes hierarchical launches + scalar D2H)";
    }
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

}  // namespace

HipStatus vector_add(const std::vector<f32>& a, const std::vector<f32>& b, std::vector<f32>& c,
                     const LaunchConfig& launch, TimingStats* kernel_ms, int warmup, int trials) {
  HipStatus st;
  if (a.size() != b.size()) {
    st.message = "vector_add: size mismatch";
    return st;
  }
  try {
    const index_t n = a.size();
    c.resize(n);
    if (n == 0) {
      st.ok = true;
      st.message = "ok empty";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    f32 *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_a, n * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_b, n * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_c, n * sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_a, a.data(), n * sizeof(f32), hipMemcpyHostToDevice));
    GKS_HIP_CHECK(hipMemcpy(d_b, b.data(), n * sizeof(f32), hipMemcpyHostToDevice));

    const index_t block = std::max<index_t>(1, launch.block_x);
    const index_t grid = (n + block - 1) / block;
    auto once = [&]() {
      hipLaunchKernelGGL(k_vector_add, dim3(grid), dim3(block), 0, 0, d_a, d_b, d_c, n);
      GKS_HIP_CHECK(hipGetLastError());
    };
    st = run_timed(once, kernel_ms, warmup, trials);
    GKS_HIP_CHECK(hipMemcpy(c.data(), d_c, n * sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_a));
    GKS_HIP_CHECK(hipFree(d_b));
    GKS_HIP_CHECK(hipFree(d_c));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

HipStatus matmul_naive(const std::vector<f32>& a, const std::vector<f32>& b, std::vector<f32>& c,
                       index_t m, index_t n, index_t k, const LaunchConfig& launch,
                       TimingStats* kernel_ms, int warmup, int trials) {
  HipStatus st;
  try {
    if (a.size() != m * k || b.size() != k * n) {
      st.message = "matmul_naive: dimension mismatch";
      return st;
    }
    c.assign(m * n, 0.0f);
    if (m == 0 || n == 0 || k == 0) {
      st.ok = true;
      st.message = "ok empty";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    f32 *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_a, m * k * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_b, k * n * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_c, m * n * sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_a, a.data(), m * k * sizeof(f32), hipMemcpyHostToDevice));
    GKS_HIP_CHECK(hipMemcpy(d_b, b.data(), k * n * sizeof(f32), hipMemcpyHostToDevice));

    dim3 block(16, 16);
    if (launch.block_x > 0 && launch.block_y > 0) {
      block = dim3(static_cast<uint32_t>(launch.block_x), static_cast<uint32_t>(launch.block_y));
    }
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    auto once = [&]() {
      hipLaunchKernelGGL(k_matmul_naive, grid, block, 0, 0, d_a, d_b, d_c, m, n, k);
      GKS_HIP_CHECK(hipGetLastError());
    };
    st = run_timed(once, kernel_ms, warmup, trials);
    GKS_HIP_CHECK(hipMemcpy(c.data(), d_c, m * n * sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_a));
    GKS_HIP_CHECK(hipFree(d_b));
    GKS_HIP_CHECK(hipFree(d_c));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

HipStatus matmul_tiled(const std::vector<f32>& a, const std::vector<f32>& b, std::vector<f32>& c,
                       index_t m, index_t n, index_t k, const LaunchConfig& launch,
                       TimingStats* kernel_ms, int warmup, int trials) {
  HipStatus st;
  try {
    if (a.size() != m * k || b.size() != k * n) {
      st.message = "matmul_tiled: dimension mismatch";
      return st;
    }
    c.assign(m * n, 0.0f);
    if (m == 0 || n == 0 || k == 0) {
      st.ok = true;
      st.message = "ok empty";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    f32 *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_a, m * k * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_b, k * n * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_c, m * n * sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_a, a.data(), m * k * sizeof(f32), hipMemcpyHostToDevice));
    GKS_HIP_CHECK(hipMemcpy(d_b, b.data(), k * n * sizeof(f32), hipMemcpyHostToDevice));

    const int tile = static_cast<int>(launch.tile);
    auto dispatch = [&](auto kernel) {
      dim3 block(tile, tile);
      dim3 grid((n + tile - 1) / tile, (m + tile - 1) / tile);
      auto once = [&]() {
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, d_a, d_b, d_c, m, n, k);
        GKS_HIP_CHECK(hipGetLastError());
      };
      return run_timed(once, kernel_ms, warmup, trials);
    };

    if (tile == 8) {
      st = dispatch(k_matmul_tiled<8>);
    } else if (tile == 32) {
      st = dispatch(k_matmul_tiled<32>);
    } else {
      st = dispatch(k_matmul_tiled<16>);
      if (tile != 16) {
        st.message += " (unsupported tile; used 16)";
      }
    }

    GKS_HIP_CHECK(hipMemcpy(c.data(), d_c, m * n * sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_a));
    GKS_HIP_CHECK(hipFree(d_b));
    GKS_HIP_CHECK(hipFree(d_c));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

HipStatus reduction_sum_baseline(const std::vector<f32>& x, f32& out, const LaunchConfig&,
                                 TimingStats* kernel_ms, int warmup, int trials) {
  HipStatus st;
  try {
    if (x.empty()) {
      out = 0.0f;
      st.ok = true;
      st.message = "ok empty";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    f32* d_in = nullptr;
    f32* d_out = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_in, x.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_out, sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_in, x.data(), x.size() * sizeof(f32), hipMemcpyHostToDevice));
    auto once = [&]() {
      hipLaunchKernelGGL(k_reduce_sum_naive, dim3(1), dim3(1), 0, 0, d_in, d_out, x.size());
      GKS_HIP_CHECK(hipGetLastError());
    };
    st = run_timed(once, kernel_ms, warmup, trials);
    GKS_HIP_CHECK(hipMemcpy(&out, d_out, sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_in));
    GKS_HIP_CHECK(hipFree(d_out));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

HipStatus reduction_max_baseline(const std::vector<f32>& x, f32& out, const LaunchConfig&,
                                 TimingStats* kernel_ms, int warmup, int trials) {
  HipStatus st;
  try {
    if (x.empty()) {
      st.message = "reduction_max: empty input";
      return st;
    }
    f32* d_in = nullptr;
    f32* d_out = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_in, x.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_out, sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_in, x.data(), x.size() * sizeof(f32), hipMemcpyHostToDevice));
    auto once = [&]() {
      hipLaunchKernelGGL(k_reduce_max_naive, dim3(1), dim3(1), 0, 0, d_in, d_out, x.size());
      GKS_HIP_CHECK(hipGetLastError());
    };
    st = run_timed(once, kernel_ms, warmup, trials);
    GKS_HIP_CHECK(hipMemcpy(&out, d_out, sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_in));
    GKS_HIP_CHECK(hipFree(d_out));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

HipStatus reduction_sum_hierarchical(const std::vector<f32>& x, f32& out,
                                     const LaunchConfig& launch, TimingStats* kernel_ms,
                                     int warmup, int trials) {
  return hierarchical_reduce(true, x, out, launch, kernel_ms, warmup, trials);
}

HipStatus reduction_max_hierarchical(const std::vector<f32>& x, f32& out,
                                     const LaunchConfig& launch, TimingStats* kernel_ms,
                                     int warmup, int trials) {
  return hierarchical_reduce(false, x, out, launch, kernel_ms, warmup, trials);
}

HipStatus transpose_baseline(const std::vector<f32>& in, std::vector<f32>& out, index_t rows,
                             index_t cols, const LaunchConfig& launch, TimingStats* kernel_ms,
                             int warmup, int trials) {
  HipStatus st;
  try {
    if (in.size() != rows * cols) {
      st.message = "transpose: dimension mismatch";
      return st;
    }
    out.resize(rows * cols);
    if (rows == 0 || cols == 0) {
      st.ok = true;
      st.message = "ok empty";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    f32 *d_in = nullptr, *d_out = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_in, in.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_out, out.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_in, in.data(), in.size() * sizeof(f32), hipMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    (void)launch;
    auto once = [&]() {
      hipLaunchKernelGGL(k_transpose_naive, grid, block, 0, 0, d_in, d_out, rows, cols);
      GKS_HIP_CHECK(hipGetLastError());
    };
    st = run_timed(once, kernel_ms, warmup, trials);
    GKS_HIP_CHECK(hipMemcpy(out.data(), d_out, out.size() * sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_in));
    GKS_HIP_CHECK(hipFree(d_out));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

HipStatus transpose_tiled(const std::vector<f32>& in, std::vector<f32>& out, index_t rows,
                          index_t cols, const LaunchConfig& launch, TimingStats* kernel_ms,
                          int warmup, int trials) {
  HipStatus st;
  try {
    if (in.size() != rows * cols) {
      st.message = "transpose: dimension mismatch";
      return st;
    }
    out.resize(rows * cols);
    if (rows == 0 || cols == 0) {
      st.ok = true;
      st.message = "ok empty";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    f32 *d_in = nullptr, *d_out = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_in, in.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_out, out.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_in, in.data(), in.size() * sizeof(f32), hipMemcpyHostToDevice));

    const int tile = (launch.tile == 8 || launch.tile == 32) ? static_cast<int>(launch.tile) : 16;
    auto run = [&](auto kernel) {
      dim3 block(tile, tile);
      dim3 grid((cols + tile - 1) / tile, (rows + tile - 1) / tile);
      auto once = [&]() {
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, d_in, d_out, rows, cols);
        GKS_HIP_CHECK(hipGetLastError());
      };
      return run_timed(once, kernel_ms, warmup, trials);
    };
    if (tile == 8) st = run(k_transpose_tiled<8>);
    else if (tile == 32) st = run(k_transpose_tiled<32>);
    else st = run(k_transpose_tiled<16>);

    GKS_HIP_CHECK(hipMemcpy(out.data(), d_out, out.size() * sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_in));
    GKS_HIP_CHECK(hipFree(d_out));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

HipStatus transpose_tiled_opt(const std::vector<f32>& in, std::vector<f32>& out, index_t rows,
                              index_t cols, const LaunchConfig& launch, TimingStats* kernel_ms,
                              int warmup, int trials) {
  HipStatus st;
  try {
    if (in.size() != rows * cols) {
      st.message = "transpose: dimension mismatch";
      return st;
    }
    out.resize(rows * cols);
    if (rows == 0 || cols == 0) {
      st.ok = true;
      st.message = "ok empty";
      if (kernel_ms) *kernel_ms = {};
      return st;
    }
    f32 *d_in = nullptr, *d_out = nullptr;
    GKS_HIP_CHECK(hipMalloc(&d_in, in.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMalloc(&d_out, out.size() * sizeof(f32)));
    GKS_HIP_CHECK(hipMemcpy(d_in, in.data(), in.size() * sizeof(f32), hipMemcpyHostToDevice));

    const int tile = (launch.tile == 8 || launch.tile == 32) ? static_cast<int>(launch.tile) : 16;
    auto run = [&](auto kernel) {
      dim3 block(tile, tile);
      dim3 grid((cols + tile - 1) / tile, (rows + tile - 1) / tile);
      auto once = [&]() {
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, d_in, d_out, rows, cols);
        GKS_HIP_CHECK(hipGetLastError());
      };
      return run_timed(once, kernel_ms, warmup, trials);
    };
    if (tile == 8) st = run(k_transpose_tiled_opt<8>);
    else if (tile == 32) st = run(k_transpose_tiled_opt<32>);
    else st = run(k_transpose_tiled_opt<16>);

    GKS_HIP_CHECK(hipMemcpy(out.data(), d_out, out.size() * sizeof(f32), hipMemcpyDeviceToHost));
    GKS_HIP_CHECK(hipFree(d_in));
    GKS_HIP_CHECK(hipFree(d_out));
  } catch (const std::exception& ex) {
    st.ok = false;
    st.message = ex.what();
  }
  return st;
}

}  // namespace gks::hip_api
