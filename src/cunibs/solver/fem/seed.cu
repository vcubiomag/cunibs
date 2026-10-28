#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cuda/cmath>

#include <algorithm>
#include <cstdint>

// Seeding a block solve from a sweep basis: x0 = Q (Q^T b) per column, with Q (n, m) fp32 and
// A-orthonormal, which makes x0 the Galerkin solution over span(Q). The guess only sets where CG
// starts, so fp32 products are plenty; what matters is that a column's x0 is the same arithmetic
// whatever block it is in. Every column is summed in a fixed order of rows and basis vectors that
// does not involve k.

namespace {

// Rows one block of the projection folds before writing its partial. Long enough that the
// cross-tile pass is short, short enough that the grid covers the device.
constexpr int kSeedTileRows = 1024;

int seed_tiles(int n) { return cuda::ceil_div(n, kSeedTileRows); }

// b in fp32, converted once, so the loops below do not convert it once per basis vector at the
// fp64 rate.
__global__ void seed_narrow_kernel(const double* __restrict__ b, std::int64_t total,
                                   float* __restrict__ out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) out[i] = static_cast<float>(b[i]);
}

// partials[tile][j][c] = sum over the tile's rows of q[r, j] b[r, c], one thread per basis
// vector, blockIdx.y selecting which stretch of them. The row's k values are a broadcast across
// the block, and the block's stretch of q's row is one coalesced read.
template <int K>
__global__ void seed_project_kernel(const float* __restrict__ q, const float* __restrict__ b,
                                    int n, int m, double* __restrict__ partials) {
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const int r0 = blockIdx.x * kSeedTileRows;
    const int r1 = min(n, r0 + kSeedTileRows);
    float acc[K];
#pragma unroll
    for (int c = 0; c < K; ++c) acc[c] = 0.0f;
    for (int r = r0; r < r1; ++r) {
        const float qv = q[static_cast<std::int64_t>(r) * m + j];
        cuda::std::array<float, K> bv;
        load_row<K>(b + static_cast<std::int64_t>(r) * K, bv);
#pragma unroll
        for (int c = 0; c < K; ++c) acc[c] = fmaf(qv, bv[c], acc[c]);
    }
    double* out = partials + (static_cast<std::int64_t>(blockIdx.x) * m + j) * K;
#pragma unroll
    for (int c = 0; c < K; ++c) out[c] = static_cast<double>(acc[c]);
}

// y[j][c] = the tiles' partials summed in tile order, narrowed for the expansion.
template <int K>
__global__ void seed_reduce_kernel(const double* __restrict__ partials, int n_tiles, int m,
                                   float* __restrict__ y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * K) return;
    double s = 0.0;
    for (int t = 0; t < n_tiles; ++t) s += partials[static_cast<std::int64_t>(t) * m * K + idx];
    y[idx] = static_cast<float>(s);
}

// Columns of q a block of the expansion stages at a time.
constexpr int kExpandCols = 16;

// x0[r][c] = sum over j of q[r, j] y[j][c], basis vectors in order, one thread per row carrying all
// K columns. A thread reading its own row of q would put every lane of a load on a separate line,
// so the block stages kExpandCols columns of its rows at a time through coalesced loads, and each
// thread folds its row from shared memory. The padding column keeps the row reads off one bank.
template <int K>
__global__ void __launch_bounds__(kBlock)
    seed_expand_kernel(const float* __restrict__ q, const float* __restrict__ y, int n, int m,
                       double* __restrict__ x0) {
    __shared__ float tile[kBlock][kExpandCols + 1];
    const int r0 = blockIdx.x * kBlock;
    cuda::std::array<float, K> acc;
#pragma unroll
    for (int c = 0; c < K; ++c) acc[c] = 0.0f;
    for (int j0 = 0; j0 < m; j0 += kExpandCols) {
        const int w = min(kExpandCols, m - j0);
        for (int e = threadIdx.x; e < kBlock * kExpandCols; e += kBlock) {
            const int rr = e / kExpandCols, jj = e % kExpandCols;
            const int row = r0 + rr;
            tile[rr][jj] =
                (row < n && jj < w) ? q[static_cast<std::int64_t>(row) * m + j0 + jj] : 0.0f;
        }
        __syncthreads();
        for (int jj = 0; jj < w; ++jj) {
            const float qv = tile[threadIdx.x][jj];
            cuda::std::array<float, K> yv;
            load_row<K>(y + static_cast<std::int64_t>(j0 + jj) * K, yv);
#pragma unroll
            for (int c = 0; c < K; ++c) acc[c] = fmaf(qv, yv[c], acc[c]);
        }
        __syncthreads();
    }
    const int r = r0 + threadIdx.x;
    if (r >= n) return;
    cuda::std::array<double, K> out;
#pragma unroll
    for (int c = 0; c < K; ++c) out[c] = static_cast<double>(acc[c]);
    store_row<K>(x0 + static_cast<std::int64_t>(r) * K, out);
}

}  // namespace

// work holds the tiles' partials, then y and b narrowed to fp32, two floats to a double.
std::int64_t sweep_seed_work_size(int n, int m, int k) {
    const std::int64_t floats = static_cast<std::int64_t>(m) * k + static_cast<std::int64_t>(n) * k;
    return static_cast<std::int64_t>(seed_tiles(n)) * m * k + (floats + 1) / 2;
}

void launch_sweep_seed(const float* q, const double* b, int n, int m, int k, double* work,
                       double* x0, cudaStream_t stream) {
    if (n <= 0) return;
    if (m <= 0) {
        check_cuda(cudaMemsetAsync(x0, 0, static_cast<std::size_t>(n) * k * sizeof(double), stream),
                   "seed", "zero x0");
        return;
    }
    const int n_tiles = seed_tiles(n);
    float* const y = reinterpret_cast<float*>(work + static_cast<std::int64_t>(n_tiles) * m * k);
    float* const b32 = y + static_cast<std::int64_t>(m) * k;
    const std::int64_t total = static_cast<std::int64_t>(n) * k;
    if (const unsigned blocks = grid_for(total)) {
        seed_narrow_kernel<<<blocks, kBlock, 0, stream>>>(b, total, b32);
    }
    dispatch_k<1, 2, 4, 8>(k, "sweep_seed: k must be in {1, 2, 4, 8}", [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        const int threads = std::min(cuda::round_up(m, kWarp), kBlock);
        const dim3 grid(n_tiles, cuda::ceil_div(m, threads));
        seed_project_kernel<K><<<grid, threads, 0, stream>>>(q, b32, n, m, work);
        seed_reduce_kernel<K><<<grid_for(m * K), kBlock, 0, stream>>>(work, n_tiles, m, y);
        seed_expand_kernel<K><<<grid_for(n), kBlock, 0, stream>>>(q, y, n, m, x0);
    });
    check_cuda(cudaGetLastError(), "seed", "launch");
}
