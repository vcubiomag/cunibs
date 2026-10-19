#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cstdint>

namespace {

// A block of draws folded in draw order, so per element the additions are the same whatever
// block width the draws were split into. The fp64 sums are read and written once per block.
__global__ void accumulate_moments_block_kernel(ConstPtrPack magn, int k,
                                                double* __restrict__ sum_e,
                                                double* __restrict__ sumsq_e, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = sum_e[i];
    double q = sumsq_e[i];
    for (int c = 0; c < k; ++c) {
        const double m = static_cast<double>(magn[c][i]);
        s += m;
        q += m * m;
    }
    sum_e[i] = s;
    sumsq_e[i] = q;
}

// out[i, c] = sum_t a[i, t] s[c, t] for a mesh-length, narrow a, one thread per (row, column),
// terms in order. A column's sum is the same loop whatever k is, so a draw's guess does not depend
// on the block it was formed in. Adjacent threads share a row, so its loads coalesce across the k
// columns.
__global__ void row_dot_kernel(const double* __restrict__ a, const double* __restrict__ s,
                               double* __restrict__ out, int n, int p, int k) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<std::int64_t>(n) * k) return;
    const std::int64_t i = idx / k;
    const int c = static_cast<int>(idx - i * k);
    const double* row = a + i * p;
    const double* sc = s + static_cast<std::int64_t>(c) * p;
    double acc = 0.0;
    for (int t = 0; t < p; ++t) acc = fma(row[t], sc[t], acc);
    out[idx] = acc;
}

}  // namespace

void launch_accumulate_moments_block(const float* const* magn, int k, double* sum_e,
                                     double* sumsq_e, int n, cudaStream_t stream) {
    const ConstPtrPack in = ptr_pack(magn, k);
    if (const unsigned blocks = grid_for(n)) {
        accumulate_moments_block_kernel<<<blocks, kBlock, 0, stream>>>(in, k, sum_e, sumsq_e, n);
    }
}

void launch_row_dot(const double* a, const double* s, double* out, int n, int p, int k,
                    cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * k)) {
        row_dot_kernel<<<blocks, kBlock, 0, stream>>>(a, s, out, n, p, k);
    }
}
