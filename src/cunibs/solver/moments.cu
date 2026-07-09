#include "kernels.hpp"

#include <cuda_runtime.h>

namespace {

constexpr int kBlock = 256;

__global__ void accumulate_moments_kernel(const float* __restrict__ magn,
                                          double* __restrict__ sum_e,
                                          double* __restrict__ sumsq_e, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double m = static_cast<double>(magn[i]);
    sum_e[i] += m;
    sumsq_e[i] += m * m;
}

}  // namespace

void launch_accumulate_moments(const float* magn, double* sum_e, double* sumsq_e, int n,
                               cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    accumulate_moments_kernel<<<blocks, kBlock, 0, stream>>>(magn, sum_e, sumsq_e, n);
}
