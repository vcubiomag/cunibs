#include "solver.hpp"

namespace {

constexpr int kBlock = 256;

__global__ void double_to_float_kernel(const double* __restrict__ in, float* __restrict__ out,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}

__global__ void float_to_double_kernel(const float* __restrict__ in, double* __restrict__ out,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<double>(in[i]);
}

}

void launch_double_to_float(const double* in, float* out, int n, cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    double_to_float_kernel<<<blocks, kBlock, 0, stream>>>(in, out, n);
}

void launch_float_to_double(const float* in, double* out, int n, cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    float_to_double_kernel<<<blocks, kBlock, 0, stream>>>(in, out, n);
}
