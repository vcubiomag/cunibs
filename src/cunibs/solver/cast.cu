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

// Keep the CG scalars on-device so the outer PCG never round-trips a reduction to the host.
// IEEE-correctly-rounded fp64 division, so the iterates match a host divide bit for bit.
__global__ void cg_alpha_kernel(const double* rz, const double* pap, double* alpha,
                                double* neg_alpha) {
    const double a = (*rz) / (*pap);
    *alpha = a;
    *neg_alpha = -a;
}

__global__ void cg_beta_kernel(const double* rz_next, double* rz, double* beta) {
    *beta = (*rz_next) / (*rz);
    *rz = *rz_next;  // carry rz forward for the next iteration in the same launch
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

void launch_cg_alpha(const double* rz, const double* pap, double* alpha, double* neg_alpha,
                     cudaStream_t stream) {
    cg_alpha_kernel<<<1, 1, 0, stream>>>(rz, pap, alpha, neg_alpha);
}

void launch_cg_beta(const double* rz_next, double* rz, double* beta, cudaStream_t stream) {
    cg_beta_kernel<<<1, 1, 0, stream>>>(rz_next, rz, beta);
}
