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

// Fuses x += α p; r -= α ap; rf = (float)r. Elementwise, so the sum order matches the three cublas
// ops it replaces (axpy(x) + axpy(r) + double_to_float(r)) and the result is bit-identical.
__global__ void cg_update_xr_kernel(const double* __restrict__ alpha,
                                    const double* __restrict__ neg_alpha,
                                    const double* __restrict__ p, const double* __restrict__ ap,
                                    double* __restrict__ x, double* __restrict__ r,
                                    float* __restrict__ rf, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += (*alpha) * p[i];
    const double ri = r[i] + (*neg_alpha) * ap[i];
    r[i] = ri;
    rf[i] = static_cast<float>(ri);
}

// Fuses p = β p + z (cublas scal(p) + axpy(z→p)): one read/write of p instead of two.
__global__ void cg_update_p_kernel(const double* __restrict__ beta, const double* __restrict__ z,
                                   double* __restrict__ p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = (*beta) * p[i] + z[i];
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

void launch_cg_update_xr(const double* alpha, const double* neg_alpha, const double* p,
                         const double* ap, double* x, double* r, float* rf, int n,
                         cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    cg_update_xr_kernel<<<blocks, kBlock, 0, stream>>>(alpha, neg_alpha, p, ap, x, r, rf, n);
}

void launch_cg_update_p(const double* beta, const double* z, double* p, int n,
                        cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    cg_update_p_kernel<<<blocks, kBlock, 0, stream>>>(beta, z, p, n);
}
