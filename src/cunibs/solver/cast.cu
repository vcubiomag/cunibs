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

// Deterministic two-stage reductions: stage 1 kernels emit one partial per block (fixed-order
// shared-memory tree), stage 2 runs in a single block that strides the partials in a fixed order.
// Replaces cuBLAS nrm2/dot passes that re-read full vectors the elementwise kernels already touch.
__device__ void block_reduce_partial(double v, double* __restrict__ partials) {
    __shared__ double sdata[kBlock];
    sdata[threadIdx.x] = v;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sdata[threadIdx.x] += sdata[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sdata[0];
}

__device__ double block_reduce_all(const double* __restrict__ partials, int nblocks) {
    __shared__ double sdata[kBlock];
    double v = 0.0;
    for (int i = threadIdx.x; i < nblocks; i += kBlock) v += partials[i];
    sdata[threadIdx.x] = v;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sdata[threadIdx.x] += sdata[threadIdx.x + off];
        __syncthreads();
    }
    return sdata[0];
}

// update_xr + ‖r‖² partials: replaces cg_update_xr + cublasDnrm2 (drops a full re-read of r).
// A standalone two-stage dot for p·ap was measured slower than cublasDdot — only reductions
// FUSED into an elementwise pass that runs anyway are wins.
__global__ void cg_update_xr_norm_kernel(const double* __restrict__ alpha,
                                         const double* __restrict__ neg_alpha,
                                         const double* __restrict__ p,
                                         const double* __restrict__ ap, double* __restrict__ x,
                                         double* __restrict__ r, float* __restrict__ rf,
                                         double* __restrict__ partials, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double ri = 0.0;
    if (i < n) {
        x[i] += (*alpha) * p[i];
        ri = r[i] + (*neg_alpha) * ap[i];
        r[i] = ri;
        rf[i] = static_cast<float>(ri);
    }
    block_reduce_partial(ri * ri, partials);
}

// norm_sq = Σ partials (‖r‖²; the host takes the square root after the pinned readback)
__global__ void cg_reduce_norm_kernel(const double* __restrict__ partials, int nblocks,
                                      double* __restrict__ norm_sq) {
    const double total = block_reduce_all(partials, nblocks);
    if (threadIdx.x == 0) *norm_sq = total;
}

// float→double cast of z + r·z partials: replaces float_to_double + cublasDdot (drops a full
// re-read of r and z).
__global__ void cg_cast_dot_kernel(const float* __restrict__ zf, double* __restrict__ z,
                                   const double* __restrict__ r, double* __restrict__ partials,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double prod = 0.0;
    if (i < n) {
        const double zi = static_cast<double>(zf[i]);
        z[i] = zi;
        prod = r[i] * zi;
    }
    block_reduce_partial(prod, partials);
}

// rz_next = Σ partials; β = rz_next/rz; rz ← rz_next (folds the old cg_beta kernel in)
__global__ void cg_reduce_beta_kernel(const double* __restrict__ partials, int nblocks,
                                      double* __restrict__ rz, double* __restrict__ rz_next,
                                      double* __restrict__ beta) {
    const double total = block_reduce_all(partials, nblocks);
    if (threadIdx.x == 0) {
        *rz_next = total;
        *beta = total / (*rz);
        *rz = total;
    }
}

// y = A x for the outer CG operator. Eight threads cooperate per row (the reduced stiffness has
// ~14 nnz/row) with a fixed-order shuffle reduction, so results are run-to-run deterministic.
// Measured ~15% faster than cusparseSpMV CSR_ALG1 at this size on RTX 5070 Ti
// (benchmarks/probe_spmv_fp16.py).
constexpr int kSpmvTpr = 8;

__global__ void csrmv_f64_kernel(int n, const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_idx,
                                 const double* __restrict__ vals,
                                 const double* __restrict__ x, double* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kSpmvTpr;
    const int lane = threadIdx.x % kSpmvTpr;
    double sum = 0.0;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int c = row_ptr[row] + lane; c < row_e; c += kSpmvTpr) {
            sum += vals[c] * __ldg(x + col_idx[c]);
        }
    }
#pragma unroll
    for (int off = kSpmvTpr / 2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off, kSpmvTpr);
    }
    if (row < n && lane == 0) y[row] = sum;
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

void launch_csrmv_f64(int n, const int* row_ptr, const int* col_idx, const double* vals,
                      const double* x, double* y, cudaStream_t stream) {
    const int blocks = (n * kSpmvTpr + kBlock - 1) / kBlock;
    csrmv_f64_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, vals, x, y);
}

int cg_partials_size(int n) { return (n + kBlock - 1) / kBlock; }

void launch_cg_update_xr_norm(const double* alpha, const double* neg_alpha, const double* p,
                              const double* ap, double* x, double* r, float* rf,
                              double* partials, double* norm_sq, int n, cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    cg_update_xr_norm_kernel<<<blocks, kBlock, 0, stream>>>(alpha, neg_alpha, p, ap, x, r, rf,
                                                            partials, n);
    cg_reduce_norm_kernel<<<1, kBlock, 0, stream>>>(partials, blocks, norm_sq);
}

void launch_cg_cast_dot_beta(const float* zf, double* z, const double* r, double* partials,
                             double* rz, double* rz_next, double* beta, int n,
                             cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    cg_cast_dot_kernel<<<blocks, kBlock, 0, stream>>>(zf, z, r, partials, n);
    cg_reduce_beta_kernel<<<1, kBlock, 0, stream>>>(partials, blocks, rz, rz_next, beta);
}
