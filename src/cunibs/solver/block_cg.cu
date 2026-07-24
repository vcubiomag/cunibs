// Block (multi-RHS) CG kernels: k independent CG chains in lockstep sharing each
// matrix read. All dense operands are row-major (n, k) so the k values of one mesh
// node are contiguous; col_idx/vals are read once per nnz and reused across the k
// columns (the amortization measured in benchmarks/probe_block_spmv.py). Every
// reduction is a fixed-order two-stage tree per column, so results are run-to-run
// deterministic and each column's arithmetic is independent of its neighbours.
#include <stdexcept>
#include <type_traits>

#include "solver.hpp"

namespace {

constexpr int kBlock = 256;

// Measured on RTX 5070 Ti (probe_block_spmv): best threads-per-row shifts down as the
// per-thread register footprint grows with K.
template <int K>
__host__ __device__ constexpr int spmv_tpr() {
    return K >= 4 ? 4 : 8;
}

// At K=8 the 8 fp64 accumulators/thread spill without an occupancy floor;
// __launch_bounds__ minBlocks=4 measured 81.3 -> 72.4 us/RHS on the 5070 Ti.
// For K <= 4 the register footprint is small enough that minBlocks=2 is a no-op.
template <int K>
__global__ void __launch_bounds__(kBlock, K == 8 ? 4 : 2)
    bcsrmv_f64_kernel(int n, const int* __restrict__ row_ptr,
                      const int* __restrict__ col_idx, const double* __restrict__ vals,
                      const double* __restrict__ x, double* __restrict__ y) {
    constexpr int kTpr = spmv_tpr<K>();
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = threadIdx.x % kTpr;
    double sum[K];
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += kTpr) {
            const double v = vals[j];
            const long base = static_cast<long>(col_idx[j]) * K;
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * __ldg(x + base + c);
        }
    }
#pragma unroll
    for (int off = kTpr / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < K; ++c) {
            sum[c] += __shfl_down_sync(0xffffffffu, sum[c], off, kTpr);
        }
    }
    if (row < n && lane == 0) {
        const long base = static_cast<long>(row) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) y[base + c] = sum[c];
    }
}

// Column-wise per-block reduction of K register accumulators: warp shuffles, one
// cross-warp pass through shared memory, a single __syncthreads(). The K-fold
// block-wide tree it replaces cost ~8 barriers per column and throttled the streaming
// kernels that embed it (measured 410 GB/s on the fused x/r update). Fixed order:
// shuffle tree within each warp, then a sequential sum over the kBlock/32 warp
// results, so partials stay deterministic.
template <int K>
__device__ __forceinline__ void bcg_block_reduce_cols(double (&local)[K],
                                                      double* __restrict__ partials,
                                                      int n_blocks) {
    constexpr int kWarps = kBlock / 32;
    __shared__ double s[kWarps][K];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int c = 0; c < K; ++c) {
        double v = local[c];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) s[warp][c] = v;
    }
    __syncthreads();
    if (threadIdx.x < K) {
        double v = 0.0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) v += s[w][threadIdx.x];
        partials[static_cast<long>(threadIdx.x) * n_blocks + blockIdx.x] = v;
    }
}

// Column-wise dot products, stage 1: one partial per (column, block).
template <int K, bool KDOT_XY>
__global__ void bcg_partials_kernel(int n, const double* __restrict__ x,
                                    const double* __restrict__ y, double* __restrict__ partials,
                                    int n_blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local[K];
#pragma unroll
    for (int c = 0; c < K; ++c) local[c] = 0.0;
    if (i < n) {
        const long base = static_cast<long>(i) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            const double xv = x[base + c];
            local[c] = KDOT_XY ? xv * y[base + c] : xv * xv;
        }
    }
    bcg_block_reduce_cols<K>(local, partials, n_blocks);
}

// Stage 2: one block per column reduces that column's partials.
__global__ void bcg_reduce_kernel(const double* __restrict__ partials, int n_blocks,
                                  double* __restrict__ out) {
    __shared__ double sdata[kBlock];
    const int c = blockIdx.x;
    double local = 0.0;
    for (int b = threadIdx.x; b < n_blocks; b += kBlock) {
        local += partials[static_cast<long>(c) * n_blocks + b];
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[c] = sdata[0];
}

__global__ void bcg_alpha_kernel(int k, const double* __restrict__ rz,
                                 const double* __restrict__ pap, double* __restrict__ alpha,
                                 double* __restrict__ neg_alpha) {
    const int c = threadIdx.x;
    if (c < k) {
        const double a = rz[c] / pap[c];
        alpha[c] = a;
        neg_alpha[c] = -a;
    }
}

template <int K>
__global__ void bcg_update_xr_kernel(int n, const double* __restrict__ alpha,
                                     const double* __restrict__ neg_alpha,
                                     const double* __restrict__ p, const double* __restrict__ ap,
                                     double* __restrict__ x, double* __restrict__ r,
                                     float* __restrict__ rf, double* __restrict__ partials,
                                     int n_blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double rloc[K];
#pragma unroll
    for (int c = 0; c < K; ++c) rloc[c] = 0.0;
    if (i < n) {
        const long base = static_cast<long>(i) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            const double a = __ldg(alpha + c);
            const double na = __ldg(neg_alpha + c);
            x[base + c] += a * p[base + c];
            const double rv = r[base + c] + na * ap[base + c];
            r[base + c] = rv;
            rf[base + c] = static_cast<float>(rv);
            rloc[c] = rv * rv;
        }
    }
    bcg_block_reduce_cols<K>(rloc, partials, n_blocks);
}

// The preconditioned residual never needs an fp64 materialization: consumers cast the
// fp32 zf on the fly (exact float->double), saving a full (n, k) fp64 write + read
// per iteration.
template <int K>
__global__ void bcg_cast_dot_kernel(int n, const float* __restrict__ zf,
                                    const double* __restrict__ r,
                                    double* __restrict__ partials, int n_blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local[K];
#pragma unroll
    for (int c = 0; c < K; ++c) local[c] = 0.0;
    if (i < n) {
        const long base = static_cast<long>(i) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            local[c] = r[base + c] * static_cast<double>(zf[base + c]);
        }
    }
    bcg_block_reduce_cols<K>(local, partials, n_blocks);
}

__global__ void bcg_reduce_beta_kernel(const double* __restrict__ partials, int n_blocks,
                                       double* __restrict__ rz, double* __restrict__ rz_next,
                                       double* __restrict__ beta) {
    __shared__ double sdata[kBlock];
    const int c = blockIdx.x;
    double local = 0.0;
    for (int b = threadIdx.x; b < n_blocks; b += kBlock) {
        local += partials[static_cast<long>(c) * n_blocks + b];
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double total = sdata[0];
        rz_next[c] = total;
        beta[c] = total / rz[c];
        rz[c] = total;
    }
}

template <int K>
__global__ void bcg_update_p_kernel(int n, const double* __restrict__ beta,
                                    const float* __restrict__ zf, double* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const long base = static_cast<long>(i) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            p[base + c] = __ldg(beta + c) * p[base + c] + static_cast<double>(zf[base + c]);
        }
    }
}

__global__ void bcg_f2d_kernel(long n_total, const float* __restrict__ in,
                               double* __restrict__ out) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) out[i] = static_cast<double>(in[i]);
}

__global__ void bcg_d2f_kernel(long n_total, const double* __restrict__ in,
                               float* __restrict__ out) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) out[i] = static_cast<float>(in[i]);
}

// R = B - AP (warm-start residual), all (n, k) row-major.
__global__ void bcg_residual_kernel(long n_total, const double* __restrict__ b,
                                    const double* __restrict__ ap, double* __restrict__ r) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) r[i] = b[i] - ap[i];
}

__global__ void strided_gather_kernel(int n, int k, int c, const float* __restrict__ in,
                                      float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[static_cast<long>(i) * k + c];
}

__global__ void strided_scatter_kernel(int n, int k, int c, const float* __restrict__ in,
                                       float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[static_cast<long>(i) * k + c] = in[i];
}

template <typename F>
void dispatch_k(int k, F&& f) {
    switch (k) {
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: throw std::invalid_argument("block CG supports k in {2, 4, 8}");
    }
}

}  // namespace

int bcg_partials_blocks(int n) { return (n + kBlock - 1) / kBlock; }

void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const int* col_idx,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream) {
    dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        const int blocks = (n * spmv_tpr<K>() + kBlock - 1) / kBlock;
        bcsrmv_f64_kernel<K><<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, vals, x, y);
    });
}

void launch_bcg_dot(int n, int k, const double* x, const double* y, double* partials,
                    double* out, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, true>
            <<<n_blocks, kBlock, 0, stream>>>(n, x, y, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, out);
}

void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, false>
            <<<n_blocks, kBlock, 0, stream>>>(n, x, nullptr, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, out);
}

void launch_bcg_alpha(int k, const double* rz, const double* pap, double* alpha,
                      double* neg_alpha, cudaStream_t stream) {
    bcg_alpha_kernel<<<1, k, 0, stream>>>(k, rz, pap, alpha, neg_alpha);
}

void launch_bcg_update_xr_norm(int n, int k, const double* alpha, const double* neg_alpha,
                               const double* p, const double* ap, double* x, double* r,
                               float* rf, double* partials, double* norms,
                               cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_update_xr_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, alpha, neg_alpha, p, ap, x,
                                                                 r, rf, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, norms);
}

void launch_bcg_cast_dot_beta(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, double* rz_next, double* beta,
                              cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_cast_dot_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, n_blocks);
    });
    bcg_reduce_beta_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, rz, rz_next, beta);
}

void launch_bcg_cast_dot_init(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_cast_dot_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, rz);
}

void launch_bcg_update_p(int n, int k, const double* beta, const float* zf, double* p,
                         cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    dispatch_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_update_p_kernel<K><<<blocks, kBlock, 0, stream>>>(n, beta, zf, p);
    });
}

void launch_bcg_f2d(long n_total, const float* in, double* out, cudaStream_t stream) {
    const long blocks = (n_total + kBlock - 1) / kBlock;
    bcg_f2d_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(n_total, in, out);
}

void launch_bcg_d2f(long n_total, const double* in, float* out, cudaStream_t stream) {
    const long blocks = (n_total + kBlock - 1) / kBlock;
    bcg_d2f_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(n_total, in, out);
}

void launch_bcg_residual(long n_total, const double* b, const double* ap, double* r,
                         cudaStream_t stream) {
    const long blocks = (n_total + kBlock - 1) / kBlock;
    bcg_residual_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(n_total, b, ap, r);
}

void launch_strided_gather(int n, int k, int c, const float* in, float* out,
                           cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    strided_gather_kernel<<<blocks, kBlock, 0, stream>>>(n, k, c, in, out);
}

void launch_strided_scatter(int n, int k, int c, const float* in, float* out,
                            cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    strided_scatter_kernel<<<blocks, kBlock, 0, stream>>>(n, k, c, in, out);
}
