// Mixed-precision CG kernels, templated on the number of right-hand sides K, following the
// policy vcycle.cu sets: every step is written once and the single-RHS path is the K = 1
// instantiation. K chains run in lockstep sharing each matrix read. All dense operands are
// row-major (n, k) so the k values of one mesh node are contiguous; col_idx/vals are read
// once per nnz and reused across the k columns. Every reduction is a fixed-order two-stage
// tree per column, so results are run-to-run deterministic and each column's arithmetic is
// independent of its neighbours.
#include <cstdint>

#include "solver.hpp"

namespace {

// Threads cooperating on one row. The reduced stiffness has ~14 nnz/row (P1 tets), so eight
// threads cover a row in a couple of strided steps. The best width shifts down as the
// per-thread register footprint grows with K. This is the fp64 outer operator; vcycle.cu
// tunes its fp32 counterpart separately and lands on 4 for every K.
template <int K>
__host__ __device__ constexpr int spmv_tpr() {
    return K >= 4 ? 4 : 8;
}

// Blocks-per-SM floor, which is what bounds the register budget ptxas will spend. The K
// accumulators dominate the footprint, so the floor has to fall as K rises: at K=8 the
// accumulators spill without a floor of 4, while K=1 has the headroom to keep six blocks
// resident and, being purely bandwidth-bound, wants them for latency hiding. Measured in
// registers per thread on sm_120: K=1 takes 40 at 6, but 80 at 2 -- half the occupancy for
// a kernel that has nothing to do with the extra registers.
template <int K>
__host__ __device__ constexpr int spmv_min_blocks() {
    if constexpr (K == 8) {
        return 4;
    } else if constexpr (K == 1) {
        return 6;
    } else {
        return 2;
    }
}

template <int K>
__global__ void __launch_bounds__(kBlock, spmv_min_blocks<K>())
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
            const std::int64_t base = static_cast<std::int64_t>(col_idx[j]) * K;
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
        const std::int64_t base = static_cast<std::int64_t>(row) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) y[base + c] = sum[c];
    }
}

// Per-block reduction of K register accumulators into one partial per (column, block).
//
// The two orders are deliberate, not an accident of history. Each is what its path has
// always used, and each stays exactly what it was: a reduction order is part of the answer
// here, so sharing these kernels across K must not quietly re-associate either sum.
template <int K>
__device__ __forceinline__ void bcg_block_reduce_cols(double (&local)[K],
                                                      double* __restrict__ partials,
                                                      int n_blocks) {
    if constexpr (K == 1) {
        // Single RHS: a fixed-order tree over the whole block. One column cannot fill the
        // warp-shuffle path's registers, and the extra barriers cost nothing that matters
        // when there is only one accumulator in flight.
        (void)n_blocks;  // column 0 starts at offset 0, so the stride never comes up
        __shared__ double sdata[kBlock];
        sdata[threadIdx.x] = local[0];
        __syncthreads();
        for (int off = kBlock / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) sdata[threadIdx.x] += sdata[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) partials[blockIdx.x] = sdata[0];
    } else {
        // Multiple RHS: warp shuffles, one cross-warp pass through shared memory, a single
        // __syncthreads(). Keeping it to one barrier matters because the streaming kernels
        // that embed it are bandwidth-bound. Fixed order: shuffle tree within each warp,
        // then a sequential sum over the kBlock/kWarp warp results.
        constexpr int kWarps = kBlock / kWarp;
        __shared__ double s[kWarps][K];
        const int lane = threadIdx.x % kWarp;
        const int warp = threadIdx.x / kWarp;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            double v = local[c];
#pragma unroll
            for (int off = kWarp / 2; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) s[warp][c] = v;
        }
        __syncthreads();
        if (threadIdx.x < K) {
            double v = 0.0;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) v += s[w][threadIdx.x];
            partials[static_cast<std::int64_t>(threadIdx.x) * n_blocks + blockIdx.x] = v;
        }
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
        const std::int64_t base = static_cast<std::int64_t>(i) * K;
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
        local += partials[static_cast<std::int64_t>(c) * n_blocks + b];
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
        const std::int64_t base = static_cast<std::int64_t>(i) * K;
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
        const std::int64_t base = static_cast<std::int64_t>(i) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            local[c] = r[base + c] * static_cast<double>(zf[base + c]);
        }
    }
    bcg_block_reduce_cols<K>(local, partials, n_blocks);
}

__global__ void bcg_reduce_beta_kernel(const double* __restrict__ partials, int n_blocks,
                                       double* __restrict__ rz, double* __restrict__ beta) {
    __shared__ double sdata[kBlock];
    const int c = blockIdx.x;
    double local = 0.0;
    for (int b = threadIdx.x; b < n_blocks; b += kBlock) {
        local += partials[static_cast<std::int64_t>(c) * n_blocks + b];
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double total = sdata[0];
        beta[c] = total / rz[c];
        rz[c] = total;
    }
}

template <int K>
__global__ void bcg_update_p_kernel(int n, const double* __restrict__ beta,
                                    const float* __restrict__ zf, double* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const std::int64_t base = static_cast<std::int64_t>(i) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            p[base + c] = __ldg(beta + c) * p[base + c] + static_cast<double>(zf[base + c]);
        }
    }
}

__global__ void bcg_f2d_kernel(std::int64_t n_total, const float* __restrict__ in,
                               double* __restrict__ out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) out[i] = static_cast<double>(in[i]);
}

__global__ void bcg_d2f_kernel(std::int64_t n_total, const double* __restrict__ in,
                               float* __restrict__ out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) out[i] = static_cast<float>(in[i]);
}

// R = B - AP (warm-start residual), all (n, k) row-major.
__global__ void bcg_residual_kernel(std::int64_t n_total, const double* __restrict__ b,
                                    const double* __restrict__ ap, double* __restrict__ r) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) r[i] = b[i] - ap[i];
}

constexpr const char* kBadK = "block CG supports k in {2, 4, 8}";

}  // namespace

int bcg_partials_blocks(int n) { return static_cast<int>(grid_for(n)); }

void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const int* col_idx,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream) {
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * spmv_tpr<K>())) {
            bcsrmv_f64_kernel<K><<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, vals, x, y);
        }
    });
}

void launch_bcg_dot(int n, int k, const double* x, const double* y, double* partials,
                    double* out, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, true>
            <<<n_blocks, kBlock, 0, stream>>>(n, x, y, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, out);
}

void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
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
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_update_xr_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, alpha, neg_alpha, p, ap, x,
                                                                 r, rf, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, norms);
}

void launch_bcg_cast_dot_beta(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, double* beta,
                              cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_cast_dot_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, n_blocks);
    });
    bcg_reduce_beta_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, rz, beta);
}

void launch_bcg_cast_dot_init(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_cast_dot_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, rz);
}

void launch_bcg_update_p(int n, int k, const double* beta, const float* zf, double* p,
                         cudaStream_t stream) {
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(n)) {
            bcg_update_p_kernel<K><<<blocks, kBlock, 0, stream>>>(n, beta, zf, p);
        }
    });
}

void launch_bcg_f2d(std::int64_t n_total, const float* in, double* out, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_total)) {
        bcg_f2d_kernel<<<blocks, kBlock, 0, stream>>>(n_total, in, out);
    }
}

void launch_bcg_d2f(std::int64_t n_total, const double* in, float* out, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_total)) {
        bcg_d2f_kernel<<<blocks, kBlock, 0, stream>>>(n_total, in, out);
    }
}

void launch_bcg_residual(std::int64_t n_total, const double* b, const double* ap, double* r,
                         cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_total)) {
        bcg_residual_kernel<<<blocks, kBlock, 0, stream>>>(n_total, b, ap, r);
    }
}

// --- single-RHS entry points -------------------------------------------------------------
//
// The scalar CG path is the K = 1 instantiation of the kernels above. These name K at the
// call site rather than going through dispatch_k, so the block launchers' supported widths
// stay {2, 4, 8} and the scalar path costs no runtime switch. Everything a k=1 solve
// touches -- the SpMV's eight threads per row, the shared-memory reduction order, the
// (n, 1) addressing that collapses to plain indexing -- is what cast.cu did before these
// were shared, so solve_mixed's arithmetic is unchanged.

int cg_partials_size(int n) { return bcg_partials_blocks(n); }

void launch_double_to_float(const double* in, float* out, int n, cudaStream_t stream) {
    launch_bcg_d2f(n, in, out, stream);
}

void launch_float_to_double(const float* in, double* out, int n, cudaStream_t stream) {
    launch_bcg_f2d(n, in, out, stream);
}

void launch_cg_alpha(const double* rz, const double* pap, double* alpha, double* neg_alpha,
                     cudaStream_t stream) {
    launch_bcg_alpha(1, rz, pap, alpha, neg_alpha, stream);
}

void launch_cg_update_p(const double* beta, const float* zf, double* p, int n,
                        cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n)) {
        bcg_update_p_kernel<1><<<blocks, kBlock, 0, stream>>>(n, beta, zf, p);
    }
}

void launch_csrmv_f64(int n, const int* row_ptr, const int* col_idx, const double* vals,
                      const double* x, double* y, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * spmv_tpr<1>())) {
        bcsrmv_f64_kernel<1><<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, vals, x, y);
    }
}

void launch_cg_update_xr_norm(const double* alpha, const double* neg_alpha, const double* p,
                              const double* ap, double* x, double* r, float* rf,
                              double* partials, double* norm_sq, int n, cudaStream_t stream) {
    if (const unsigned n_blocks = grid_for(n)) {
        const int nb = static_cast<int>(n_blocks);
        bcg_update_xr_kernel<1>
            <<<n_blocks, kBlock, 0, stream>>>(n, alpha, neg_alpha, p, ap, x, r, rf, partials, nb);
        bcg_reduce_kernel<<<1, kBlock, 0, stream>>>(partials, nb, norm_sq);
    }
}

void launch_cg_cast_dot_beta(const float* zf, const double* r, double* partials, double* rz,
                             double* beta, int n, cudaStream_t stream) {
    if (const unsigned n_blocks = grid_for(n)) {
        const int nb = static_cast<int>(n_blocks);
        bcg_cast_dot_kernel<1><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, nb);
        bcg_reduce_beta_kernel<<<1, kBlock, 0, stream>>>(partials, nb, rz, beta);
    }
}
