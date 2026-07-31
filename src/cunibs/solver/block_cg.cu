// Mixed-precision CG kernels, templated on the number of right-hand sides K, following the
// policy vcycle.cu sets: every step is written once and the single-RHS path is the K = 1
// instantiation. K chains run in lockstep sharing each matrix read. All dense operands are
// row-major (n, k) so the k values of one mesh node are contiguous; col_idx/vals are read
// once per nnz and reused across the k columns. Every reduction is a fixed-order two-stage
// tree per column, so results are run-to-run deterministic and each column's arithmetic is
// independent of its neighbours.
#include <cstdint>

#include "cg_kernels.hpp"

namespace {

// Threads cooperating on one row in the strided SpMV. The reduced stiffness has ~14 nnz/row
// (P1 tets), so eight threads cover a row in a couple of strided steps. This is the fp64 outer
// operator; vcycle.cu tunes its fp32 counterpart separately and lands on 4 for every K.
constexpr int kTpr = 8;

// Blocks-per-SM floor for both fp64 SpMV kernels, which is what bounds the register budget
// ptxas will spend. Both are purely bandwidth-bound and want the residency for latency hiding;
// six buys the strided kernel 40 registers per thread. Left to the default floor ptxas unrolls
// until a wide instantiation fits only two blocks per SM, which costs more than the unrolling
// wins. The column-lane kernel needs the floor just as much, since it holds kTpr fp64 partials
// per thread where the strided one spreads them across lanes.
constexpr int kSpmvMinBlocks = 6;

template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    bcsrmv_f64_kernel(int n, const int* __restrict__ row_ptr,
                      const int* __restrict__ col_idx, const double* __restrict__ vals,
                      const double* __restrict__ x, double* __restrict__ y) {
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

// One thread per (row, column), which is what K >= 4 uses.
//
// The strided form above gives each thread a K-wide accumulator, so it issues K separate
// eight-byte loads at x[col * K + c] per nonzero, to addresses unrelated to its warp
// neighbours'. A warp then generates up to 32 distinct wavefronts for data occupying eight, and
// the kernel saturates L1 request throughput well before DRAM: 91% of the STREAM ceiling at
// k=1 against 39% at k=8. Morton ordering cannot rescue it, having been tuned so the k=1 gather
// is cache-resident while the per-row working set here is K times larger.
//
// Giving each lane one column instead makes col_idx[j] and vals[j] broadcasts across a row's K
// lanes, and turns the gather into one coalesced wavefront. Below K = 4 the strided form still
// wins, because two lanes per row cannot hide the row loop.
//
// A lane owns its column outright, so no cross-lane reduction is needed, but it still carries the
// strided kernel's kTpr accumulators and combines them with the same tree. The summation order is
// part of the answer: which kernel runs is a throughput choice, and block_k must not move a
// result. Holding the partials in registers rather than across lanes costs kTpr doubles per
// thread and changes nothing about the loads.
template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    bcsrmv_f64_collane_kernel(int n, const int* __restrict__ row_ptr,
                              const int* __restrict__ col_idx, const double* __restrict__ vals,
                              const double* __restrict__ x, double* __restrict__ y) {
    const std::int64_t tid =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = static_cast<int>(tid / K);
    if (row >= n) return;
    const int c = static_cast<int>(tid % K);
    double part[kTpr];
#pragma unroll
    for (int l = 0; l < kTpr; ++l) part[l] = 0.0;

    // Walk the row in groups of kTpr so the accumulator index stays a compile-time constant;
    // a running index would spill `part` to local memory. Group g slot l is nonzero
    // row_s + kTpr*g + l, which is exactly the strided kernel's lane-l subsequence.
    const int row_s = row_ptr[row];
    const int row_e = row_ptr[row + 1];
    int j = row_s;
    for (; j + kTpr <= row_e; j += kTpr) {
#pragma unroll
        for (int l = 0; l < kTpr; ++l) {
            part[l] += vals[j + l] * x[static_cast<std::int64_t>(col_idx[j + l]) * K + c];
        }
    }
#pragma unroll
    for (int l = 0; l < kTpr; ++l) {
        if (j + l < row_e) {
            part[l] += vals[j + l] * x[static_cast<std::int64_t>(col_idx[j + l]) * K + c];
        }
    }

    // The same pairing __shfl_down_sync performs above, so both kernels land on
    // ((p0+p4)+(p2+p6)) + ((p1+p5)+(p3+p7)).
#pragma unroll
    for (int off = kTpr / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int l = 0; l < off; ++l) part[l] += part[l + off];
    }
    y[static_cast<std::int64_t>(row) * K + c] = part[0];
}

// Per-block reduction of K register accumulators into one partial per (column, block): warp
// shuffles, one cross-warp pass through shared memory, a single __syncthreads(). Keeping it to
// one barrier matters because the streaming kernels that embed it are bandwidth-bound.
//
// The order is the shuffle tree within each warp, then a sequential sum over the kBlock/kWarp
// warp results. It does not depend on K, which is what makes the same placement solved at
// block_k=1 and block_k=8 come back bitwise identical.
template <int K>
__device__ __forceinline__ void bcg_block_reduce_cols(double (&local)[K],
                                                      double* __restrict__ partials,
                                                      int n_blocks) {
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

constexpr const char* kBadK = "block CG supports k in {1, 2, 4, 8}";

}  // namespace

int bcg_partials_blocks(int n) { return static_cast<int>(grid_for(n)); }

void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const int* col_idx,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream) {
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        constexpr int kThreadsPerRow = (K >= 4) ? K : kTpr;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * kThreadsPerRow)) {
            if constexpr (K >= 4) {
                bcsrmv_f64_collane_kernel<K>
                    <<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, vals, x, y);
            } else {
                bcsrmv_f64_kernel<K>
                    <<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, vals, x, y);
            }
        }
    });
}

void launch_bcg_dot(int n, int k, const double* x, const double* y, double* partials,
                    double* out, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, true>
            <<<n_blocks, kBlock, 0, stream>>>(n, x, y, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, out);
}

void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
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
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
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
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_cast_dot_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, n_blocks);
    });
    bcg_reduce_beta_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, rz, beta);
}

void launch_bcg_cast_dot_init(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_cast_dot_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, rz);
}

void launch_bcg_update_p(int n, int k, const double* beta, const float* zf, double* p,
                         cudaStream_t stream) {
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
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
