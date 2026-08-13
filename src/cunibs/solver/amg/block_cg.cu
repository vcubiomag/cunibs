// Mixed-precision CG kernels, templated on the number of right-hand sides K, following the
// policy vcycle.cu sets: every step is written once and the single-RHS path is the K = 1
// instantiation. K chains run in lockstep sharing each matrix read. All dense operands are
// row-major (n, k) so the k values of one mesh node are contiguous; col_idx/vals are read
// once per nnz and reused across the k columns. Every reduction is a fixed-order two-stage
// tree per column, so results are run-to-run deterministic and each column's arithmetic is
// independent of its neighbours.
#include <cstdint>

#include "amg/cg_kernels.hpp"
#include "core/device_math.cuh"

namespace {

// Threads cooperating on one row in the strided SpMV. The reduced stiffness has ~14 nnz/row
// (P1 tets), so eight threads cover a row in a couple of strided steps. vcycle.cu tunes its
// fp32 counterpart separately.
constexpr int kTpr = 8;

// Blocks-per-SM floor for both fp64 SpMV kernels, which is what bounds the register budget ptxas
// will spend. Both are bandwidth-bound and want the residency for latency hiding; without the
// floor ptxas unrolls a wide instantiation until only two blocks fit per SM.
constexpr int kSpmvMinBlocks = 6;

template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    bcsrmv_f64_kernel(int n, const int* __restrict__ row_ptr,
                      const int* __restrict__ col_idx, const double* __restrict__ vals,
                      const double* __restrict__ x, double* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = threadIdx.x % kTpr;
    cuda::std::array<double, K> sum;
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0;
    if (row < n) {
        const WidthView<const double, K> xv(x, n);
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += kTpr) {
            const double v = vals[j];
            const double* xrow = &xv(col_idx[j], 0);
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * __ldg(xrow + c);
        }
    }
    warp_reduce_sum<kTpr>(sum);
    if (row < n && lane == 0) {
        const WidthView<double, K> yv(y, n);
#pragma unroll
        for (int c = 0; c < K; ++c) yv(row, c) = sum[c];
    }
}

// One thread per (row, column), which is what K >= 4 uses.
//
// The strided form above gives each thread a K-wide accumulator, so it issues K separate
// eight-byte loads at x[col * K + c] per nonzero, to addresses unrelated to its warp
// neighbours'. A warp then generates up to 32 distinct wavefronts for data occupying eight and
// the kernel saturates L1 request throughput well before DRAM. Giving each lane one column
// instead makes col_idx[j] and vals[j] broadcasts across a row's K lanes and turns the gather
// into one coalesced wavefront. Below K = 4 the strided form still wins, because two lanes per
// row cannot hide the row loop.
//
// A lane owns its column outright, so no cross-lane reduction is needed, but it still carries the
// strided kernel's kTpr accumulators and combines them with the same tree: which kernel runs is a
// throughput choice, and block_k must not move a result.
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
    const WidthView<const double, K> xv(x, n);
    const int row_s = row_ptr[row];
    const int row_e = row_ptr[row + 1];
    int j = row_s;
    for (; j + kTpr <= row_e; j += kTpr) {
#pragma unroll
        for (int l = 0; l < kTpr; ++l) part[l] += vals[j + l] * xv(col_idx[j + l], c);
    }
#pragma unroll
    for (int l = 0; l < kTpr; ++l) {
        if (j + l < row_e) part[l] += vals[j + l] * xv(col_idx[j + l], c);
    }

    // The same pairing warp_reduce_sum performs above, in registers rather than across lanes,
    // so both kernels land on ((p0+p4)+(p2+p6)) + ((p1+p5)+(p3+p7)).
#pragma unroll
    for (int off = kTpr / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int l = 0; l < off; ++l) part[l] += part[l + off];
    }
    WidthView<double, K>(y, n)(row, c) = part[0];
}

// Rows each thread of a reducing kernel carries. A warp's shuffle tree spends five fp64 adds per
// lane and per column to combine 32 values, and fp64 runs at a sixty-fourth of fp32 on a GeForce
// part, so the tree rather than the loads is what bounds these kernels: with the tree removed the
// r.z dot runs in 86 us against 150. Folding rows into a thread's accumulator first replaces
// eight trees with seven sequential adds and one tree. The rows a thread takes are a grid stride
// apart, so every access stays as coalesced as it was at one row per thread.
//
// Past eight the return is under a percent and the grid stops covering the SMs.
constexpr int kRowsPerThread = 8;

// Per-block reduction of K register accumulators into one partial per (column, block): warp
// shuffles, one cross-warp pass through shared memory, a single __syncthreads(). Keeping it to
// one barrier matters because the streaming kernels that embed it are bandwidth-bound.
//
// The order is the rows a thread folded in, then the shuffle tree within each warp, then a
// sequential sum over the kBlock/kWarp warp results. None of that depends on K, which is what
// makes the same placement solved at block_k=1 and block_k=8 come back bitwise identical.
template <int K>
__device__ __forceinline__ void bcg_block_reduce_cols(cuda::std::array<double, K>& local,
                                                      double* __restrict__ partials,
                                                      int n_blocks) {
    constexpr int kWarps = kBlock / kWarp;
    __shared__ double s[kWarps][K];
    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    warp_reduce_sum<kWarp>(local);
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < K; ++c) s[warp][c] = local[c];
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
    const WidthView<const double, K> xv(x, n);
    // Null on the norm path, where KDOT_XY folds the read away.
    const WidthView<const double, K> yv(y, n);
    cuda::std::array<double, K> local;
#pragma unroll
    for (int c = 0; c < K; ++c) local[c] = 0.0;
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < kRowsPerThread; ++t, i += stride) {
        if (i >= n) break;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            const double xc = xv(i, c);
            local[c] += KDOT_XY ? xc * yv(i, c) : xc * xc;
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

// A column freezes at the iteration its own residual first meets tolerance, rather than at the
// one the block's worst column reaches it. Both alpha and beta go to zero, which leaves x, r and
// rz untouched and pins p at the preconditioned residual, so the column contributes nothing
// further and its answer is the one it would have had solved on its own.
//
// Freezing on the device rather than by skipping launches on the host is what keeps the CG body
// replayable as a captured graph: the kernels are the same every iteration and only the mask's
// contents change. A null mask means no column is frozen, which the single-RHS solve passes.
__device__ __forceinline__ bool frozen(const double* __restrict__ converged, int c) {
    return converged != nullptr && converged[c] != 0.0;
}

__global__ void bcg_mark_converged_kernel(int k, const double* __restrict__ norm_sq,
                                          const double* __restrict__ ref_sq, double tolerance,
                                          double* __restrict__ converged) {
    const int c = threadIdx.x;
    if (c < k) {
        // Deliberately the same expression the host stopping test uses, rather than the cheaper
        // comparison of squares. fp64 sqrt and divide are both correctly rounded, so the two
        // sides agree bit for bit on the same inputs. If they could disagree by an ulp at the
        // boundary, a column the device had frozen but the host still called unconverged would
        // stop improving and run the block to max_iters.
        converged[c] = (sqrt(norm_sq[c]) / sqrt(ref_sq[c]) <= tolerance) ? 1.0 : 0.0;
    }
}

__global__ void bcg_alpha_kernel(int k, const double* __restrict__ rz,
                                 const double* __restrict__ pap,
                                 const double* __restrict__ converged, double* __restrict__ alpha,
                                 double* __restrict__ neg_alpha) {
    const int c = threadIdx.x;
    if (c < k) {
        // Branch rather than multiply by the mask: a column frozen at an exactly zero residual
        // has rz = pap = 0, and 0/0 * 0 would poison x with a NaN instead of leaving it alone.
        const double a = frozen(converged, c) ? 0.0 : rz[c] / pap[c];
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
    const WidthView<const double, K> prows(p, n);
    const WidthView<const double, K> aprows(ap, n);
    const WidthView<double, K> xrows(x, n);
    const WidthView<double, K> rrows(r, n);
    const WidthView<float, K> rfrows(rf, n);
    cuda::std::array<double, K> rloc;
#pragma unroll
    for (int c = 0; c < K; ++c) rloc[c] = 0.0;
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < kRowsPerThread; ++t, i += stride) {
        if (i >= n) break;
        cuda::std::array<double, K> pv, apv, xv, rv;
        load_row<K>(&prows(i, 0), pv);
        load_row<K>(&aprows(i, 0), apv);
        load_row<K>(&xrows(i, 0), xv);
        load_row<K>(&rrows(i, 0), rv);
        cuda::std::array<float, K> rfv;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            xv[c] += __ldg(alpha + c) * pv[c];
            rv[c] += __ldg(neg_alpha + c) * apv[c];
            rfv[c] = static_cast<float>(rv[c]);
            rloc[c] += rv[c] * rv[c];
        }
        store_row<K>(&xrows(i, 0), xv);
        store_row<K>(&rrows(i, 0), rv);
        store_row<K>(&rfrows(i, 0), rfv);
    }
    bcg_block_reduce_cols<K>(rloc, partials, n_blocks);
}

// The preconditioned residual is never materialized in fp64: consumers cast the fp32 zf on the
// fly, which is exact.
template <int K>
__global__ void bcg_cast_dot_kernel(int n, const float* __restrict__ zf,
                                    const double* __restrict__ r,
                                    double* __restrict__ partials, int n_blocks) {
    const WidthView<const double, K> rv(r, n);
    const WidthView<const float, K> zfv(zf, n);
    cuda::std::array<double, K> local;
#pragma unroll
    for (int c = 0; c < K; ++c) local[c] = 0.0;
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < kRowsPerThread; ++t, i += stride) {
        if (i >= n) break;
#pragma unroll
        for (int c = 0; c < K; ++c) local[c] += rv(i, c) * static_cast<double>(zfv(i, c));
    }
    bcg_block_reduce_cols<K>(local, partials, n_blocks);
}

__global__ void bcg_reduce_beta_kernel(const double* __restrict__ partials, int n_blocks,
                                       const double* __restrict__ converged,
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
        // beta = 0 on a frozen column pins p at zf. Left at total/rz it would be 1, since r has
        // not moved, and p would grow by zf every remaining iteration for no purpose.
        beta[c] = frozen(converged, c) ? 0.0 : total / rz[c];
        rz[c] = total;
    }
}

template <int K>
__global__ void bcg_update_p_kernel(int n, const double* __restrict__ beta,
                                    const float* __restrict__ zf, double* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const WidthView<double, K> pv(p, n);
        const WidthView<const float, K> zfv(zf, n);
#pragma unroll
        for (int c = 0; c < K; ++c) {
            pv(i, c) = __ldg(beta + c) * pv(i, c) + static_cast<double>(zfv(i, c));
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

int bcg_partials_blocks(int n) { return static_cast<int>(grid_for(n, kBlock * kRowsPerThread)); }

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

void launch_bcg_alpha(int k, const double* rz, const double* pap, const double* converged,
                      double* alpha, double* neg_alpha, cudaStream_t stream) {
    bcg_alpha_kernel<<<1, k, 0, stream>>>(k, rz, pap, converged, alpha, neg_alpha);
}

void launch_bcg_mark_converged(int k, const double* norm_sq, const double* ref_sq,
                               double tolerance, double* converged, cudaStream_t stream) {
    bcg_mark_converged_kernel<<<1, k, 0, stream>>>(k, norm_sq, ref_sq, tolerance, converged);
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
                              const double* converged, double* partials, double* rz, double* beta,
                              cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_cast_dot_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, zf, r, partials, n_blocks);
    });
    bcg_reduce_beta_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, converged, rz, beta);
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
