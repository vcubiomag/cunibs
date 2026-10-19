// Every V-cycle step is written once, templated on the number of right-hand sides K, and
// the single-RHS cycle is the K = 1 instantiation: for K = 1 the row-major (n, K) operands
// collapse to plain vectors and the compiler folds the column loops away.
#include "amg/vcycle.hpp"
#include "amg/slot_csr.hpp"
#include "core/device_math.cuh"
#include "core/slot_csr.cuh"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

template <typename T>
using Buffer = DeviceBuffer<T>;

template <typename T>
Buffer<T> device_alloc(std::size_t count, const char* what) {
    return ::device_alloc<T>(count, "vcycle", what);
}

template <typename T>
Buffer<T> device_clone(const T* src, std::size_t count, cudaStream_t stream, const char* what) {
    return ::device_clone<T>(src, count, stream, "vcycle", what);
}

// Threads per row for the SpMV-shaped kernels, chosen per level from that level's own shape. The
// hierarchy is not self-similar: every Galerkin product widens the stencil, so nnz/row climbs from
// ~14 at the fine level into the hundreds at the coarsest while the row count collapses, and one
// split cannot serve that spread.
//
// Widening trades the two halves of the kernel against each other. The matrix half improves: a
// row's lanes read TPR consecutive fp16 values, and below a 32-byte sector that is a partial
// transaction. The epilogue half degrades: only lane 0 writes, and it touches six fp32 streams
// indexed by row, so halving the active lanes per warp halves the coalescing on all of them. At
// the fine level the epilogue is about a fifth of the row's traffic and its loss outweighs the
// matrix gain; by the coarsest it is ~1% and the matrix decides alone.
//
// Two lanes is the fine level's choice once its columns are 2-byte slots: measured on a head
// mesh, level-0 widths 1 / 2 / 4 / 8 give 19.2 / 18.4 / 19.4 / 21.5 ms per placement at k = 1 and
// 7.60 / 6.90 / 7.06 / 7.37 at k = 8.
constexpr int kTprChoices[] = {2, 4, 8, 16, 32};

// Strided steps of matrix work per thread needed before widening pays for what the epilogue loses.
constexpr int kMinRowIters = 4;

// Blocks per SM to cover when a level is too small to fill the GPU, where idle SMs cost more than
// either side of that trade. Past this, latency hiding stops improving on this shape of kernel.
constexpr int kTargetBlocksPerSm = 4;

// Every choice has to hold the invariant, not just the narrowest: a width that did not divide the
// warp would put one row's lanes across two shuffle groups and silently drop part of its sum.
static_assert(std::ranges::all_of(kTprChoices,
                                  [](int tpr) { return kWarp % tpr == 0 && kBlock % tpr == 0; }),
              "every threads-per-row choice must divide both the warp and the block");

// One warp fraction per row: strided partial products over the row's nonzeros, then a
// fixed-order shuffle reduction leaving the row total in lane 0. Threads whose row is past the
// end still reach the shuffle with zero accumulators, which is why the mask is the full warp.
// The fixed order is what keeps an apply run-to-run deterministic.
//
// TPR is a template argument so the strided loop and the shuffle tree fold. It is a property of
// the level's operator and never of K, which is what keeps a placement's result identical at every
// block width.
template <int K, int TPR>
__device__ __forceinline__ void row_spmv(int n, const int* __restrict__ row_ptr,
                                         const std::int16_t* __restrict__ cols,
                                         const __half* __restrict__ vals,
                                         const float* __restrict__ x, int row, int lane,
                                         cuda::std::array<float, K>& sum) {
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    if (row < n) {
        const WidthView<const float, K> xv_rows(x, n);
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += TPR) {
            const float v = __half2float(vals[j]);
            cuda::std::array<float, K> xv;
            load_row<K>(&xv_rows(slot_column(cols, j, row), 0), xv);
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * xv[c];
        }
    }
    warp_reduce_sum<TPR>(sum);
}

// Whether the operator kernels issue the row's own operands before the gather rather than after
// the reduction. Single width only: there the row-local loads are a few words whose latency the
// gather can hide, measured 2.8% off the k = 1 solve; at k = 8 they are 24 live floats per row
// and the occupancy they cost is worth more, +1.4%. Where the loads land never moves a result.
template <int K>
constexpr bool kHoistEpilogue = (K == 1);

// Threads per row for one level, from that level's shape and the device's SM count: independent
// of k, so the same operator always gets the same split and a solve reproduces across block
// widths.
inline int choose_tpr(int n_rows, int nnz, int n_sm) {
    if (n_rows <= 0) return kTprChoices[0];
    const std::int64_t nnz_per_row = (static_cast<std::int64_t>(nnz) + n_rows - 1) / n_rows;
    const std::int64_t to_fill = static_cast<std::int64_t>(n_sm) * kTargetBlocksPerSm * kBlock;
    int tpr = kTprChoices[0];
    for (std::size_t i = 1; i < std::size(kTprChoices); ++i) {
        // Widen while the matrix half still has enough work per thread to pay for what the
        // epilogue loses, or while the level is too small to fill the machine.
        const bool matrix_pays = kTprChoices[i] * kMinRowIters <= nnz_per_row;
        const bool starved = static_cast<std::int64_t>(n_rows) * tpr < to_fill;
        if (!matrix_pays && !starved) break;
        tpr = kTprChoices[i];
    }
    return tpr;
}

// n_total = n * K throughout the elementwise kernels; K is a power of two, so the row and
// column of a linear index come out as a shift and a mask on an unsigned index.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_jacobi_zero_kernel(std::uint64_t n_total, const float* __restrict__ dinv,
                          const float* __restrict__ b, float alpha, float* __restrict__ x) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) x[i] = alpha * dinv[i / K] * b[i];
}

template <int K, int TPR>
__global__ void __launch_bounds__(kBlock)
    vc_residual_kernel(int n, const int* __restrict__ row_ptr,
                       const std::int16_t* __restrict__ cols, const __half* __restrict__ vals,
                       const float* __restrict__ row_scale, const float* __restrict__ x,
                       const float* __restrict__ b, float* __restrict__ r) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / TPR;
    const int lane = static_cast<int>(threadIdx.x) % TPR;
    const bool owner = row < n && lane == 0;
    cuda::std::array<float, K> rhs;
    float s = 0.0f;
    const auto load_operands = [&] {
        load_row<K>(&WidthView<const float, K>(b, n)(row, 0), rhs);
        s = row_scale[row];
    };
    if (kHoistEpilogue<K> && owner) load_operands();
    cuda::std::array<float, K> sum;
    row_spmv<K, TPR>(n, row_ptr, cols, vals, x, row, lane, sum);
    if (owner) {
        if (!kHoistEpilogue<K>) load_operands();
        cuda::std::array<float, K> res;
#pragma unroll
        for (int c = 0; c < K; ++c) res[c] = rhs[c] - s * sum[c];
        store_row<K>(&WidthView<float, K>(r, n)(row, 0), res);
    }
}

// Threads per coarse row in the restriction. R = P^T has by far the widest rows in the
// hierarchy and the fewest of them, so a whole warp per row keeps the coarse end from starving
// for parallelism and the fine end from walking each row alone. Unlike the operator's tpr this
// needs no per-level choice: R's rows are wide at every level, its values are fp32 so a warp's
// segment is already several sectors, and consecutive entries of a restriction row are
// consecutive fine indices.
constexpr int kRestrictTpr = 32;

static_assert(kWarp % kRestrictTpr == 0, "shuffle width must divide the warp");
static_assert(kBlock % kRestrictTpr == 0, "a block must hold a whole number of rows");

// b_coarse = R r_fine. R is sorted by column index at setup, and each lane takes a fixed
// strided slice reduced in a fixed shuffle order, so the sum order is pinned by the sizes alone
// and the apply stays deterministic. Threads past the last row still reach the shuffle with
// zero accumulators, which is why the mask is the full warp.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_restrict_kernel(int n_coarse, const int* __restrict__ r_row_ptr,
                       const int* __restrict__ r_col_idx, const float* __restrict__ r_vals,
                       const float* __restrict__ r_fine, float* __restrict__ b_coarse) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / kRestrictTpr;
    const int lane = static_cast<int>(threadIdx.x) % kRestrictTpr;
    cuda::std::array<float, K> sum;
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    if (row < n_coarse) {
        const auto fine = unsized<WidthView<const float, K>>(r_fine);
        const int row_e = r_row_ptr[row + 1];
        for (int j = r_row_ptr[row] + lane; j < row_e; j += kRestrictTpr) {
            const float v = r_vals[j];
            cuda::std::array<float, K> xv;
            load_row<K>(&fine(r_col_idx[j], 0), xv);
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * xv[c];
        }
    }
    warp_reduce_sum<kRestrictTpr>(sum);
    if (row < n_coarse && lane == 0) {
        const WidthView<float, K> out(b_coarse, n_coarse);
#pragma unroll
        for (int c = 0; c < K; ++c) out(row, c) = sum[c];
    }
}

// x_fine += P x_coarse, one thread per fine row. A smoothed P holds a handful of nonzeros per
// row, one per aggregate that row's neighbourhood reaches.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_prolongate_kernel(int n, const int* __restrict__ p_row_ptr,
                         const int* __restrict__ p_col_idx, const float* __restrict__ p_vals,
                         const float* __restrict__ x_coarse, float* __restrict__ x_fine) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;
    const int row_e = p_row_ptr[row + 1];
    cuda::std::array<float, K> sum;
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    const auto coarse = unsized<WidthView<const float, K>>(x_coarse);
    const WidthView<float, K> fine(x_fine, n);
    for (int j = p_row_ptr[row]; j < row_e; ++j) {
        const float v = p_vals[j];
        cuda::std::array<float, K> xv;
        load_row<K>(&coarse(p_col_idx[j], 0), xv);
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += v * xv[c];
    }
#pragma unroll
    for (int c = 0; c < K; ++c) fine(row, c) += sum[c];
}

// One Chebyshev step, fused with the residual it needs:
//     x_out = c_cur * x_cur + c_prev * x_prev + alpha * dinv * (b - A x_cur).
//
// Out-of-place in x_cur, because rows gather it across the whole vector. x_prev is read only
// at the row this thread owns, so x_out is allowed to alias x_prev, and from the second step of
// a sweep on it always does: that is what lets any degree run on the level's two existing
// buffers with no third one.
//
// x_prev and x_out therefore must NOT be __restrict__, however tempting it looks next to the
// pointers that are: restrict on x_out would promise the compiler it aliases nothing, which is
// exactly the promise this call pattern breaks, and it would be free to hoist the K stores
// above the x_prev loads. The row's values are staged into registers first so the read/write
// order is explicit rather than resting on alias analysis at all. x_cur keeps its restrict: it
// is the gather's hot pointer, and x_out is never the buffer it reads.
//
// A zero c_prev does not excuse passing a pointer at unwritten memory, because 0 * NaN is NaN.
// See the caller.
//
// The coefficients also cover the degenerate case (c_cur, 0, alpha), the first recurrence step
// of a zero-initial-guess sweep, so this is the only smoother kernel there is.
template <int K, int TPR>
__global__ void __launch_bounds__(kBlock)
    vc_cheby_step_kernel(int n, const int* __restrict__ row_ptr,
                         const std::int16_t* __restrict__ cols,
                         const __half* __restrict__ vals, const float* __restrict__ row_scale,
                         const float* __restrict__ dinv, const float* __restrict__ b,
                         const float* __restrict__ x_cur, const float* x_prev, float c_cur,
                         float c_prev, float alpha, float* x_out) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / TPR;
    const int lane = static_cast<int>(threadIdx.x) % TPR;
    const bool owner = row < n && lane == 0;
    // x_prev is read before this row's x_out is written, wherever the load lands, which is all
    // its aliasing needs.
    cuda::std::array<float, K> cur;
    cuda::std::array<float, K> prev;
    cuda::std::array<float, K> rhs;
    float d = 0.0f;
    float s = 0.0f;
    const auto load_operands = [&] {
        load_row<K>(&WidthView<const float, K>(x_cur, n)(row, 0), cur);
        load_row<K>(&WidthView<const float, K>(x_prev, n)(row, 0), prev);
        load_row<K>(&WidthView<const float, K>(b, n)(row, 0), rhs);
        d = alpha * dinv[row];
        s = row_scale[row];
    };
    if (kHoistEpilogue<K> && owner) load_operands();
    cuda::std::array<float, K> sum;
    row_spmv<K, TPR>(n, row_ptr, cols, vals, x_cur, row, lane, sum);
    if (owner) {
        if (!kHoistEpilogue<K>) load_operands();
        // Plain, not __restrict__: x_out is allowed to alias x_prev, and from the second step
        // of a sweep it always does.
        cuda::std::array<float, K> out;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            out[c] = c_cur * cur[c] + c_prev * prev[c] + d * (rhs[c] - s * sum[c]);
        }
        store_row<K>(&WidthView<float, K>(x_out, n)(row, 0), out);
    }
}

// The first two steps of a sweep from a zero initial guess in one pass. The first step is the
// Jacobi point x1 = alpha0 D^-1 b and needs no SpMV, so the second step's gather forms each
// neighbour's x1 on the fly from b and dinv instead of reading a materialised x1. The expression
// is vc_jacobi_zero_kernel's, so the result is the same bit for bit; what goes is that kernel's
// launch, its write of x1 and the step's re-read of it. x1_out, when not null, still receives x1,
// for a degree high enough that a later step needs it as x_prev.
template <int K, int TPR>
__global__ void __launch_bounds__(kBlock)
    vc_cheby_first_kernel(int n, const int* __restrict__ row_ptr,
                          const std::int16_t* __restrict__ cols,
                          const __half* __restrict__ vals, const float* __restrict__ row_scale,
                          const float* __restrict__ dinv, const float* __restrict__ b,
                          float alpha0, float c_cur, float alpha,
                          float* __restrict__ x1_out, float* __restrict__ x_out) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / TPR;
    const int lane = static_cast<int>(threadIdx.x) % TPR;
    const bool owner = row < n && lane == 0;
    const WidthView<const float, K> bv(b, n);
    cuda::std::array<float, K> rhs;
    float dr = 0.0f;
    float s = 0.0f;
    const auto load_operands = [&] {
        load_row<K>(&bv(row, 0), rhs);
        dr = dinv[row];
        s = row_scale[row];
    };
    if (kHoistEpilogue<K> && owner) load_operands();
    cuda::std::array<float, K> sum;
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += TPR) {
            const float v = __half2float(vals[j]);
            const int col = slot_column(cols, j, row);
            const float w = alpha0 * dinv[col];
            cuda::std::array<float, K> bj;
            load_row<K>(&bv(col, 0), bj);
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * (w * bj[c]);
        }
    }
    warp_reduce_sum<TPR>(sum);
    if (owner) {
        if (!kHoistEpilogue<K>) load_operands();
        const float w = alpha0 * dr;
        const float d = alpha * dr;
        cuda::std::array<float, K> x1;
#pragma unroll
        for (int c = 0; c < K; ++c) x1[c] = w * rhs[c];
        const WidthView<float, K> outv(x_out, n);
        // Spelled out so that every K contracts the same way: c_cur x1 rounds on its own.
#pragma unroll
        for (int c = 0; c < K; ++c) {
            outv(row, c) =
                __fmaf_rn(d, __fmaf_rn(-s, sum[c], rhs[c]), __fmul_rn(c_cur, x1[c]));
        }
        if (x1_out != nullptr) store_row<K>(&WidthView<float, K>(x1_out, n)(row, 0), x1);
    }
}

// One warp per row of the dense inverse, strided then reduced in a fixed shuffle order. Threads
// whose row is past the end still reach the shuffle with zero accumulators, which is why the mask
// is the full warp.
//
// The coarsest system is only a few hundred rows, but it is one launch on the critical path of
// every V-cycle. A thread per row would read ainv[row * n + j] with a warp's lanes n floats apart,
// a separate sector per useful value; striding a warp along the row makes each load one
// contiguous 128-byte transaction.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_coarse_gemv_kernel(int n, const float* __restrict__ ainv, const float* __restrict__ b,
                          float* __restrict__ x) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    cuda::std::array<float, K> sum;
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    if (row < n) {
        const Mat2View<const float> a_inv(ainv, n, n);
        const WidthView<const float, K> brows(b, n);
        for (int j = lane; j < n; j += kWarp) {
            const float a = a_inv(row, j);
            cuda::std::array<float, K> bv;
            load_row<K>(&brows(j, 0), bv);
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += a * bv[c];
        }
    }
    warp_reduce_sum<kWarp>(sum);
    if (row < n && lane == 0) {
        const WidthView<float, K> out(x, n);
#pragma unroll
        for (int c = 0; c < K; ++c) out(row, c) = sum[c];
    }
}

// The launchers deliberately do not check cudaGetLastError(). A cycle runs inside the CG
// stream capture, and throwing there would skip cudaStreamEndCapture and leave the stream
// capturing for good; PcgAmgSolver already handles a failed capture, and anything a launch
// could report surfaces at its next synchronize. Grid dimensions are checked here instead,
// which is the only launch failure this code can cause on its own.
template <int K>
void launch_jacobi_zero(int n, const float* dinv, const float* b, float alpha, float* x,
                        cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(n) * K;
    if (const unsigned blocks = grid_for(total)) {
        vc_jacobi_zero_kernel<K><<<blocks, kBlock, 0, stream>>>(
            static_cast<std::uint64_t>(total), dinv, b, alpha, x);
    }
}

// Turns a level's threads-per-row into the template argument the SpMV-shaped kernels take, over
// exactly the set choose_tpr picks from, so an unmatched value is a disagreement between the two
// and not something a caller can provoke.
template <typename F>
void dispatch_tpr(int tpr, F&& f) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        dispatch_k<kTprChoices[I]...>(tpr, "V-cycle threads-per-row outside kTprChoices",
                                      std::forward<F>(f));
    }(std::make_index_sequence<std::size(kTprChoices)>{});
}

template <int K>
void launch_residual(int n, int tpr, const int* row_ptr, const std::int16_t* cols,
                     const __half* values,
                     const float* row_scale, const float* x, const float* b, float* r,
                     cudaStream_t stream) {
    dispatch_tpr(tpr, [&](auto tc) {
        constexpr int TPR = decltype(tc)::value;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * TPR)) {
            vc_residual_kernel<K, TPR>
                <<<blocks, kBlock, 0, stream>>>(n, row_ptr, cols, values, row_scale, x, b, r);
        }
    });
}

template <int K>
void launch_restrict(int n_coarse, const int* r_row_ptr, const int* r_col_idx,
                     const float* r_vals, const float* r_fine, float* b_coarse,
                     cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n_coarse) * kRestrictTpr)) {
        vc_restrict_kernel<K><<<blocks, kBlock, 0, stream>>>(n_coarse, r_row_ptr, r_col_idx,
                                                             r_vals, r_fine, b_coarse);
    }
}

template <int K>
void launch_prolongate(int n, const int* p_row_ptr, const int* p_col_idx, const float* p_vals,
                       const float* x_coarse, float* x_fine, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n)) {
        vc_prolongate_kernel<K><<<blocks, kBlock, 0, stream>>>(n, p_row_ptr, p_col_idx, p_vals,
                                                               x_coarse, x_fine);
    }
}

template <int K>
void launch_cheby_step(int n, int tpr, const int* row_ptr, const std::int16_t* cols,
                       const __half* values, const float* row_scale, const float* dinv,
                       const float* b, const float* x_cur, const float* x_prev, float c_cur,
                       float c_prev, float alpha, float* x_out, cudaStream_t stream) {
    dispatch_tpr(tpr, [&](auto tc) {
        constexpr int TPR = decltype(tc)::value;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * TPR)) {
            vc_cheby_step_kernel<K, TPR><<<blocks, kBlock, 0, stream>>>(
                n, row_ptr, cols, values, row_scale, dinv, b, x_cur, x_prev, c_cur, c_prev,
                alpha, x_out);
        }
    });
}

template <int K>
void launch_cheby_first(int n, int tpr, const int* row_ptr, const std::int16_t* cols,
                        const __half* values, const float* row_scale, const float* dinv,
                        const float* b, float alpha0, float c_cur, float alpha,
                        float* x1_out, float* x_out, cudaStream_t stream) {
    dispatch_tpr(tpr, [&](auto tc) {
        constexpr int TPR = decltype(tc)::value;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * TPR)) {
            vc_cheby_first_kernel<K, TPR><<<blocks, kBlock, 0, stream>>>(
                n, row_ptr, cols, values, row_scale, dinv, b, alpha0, c_cur, alpha,
                x1_out, x_out);
        }
    });
}

template <int K>
void launch_coarse_gemv(int n, const float* ainv, const float* b, float* x,
                        cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * kWarp)) {
        vc_coarse_gemv_kernel<K><<<blocks, kBlock, 0, stream>>>(n, ainv, b, x);
    }
}

// Setup-only: one thread per row takes the row's max |a_ij| as its scale, then lays the row out
// in its slots as a_ij / scale in fp16, with a zero in each far column's low-half slot. An
// all-zero row keeps a scale of 1 so the reciprocal stays finite.
__global__ void __launch_bounds__(kBlock)
    vc_pack_values_kernel(int n, const int* __restrict__ row_ptr,
                          const int* __restrict__ slot_ptr, const std::int16_t* __restrict__ cols,
                          const float* __restrict__ src, __half* __restrict__ dst,
                          float* __restrict__ row_scale) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;
    const int row_b = row_ptr[row];
    const int row_e = row_ptr[row + 1];
    float scale = 0.0f;
    for (int j = row_b; j < row_e; ++j) scale = fmaxf(scale, fabsf(src[j]));
    if (!(scale > 0.0f)) scale = 1.0f;
    row_scale[row] = scale;
    const float inv = 1.0f / scale;
    int j = row_b;
    for (int s = slot_ptr[row]; s < slot_ptr[row + 1]; ++s) {
        dst[s] = (cols[s] > kSlotNearMax) ? __float2half(0.0f) : __float2half(src[j++] * inv);
    }
}

constexpr const char* kBadK = "V-cycle block apply supports k in {1, 2, 4, 8}";

std::atomic<int> g_vcycle_generation{0};

}  // namespace

NativeVCycle::NativeVCycle() : generation_(++g_vcycle_generation) {}

// Chebyshev over [1 / lower_ratio, 1], rewritten from the usual d-recurrence into the
// x-recurrence the step kernel takes:
//     x_{i+1} = (1 + beta_i) x_i - beta_i x_{i-1} + alpha_i D^-1 (b - A x_i).
//
// The interval's top is 1 because dinv carries the l1 diagonal, for which rho(D^-1 A) <= 1 holds
// analytically on any SPD operator (D - A is weakly diagonally dominant with a non-negative
// diagonal, hence PSD). That is a bound, not an estimate, so alpha0 = 2r/(r+1) < 2 can never
// reach 2/rho: the smoother is A-convergent and the cycle SPD on every level and every mesh.
// An estimated rho would be the footgun, since an under-estimate over-relaxes the smoother
// badly enough to stall the cycle.
void NativeVCycle::set_smoother(int degree, float lower_ratio) {
    if (!levels_.empty()) {
        throw std::runtime_error("NativeVCycle: set_smoother after add_level");
    }
    if (degree < 1 || degree > kMaxSmootherDegree) {
        throw std::invalid_argument("NativeVCycle: smoother degree out of range");
    }
    // At ratio 1 the interval collapses to a point, delta is 0 and sigma is infinite.
    if (!(lower_ratio > 1.0f)) {
        throw std::invalid_argument("NativeVCycle: smoother lower_ratio must exceed 1");
    }

    constexpr float hi = 1.0f;
    const float lo = hi / lower_ratio;
    const float theta = 0.5f * (hi + lo);
    const float delta = 0.5f * (hi - lo);
    const float sigma = theta / delta;
    cheby_ = Cheby{};
    cheby_.alpha0 = 1.0f / theta;
    float rho = 1.0f / sigma;
    for (int j = 0; j + 1 < degree; ++j) {
        const float rho_next = 1.0f / (2.0f * sigma - rho);
        const float beta = rho_next * rho;
        cheby_.c_cur[j] = 1.0f + beta;
        cheby_.c_prev[j] = -beta;
        cheby_.alpha[j] = 2.0f * rho_next / delta;
        rho = rho_next;
    }
    degree_ = degree;
    smoother_set_ = true;
}

void NativeVCycle::add_level(int n_rows, int nnz, int n_coarse, int p_nnz, const int* row_ptr,
                             const int* col_idx, const float* values, const float* dinv,
                             const int* p_row_ptr, const int* p_col_idx, const float* p_values,
                             const int* r_row_ptr, const int* r_col_idx,
                             const float* r_values, cudaStream_t stream) {
    if (finalized_) throw std::runtime_error("NativeVCycle: add_level after finalize");
    if (!smoother_set_) throw std::runtime_error("NativeVCycle: add_level before set_smoother");
    if (n_rows <= 0 || n_coarse <= 0 || nnz < 0 || p_nnz < 0) {
        throw std::invalid_argument("NativeVCycle: level dimensions must be positive");
    }

    Level lvl;
    lvl.n = n_rows;
    lvl.n_coarse = n_coarse;
    lvl.tpr = choose_tpr(n_rows, nnz, sm_count());
    SlotCsr slots = build_slot_csr(n_rows, row_ptr, col_idx, stream, "vcycle");
    lvl.row_ptr = std::move(slots.ptr);
    lvl.cols = std::move(slots.cols);
    lvl.values = device_alloc<__half>(static_cast<std::size_t>(slots.n_slots), "level values");
    lvl.row_scale = device_alloc<float>(static_cast<std::size_t>(n_rows), "level row_scale");
    if (const unsigned blocks = grid_for(n_rows)) {
        vc_pack_values_kernel<<<blocks, kBlock, 0, stream>>>(
            n_rows, row_ptr, lvl.row_ptr.get(), lvl.cols.get(), values, lvl.values.get(),
            lvl.row_scale.get());
        check_cuda(cudaGetLastError(), "vcycle", "pack values");
    }
    lvl.dinv = device_clone(dinv, static_cast<std::size_t>(n_rows), stream, "level dinv");
    lvl.p_row_ptr =
        device_clone(p_row_ptr, static_cast<std::size_t>(n_rows) + 1, stream, "level P ptr");
    lvl.p_col_idx =
        device_clone(p_col_idx, static_cast<std::size_t>(p_nnz), stream, "level P idx");
    lvl.p_values = device_clone(p_values, static_cast<std::size_t>(p_nnz), stream, "level P val");
    lvl.r_row_ptr =
        device_clone(r_row_ptr, static_cast<std::size_t>(n_coarse) + 1, stream, "level R ptr");
    lvl.r_col_idx =
        device_clone(r_col_idx, static_cast<std::size_t>(p_nnz), stream, "level R idx");
    lvl.r_values = device_clone(r_values, static_cast<std::size_t>(p_nnz), stream, "level R val");
    check_cuda(cudaStreamSynchronize(stream), "vcycle", "level copies");
    // The finest level reads the caller's b in place, so it needs no RHS of its own.
    if (!levels_.empty()) {
        lvl.b = device_alloc<float>(static_cast<std::size_t>(n_rows), "level b");
    }
    lvl.x = device_alloc<float>(static_cast<std::size_t>(n_rows), "level x");
    lvl.r = device_alloc<float>(static_cast<std::size_t>(n_rows), "level r");
    levels_.push_back(std::move(lvl));
}

void NativeVCycle::set_coarse(int n, const float* ainv, cudaStream_t stream) {
    if (finalized_) throw std::runtime_error("NativeVCycle: set_coarse after finalize");
    if (n <= 0) throw std::invalid_argument("NativeVCycle: coarse size must be positive");
    coarse_ainv_ = device_clone(ainv, static_cast<std::size_t>(n) * n, stream, "coarse ainv");
    check_cuda(cudaStreamSynchronize(stream), "vcycle", "coarse copy");
    coarse_b_ = device_alloc<float>(static_cast<std::size_t>(n), "coarse b");
    coarse_x_ = device_alloc<float>(static_cast<std::size_t>(n), "coarse x");
    coarse_n_ = n;
}

void NativeVCycle::finalize() {
    if (!coarse_ainv_) throw std::runtime_error("NativeVCycle: finalize without set_coarse");
    for (std::size_t i = 0; i < levels_.size(); ++i) {
        const int expected = (i + 1 < levels_.size()) ? levels_[i + 1].n : coarse_n_;
        if (levels_[i].n_coarse != expected) {
            throw std::runtime_error("NativeVCycle: inconsistent level dimensions");
        }
    }
    finalized_ = true;
}

// Buffers sized for a larger k serve a smaller one: a cycle only ever touches the first
// n * k entries of each. Growing k reallocates, which is why callers warm up before CUDA
// graph capture (allocation inside a capture fails).
void NativeVCycle::ensure_block_buffers(int k) {
    if (block_k_ >= k) return;
    // A captured graph holds the buffers freed below, so it must not replay against this cycle.
    generation_ = ++g_vcycle_generation;
    block_k_ = 0;
    for (std::size_t i = 0; i < levels_.size(); ++i) {
        Level& lvl = levels_[i];
        const std::size_t count = static_cast<std::size_t>(lvl.n) * k;
        lvl.bk.reset();
        lvl.xk.reset();
        lvl.rk.reset();
        if (i != 0) lvl.bk = device_alloc<float>(count, "level bk");
        lvl.xk = device_alloc<float>(count, "level xk");
        lvl.rk = device_alloc<float>(count, "level rk");
    }
    const std::size_t coarse_count = static_cast<std::size_t>(coarse_n_) * k;
    coarse_bk_.reset();
    coarse_xk_.reset();
    coarse_bk_ = device_alloc<float>(coarse_count, "coarse bk");
    coarse_xk_ = device_alloc<float>(coarse_count, "coarse xk");
    block_k_ = k;
}

void NativeVCycle::check_ready(int n, const char* what) const {
    if (!finalized_) {
        throw std::runtime_error(std::string("NativeVCycle: ") + what + " before finalize");
    }
    const int expected = levels_.empty() ? coarse_n_ : levels_[0].n;
    if (n != expected) throw std::runtime_error("NativeVCycle: size mismatch");
}

template <int K>
void NativeVCycle::run_cycle(const float* b, float* x, cudaStream_t stream) {
    constexpr bool kSingle = (K == 1);
    const auto level_b = [](Level& l) { return kSingle ? l.b.get() : l.bk.get(); };
    const auto level_x = [](Level& l) { return kSingle ? l.x.get() : l.xk.get(); };
    const auto level_r = [](Level& l) { return kSingle ? l.r.get() : l.rk.get(); };
    float* const coarse_b = kSingle ? coarse_b_.get() : coarse_bk_.get();
    float* const coarse_x = kSingle ? coarse_x_.get() : coarse_xk_.get();

    if (levels_.empty()) {
        launch_coarse_gemv<K>(coarse_n_, coarse_ainv_.get(), b, x, stream);
        return;
    }

    // One Chebyshev sweep of degree_ steps, returning the buffer the result landed in. Each
    // step gathers x_cur across the whole vector, so it cannot write what it reads: writes
    // alternate between `first` and `other`. From the second step on the output is the buffer
    // that held x_{i-1}, which is legal because x_prev is read only at the row being written.
    //
    // x0 == nullptr is a zero initial guess. That drops the first step's SpMV altogether, and
    // from degree 2 up the first two steps run as one kernel that never reads `other`, so no
    // zeroed buffer is needed to multiply by.
    //
    // final_out, when set, takes the last step in place of the alternating buffers. Only level
    // 0 on the up-sweep uses it, to write the caller's x directly.
    const auto smooth = [&](Level& lvl, const float* bi, const float* x0, float* first,
                            float* other, float* final_out) {
        const auto target = [&](int s) {
            if (s + 1 == degree_ && final_out) return final_out;
            return (s % 2) ? other : first;
        };
        int first_step = 1;
        float* cur = target(0);
        float* prev = other;
        if (x0 == nullptr && degree_ > 1) {
            // Steps 0 and 1 fused; x1 is only materialised when step 2 will read it as x_prev.
            // Step 1's x_prev term drops out because x0 is zero.
            launch_cheby_first<K>(lvl.n, lvl.tpr, lvl.row_ptr.get(), lvl.cols.get(),
                                  lvl.values.get(), lvl.row_scale.get(), lvl.dinv.get(), bi,
                                  cheby_.alpha0, cheby_.c_cur[0], cheby_.alpha[0],
                                  degree_ > 2 ? target(0) : nullptr, target(1), stream);
            prev = target(0);
            cur = target(1);
            first_step = 2;
        } else if (x0 == nullptr) {
            launch_jacobi_zero<K>(lvl.n, lvl.dinv.get(), bi, cheby_.alpha0, target(0), stream);
        } else {
            launch_cheby_step<K>(lvl.n, lvl.tpr, lvl.row_ptr.get(), lvl.cols.get(),
                                 lvl.values.get(), lvl.row_scale.get(), lvl.dinv.get(), bi, x0,
                                 x0, 1.0f, 0.0f, cheby_.alpha0, target(0), stream);
        }
        for (int s = first_step; s < degree_; ++s) {
            float* out = target(s);
            launch_cheby_step<K>(lvl.n, lvl.tpr, lvl.row_ptr.get(), lvl.cols.get(),
                                 lvl.values.get(), lvl.row_scale.get(), lvl.dinv.get(), bi, cur,
                                 prev, cheby_.c_cur[s - 1], cheby_.c_prev[s - 1],
                                 cheby_.alpha[s - 1], out, stream);
            prev = cur;
            cur = out;
        }
        return cur;
    };

    const int n_levels = static_cast<int>(levels_.size());
    // The pre-sweep has to land in x, because the up-sweep prolongates the coarse correction
    // onto it; a degree-d sweep alternates d times, so that fixes which buffer it starts in.
    // r then stays free for the residual restriction reads.
    const bool odd = (degree_ % 2) != 0;
    for (int i = 0; i < n_levels; ++i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? b : level_b(lvl);
        smooth(lvl, bi, nullptr, odd ? level_x(lvl) : level_r(lvl),
               odd ? level_r(lvl) : level_x(lvl), nullptr);
        launch_residual<K>(lvl.n, lvl.tpr, lvl.row_ptr.get(), lvl.cols.get(),
                           lvl.values.get(), lvl.row_scale.get(), level_x(lvl), bi, level_r(lvl),
                           stream);
        float* next_b = (i + 1 < n_levels) ? level_b(levels_[i + 1]) : coarse_b;
        launch_restrict<K>(lvl.n_coarse, lvl.r_row_ptr.get(), lvl.r_col_idx.get(),
                           lvl.r_values.get(), level_r(lvl), next_b, stream);
    }

    launch_coarse_gemv<K>(coarse_n_, coarse_ainv_.get(), coarse_b, coarse_x, stream);

    // xc is the coarser level's smoothed correction, wherever its sweep left it.
    const float* xc = coarse_x;
    for (int i = n_levels - 1; i >= 0; --i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? b : level_b(lvl);
        launch_prolongate<K>(lvl.n, lvl.p_row_ptr.get(), lvl.p_col_idx.get(),
                             lvl.p_values.get(), xc, level_x(lvl), stream);
        xc = smooth(lvl, bi, level_x(lvl), level_r(lvl), level_x(lvl), (i == 0) ? x : nullptr);
    }
}

void NativeVCycle::apply(int n, const float* b, float* x, cudaStream_t stream) {
    check_ready(n, "apply");
    run_cycle<1>(b, x, stream);
}

void NativeVCycle::apply_block(int n, int k, const float* B, float* X, cudaStream_t stream) {
    check_ready(n, "apply_block");
    // Dispatch before allocating so an unsupported k costs nothing.
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if constexpr (K > 1) ensure_block_buffers(K);
        run_cycle<K>(B, X, stream);
    });
}
