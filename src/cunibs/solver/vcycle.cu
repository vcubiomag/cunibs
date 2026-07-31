// Every V-cycle step is written once, templated on the number of right-hand sides K, and
// the single-RHS cycle is the K = 1 instantiation: for K = 1 the row-major (n, K) operands
// collapse to plain vectors and the compiler folds the column loops away.
#include "vcycle.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
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
Buffer<T> device_clone(const T* src, std::size_t count, const char* what) {
    return ::device_clone<T>(src, count, "vcycle", what);
}

// Threads per row for the SpMV-shaped kernels. A_0 has ~14 nnz/row (P1 tets) and the
// Galerkin coarse levels only densify to ~20, so a quarter warp covers a row in a few
// strided steps and pays two shuffle levels to reduce it. Measured 6-11% faster than 8 at
// every K on the 881k-row, 6-level hierarchy of a full head mesh, clocks locked. 2 ties it
// there but loses once column locality is poor, so 4 is the safer of the two.
constexpr int kTpr = 4;

static_assert(kWarp % kTpr == 0, "shuffle width must divide the warp");
static_assert(kBlock % kTpr == 0, "a block must hold a whole number of rows");

// One warp fraction per row: strided partial products over the row's nonzeros, then a
// fixed-order shuffle reduction leaving the row total in lane 0. Threads whose row is past
// the end still reach the shuffle with zero accumulators, which is why the mask is the full
// warp. The fixed order is what keeps an apply run-to-run deterministic.
template <int K>
__device__ __forceinline__ void row_spmv(int n, const int* __restrict__ row_ptr,
                                         const int* __restrict__ col_idx,
                                         const float* __restrict__ vals,
                                         const float* __restrict__ x, int row, int lane,
                                         float (&sum)[K]) {
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += kTpr) {
            const float v = vals[j];
            const std::int64_t base = static_cast<std::int64_t>(col_idx[j]) * K;
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * __ldg(x + base + c);
        }
    }
#pragma unroll
    for (int off = kTpr / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += __shfl_down_sync(0xffffffffu, sum[c], off, kTpr);
    }
}

// n_total = n * K throughout the elementwise kernels; K is a power of two, so the row and
// column of a linear index come out as a shift and a mask on an unsigned index.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_jacobi_zero_kernel(std::uint64_t n_total, const float* __restrict__ dinv,
                          const float* __restrict__ b, float* __restrict__ x) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) x[i] = dinv[i / K] * b[i];
}

template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_residual_kernel(int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                       const float* __restrict__ vals, const float* __restrict__ x,
                       const float* __restrict__ b, float* __restrict__ r) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = static_cast<int>(threadIdx.x) % kTpr;
    float sum[K];
    row_spmv<K>(n, row_ptr, col_idx, vals, x, row, lane, sum);
    if (row < n && lane == 0) {
        const std::int64_t base = static_cast<std::int64_t>(row) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) r[base + c] = b[base + c] - sum[c];
    }
}

// One thread per coarse row; the restriction row lists its fine indices in stable-sorted
// order, so the sequential sum is deterministic.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_restrict_kernel(int n_coarse, const int* __restrict__ r_row_ptr,
                       const int* __restrict__ r_col_idx, const float* __restrict__ r_fine,
                       float* __restrict__ b_coarse) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_coarse) return;
    const int row_e = r_row_ptr[row + 1];
    float sum[K];
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    for (int j = r_row_ptr[row]; j < row_e; ++j) {
        const std::int64_t base = static_cast<std::int64_t>(r_col_idx[j]) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += r_fine[base + c];
    }
    const std::int64_t out = static_cast<std::int64_t>(row) * K;
#pragma unroll
    for (int c = 0; c < K; ++c) b_coarse[out + c] = sum[c];
}

template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_prolongate_kernel(std::uint64_t n_total, const int* __restrict__ aggregates,
                         const float* __restrict__ x_coarse, float* __restrict__ x_fine) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    const std::int64_t src =
        static_cast<std::int64_t>(aggregates[i / K]) * K + static_cast<int>(i % K);
    x_fine[i] += __ldg(x_coarse + src);
}

// Fused post-sweep: x_out = x_in + dinv * (b - A x_in). Out-of-place because rows gather
// x_in across the whole vector.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_postsweep_kernel(int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                        const float* __restrict__ vals, const float* __restrict__ dinv,
                        const float* __restrict__ b, const float* __restrict__ x_in,
                        float* __restrict__ x_out) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = static_cast<int>(threadIdx.x) % kTpr;
    float sum[K];
    row_spmv<K>(n, row_ptr, col_idx, vals, x_in, row, lane, sum);
    if (row < n && lane == 0) {
        const float d = dinv[row];
        const std::int64_t base = static_cast<std::int64_t>(row) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            x_out[base + c] = x_in[base + c] + d * (b[base + c] - sum[c]);
        }
    }
}

// One thread per row, sequential dot over the dense inverse row: deterministic, and the
// coarsest system is a few hundred rows, so efficiency is irrelevant.
template <int K>
__global__ void __launch_bounds__(kBlock)
    vc_coarse_gemv_kernel(int n, const float* __restrict__ ainv, const float* __restrict__ b,
                          float* __restrict__ x) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;
    const float* arow = ainv + static_cast<std::int64_t>(row) * n;
    float sum[K];
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    for (int j = 0; j < n; ++j) {
        const float a = arow[j];
        const std::int64_t base = static_cast<std::int64_t>(j) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += a * b[base + c];
    }
    const std::int64_t out = static_cast<std::int64_t>(row) * K;
#pragma unroll
    for (int c = 0; c < K; ++c) x[out + c] = sum[c];
}

// The launchers deliberately do not check cudaGetLastError(). A cycle runs inside the CG
// stream capture, and throwing there would skip cudaStreamEndCapture and leave the stream
// capturing for good; PcgAmgSolver already handles a failed capture, and anything a launch
// could report surfaces at its next synchronize. Grid dimensions are checked here instead,
// which is the only launch failure this code can cause on its own.
template <int K>
void launch_jacobi_zero(int n, const float* dinv, const float* b, float* x,
                        cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(n) * K;
    if (const unsigned blocks = grid_for(total)) {
        vc_jacobi_zero_kernel<K><<<blocks, kBlock, 0, stream>>>(
            static_cast<std::uint64_t>(total), dinv, b, x);
    }
}

template <int K>
void launch_residual(int n, const int* row_ptr, const int* col_idx, const float* values,
                     const float* x, const float* b, float* r, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * kTpr)) {
        vc_residual_kernel<K>
            <<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, values, x, b, r);
    }
}

template <int K>
void launch_restrict(int n_coarse, const int* r_row_ptr, const int* r_col_idx,
                     const float* r_fine, float* b_coarse, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_coarse)) {
        vc_restrict_kernel<K>
            <<<blocks, kBlock, 0, stream>>>(n_coarse, r_row_ptr, r_col_idx, r_fine, b_coarse);
    }
}

template <int K>
void launch_prolongate(int n, const int* aggregates, const float* x_coarse, float* x_fine,
                       cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(n) * K;
    if (const unsigned blocks = grid_for(total)) {
        vc_prolongate_kernel<K><<<blocks, kBlock, 0, stream>>>(
            static_cast<std::uint64_t>(total), aggregates, x_coarse, x_fine);
    }
}

template <int K>
void launch_postsweep(int n, const int* row_ptr, const int* col_idx, const float* values,
                      const float* dinv, const float* b, const float* x_in, float* x_out,
                      cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * kTpr)) {
        vc_postsweep_kernel<K><<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, values, dinv,
                                                              b, x_in, x_out);
    }
}

template <int K>
void launch_coarse_gemv(int n, const float* ainv, const float* b, float* x,
                        cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n)) {
        vc_coarse_gemv_kernel<K><<<blocks, kBlock, 0, stream>>>(n, ainv, b, x);
    }
}

constexpr const char* kBadK = "V-cycle block apply supports k in {2, 4, 8}";

std::atomic<int> g_vcycle_generation{0};

}  // namespace

NativeVCycle::NativeVCycle() : generation_(++g_vcycle_generation) {}

void NativeVCycle::add_level(int n_rows, int nnz, int n_coarse, const int* row_ptr,
                             const int* col_idx, const float* values, const float* dinv,
                             const int* r_row_ptr, const int* r_col_idx,
                             const int* aggregates) {
    if (finalized_) throw std::runtime_error("NativeVCycle: add_level after finalize");
    if (n_rows <= 0 || n_coarse <= 0 || nnz < 0) {
        throw std::invalid_argument("NativeVCycle: level dimensions must be positive");
    }

    Level lvl;
    lvl.n = n_rows;
    lvl.n_coarse = n_coarse;
    lvl.row_ptr = device_clone(row_ptr, static_cast<std::size_t>(n_rows) + 1, "level row_ptr");
    lvl.col_idx = device_clone(col_idx, static_cast<std::size_t>(nnz), "level col_idx");
    lvl.values = device_clone(values, static_cast<std::size_t>(nnz), "level values");
    lvl.dinv = device_clone(dinv, static_cast<std::size_t>(n_rows), "level dinv");
    lvl.r_row_ptr =
        device_clone(r_row_ptr, static_cast<std::size_t>(n_coarse) + 1, "level R ptr");
    lvl.r_col_idx = device_clone(r_col_idx, static_cast<std::size_t>(n_rows), "level R idx");
    lvl.aggregates = device_clone(aggregates, static_cast<std::size_t>(n_rows), "level agg");
    // The finest level reads the caller's b in place, so it needs no RHS of its own.
    if (!levels_.empty()) {
        lvl.b = device_alloc<float>(static_cast<std::size_t>(n_rows), "level b");
    }
    lvl.x = device_alloc<float>(static_cast<std::size_t>(n_rows), "level x");
    lvl.r = device_alloc<float>(static_cast<std::size_t>(n_rows), "level r");
    levels_.push_back(std::move(lvl));
}

void NativeVCycle::set_coarse(int n, const float* ainv) {
    if (finalized_) throw std::runtime_error("NativeVCycle: set_coarse after finalize");
    if (n <= 0) throw std::invalid_argument("NativeVCycle: coarse size must be positive");
    coarse_ainv_ = device_clone(ainv, static_cast<std::size_t>(n) * n, "coarse ainv");
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

    const int n_levels = static_cast<int>(levels_.size());
    for (int i = 0; i < n_levels; ++i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? b : level_b(lvl);
        launch_jacobi_zero<K>(lvl.n, lvl.dinv.get(), bi, level_x(lvl), stream);
        launch_residual<K>(lvl.n, lvl.row_ptr.get(), lvl.col_idx.get(), lvl.values.get(),
                           level_x(lvl), bi, level_r(lvl), stream);
        float* next_b = (i + 1 < n_levels) ? level_b(levels_[i + 1]) : coarse_b;
        launch_restrict<K>(lvl.n_coarse, lvl.r_row_ptr.get(), lvl.r_col_idx.get(), level_r(lvl),
                           next_b, stream);
    }

    launch_coarse_gemv<K>(coarse_n_, coarse_ainv_.get(), coarse_b, coarse_x, stream);

    for (int i = n_levels - 1; i >= 0; --i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? b : level_b(lvl);
        // Below the top of the up-sweep, a level's final smoothed x was written into its
        // r buffer by the out-of-place post-sweep.
        const float* xc = (i + 1 < n_levels) ? level_r(levels_[i + 1]) : coarse_x;
        launch_prolongate<K>(lvl.n, lvl.aggregates.get(), xc, level_x(lvl), stream);
        float* out = (i == 0) ? x : level_r(lvl);
        launch_postsweep<K>(lvl.n, lvl.row_ptr.get(), lvl.col_idx.get(), lvl.values.get(),
                            lvl.dinv.get(), bi, level_x(lvl), out, stream);
    }
}

void NativeVCycle::apply(int n, const float* b, float* x, cudaStream_t stream) {
    check_ready(n, "apply");
    run_cycle<1>(b, x, stream);
}

void NativeVCycle::apply_block(int n, int k, const float* B, float* X, cudaStream_t stream) {
    check_ready(n, "apply_block");
    // Dispatch before allocating so an unsupported k costs nothing.
    dispatch_k<2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        ensure_block_buffers(K);
        run_cycle<K>(B, X, stream);
    });
}
