#include "vcycle.hpp"

#include <atomic>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int kBlock = 256;
// A_0 has ~14 nnz/row (P1 tets); Galerkin coarse levels stay in the same regime, so the
// warp-fraction row split of the production outer SpMV is reused. The fixed shuffle
// order keeps every apply run-to-run deterministic.
constexpr int kTpr = 8;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("vcycle CUDA error (") + what +
                                 "): " + cudaGetErrorString(err));
    }
}

template <typename T>
T* device_copy(const T* src, size_t count, const char* what) {
    T* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, count * sizeof(T)), what);
    check_cuda(cudaMemcpy(ptr, src, count * sizeof(T), cudaMemcpyDeviceToDevice), what);
    return ptr;
}

__global__ void vc_jacobi_zero_kernel(int n, const float* __restrict__ dinv,
                                      const float* __restrict__ b, float* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = dinv[i] * b[i];
}

__global__ void vc_residual_kernel(int n, const int* __restrict__ row_ptr,
                                   const int* __restrict__ col_idx,
                                   const float* __restrict__ vals,
                                   const float* __restrict__ x, const float* __restrict__ b,
                                   float* __restrict__ r) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = threadIdx.x % kTpr;
    float sum = 0.0f;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int c = row_ptr[row] + lane; c < row_e; c += kTpr) {
            sum += vals[c] * __ldg(x + col_idx[c]);
        }
    }
#pragma unroll
    for (int off = kTpr / 2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off, kTpr);
    }
    if (row < n && lane == 0) r[row] = b[row] - sum;
}

// One thread per coarse row; the restriction row lists fine indices in the fork's
// stable-sorted order, so the sequential sum is deterministic.
__global__ void vc_restrict_kernel(int n_coarse, const int* __restrict__ r_row_ptr,
                                   const int* __restrict__ r_col_idx,
                                   const float* __restrict__ r_fine,
                                   float* __restrict__ b_coarse) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_coarse) return;
    const int row_e = r_row_ptr[row + 1];
    float sum = 0.0f;
    for (int c = r_row_ptr[row]; c < row_e; ++c) {
        sum += r_fine[r_col_idx[c]];
    }
    b_coarse[row] = sum;
}

__global__ void vc_prolongate_kernel(int n, const int* __restrict__ aggregates,
                                     const float* __restrict__ x_coarse,
                                     float* __restrict__ x_fine) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x_fine[i] += __ldg(x_coarse + aggregates[i]);
}

// Fused post-sweep: x_out = x_in + dinv * (b - A x_in). Out-of-place because rows gather
// x_in across the whole vector (same shape as the fork's jacobi_l1_fused_postsmooth).
__global__ void vc_postsweep_kernel(int n, const int* __restrict__ row_ptr,
                                    const int* __restrict__ col_idx,
                                    const float* __restrict__ vals,
                                    const float* __restrict__ dinv,
                                    const float* __restrict__ b,
                                    const float* __restrict__ x_in,
                                    float* __restrict__ x_out) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = threadIdx.x % kTpr;
    float sum = 0.0f;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int c = row_ptr[row] + lane; c < row_e; c += kTpr) {
            sum += vals[c] * __ldg(x_in + col_idx[c]);
        }
    }
#pragma unroll
    for (int off = kTpr / 2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off, kTpr);
    }
    if (row < n && lane == 0) {
        x_out[row] = x_in[row] + dinv[row] * (b[row] - sum);
    }
}

// One thread per row, sequential dot over the dense inverse row: deterministic and the
// coarsest system is tiny (min_coarse_rows=32 scale), so efficiency is irrelevant.
__global__ void vc_coarse_gemv_kernel(int n, const float* __restrict__ ainv,
                                      const float* __restrict__ b, float* __restrict__ x) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const float* arow = ainv + static_cast<size_t>(row) * n;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        sum += arow[j] * b[j];
    }
    x[row] = sum;
}

std::atomic<int> g_vcycle_generation{0};

template <int K>
__host__ __device__ constexpr int block_tpr() {
    // fp32 block SpMV: probe_block_spmv measured tpr 8 best through k=4, 4 at k=8.
    return K >= 8 ? 4 : 8;
}

__global__ void vc_jacobi_zero_block_kernel(long n_total, int k,
                                            const float* __restrict__ dinv,
                                            const float* __restrict__ b,
                                            float* __restrict__ x) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) x[i] = dinv[i / k] * b[i];
}

template <int K>
__global__ void vc_residual_block_kernel(int n, const int* __restrict__ row_ptr,
                                         const int* __restrict__ col_idx,
                                         const float* __restrict__ vals,
                                         const float* __restrict__ x,
                                         const float* __restrict__ b, float* __restrict__ r) {
    constexpr int kTprB = block_tpr<K>();
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTprB;
    const int lane = threadIdx.x % kTprB;
    float sum[K];
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += kTprB) {
            const float v = vals[j];
            const long base = static_cast<long>(col_idx[j]) * K;
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * __ldg(x + base + c);
        }
    }
#pragma unroll
    for (int off = kTprB / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += __shfl_down_sync(0xffffffffu, sum[c], off, kTprB);
    }
    if (row < n && lane == 0) {
        const long base = static_cast<long>(row) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) r[base + c] = b[base + c] - sum[c];
    }
}

template <int K>
__global__ void vc_restrict_block_kernel(int n_coarse, const int* __restrict__ r_row_ptr,
                                         const int* __restrict__ r_col_idx,
                                         const float* __restrict__ r_fine,
                                         float* __restrict__ b_coarse) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_coarse) return;
    const int row_e = r_row_ptr[row + 1];
    float sum[K];
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    for (int j = r_row_ptr[row]; j < row_e; ++j) {
        const long base = static_cast<long>(r_col_idx[j]) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += r_fine[base + c];
    }
    const long out = static_cast<long>(row) * K;
#pragma unroll
    for (int c = 0; c < K; ++c) b_coarse[out + c] = sum[c];
}

__global__ void vc_prolongate_block_kernel(long n_total, int k,
                                           const int* __restrict__ aggregates,
                                           const float* __restrict__ x_coarse,
                                           float* __restrict__ x_fine) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_total) {
        x_fine[i] += __ldg(x_coarse + static_cast<long>(aggregates[i / k]) * k + (i % k));
    }
}

template <int K>
__global__ void vc_postsweep_block_kernel(int n, const int* __restrict__ row_ptr,
                                          const int* __restrict__ col_idx,
                                          const float* __restrict__ vals,
                                          const float* __restrict__ dinv,
                                          const float* __restrict__ b,
                                          const float* __restrict__ x_in,
                                          float* __restrict__ x_out) {
    constexpr int kTprB = block_tpr<K>();
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTprB;
    const int lane = threadIdx.x % kTprB;
    float sum[K];
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    if (row < n) {
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += kTprB) {
            const float v = vals[j];
            const long base = static_cast<long>(col_idx[j]) * K;
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * __ldg(x_in + base + c);
        }
    }
#pragma unroll
    for (int off = kTprB / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += __shfl_down_sync(0xffffffffu, sum[c], off, kTprB);
    }
    if (row < n && lane == 0) {
        const float d = dinv[row];
        const long base = static_cast<long>(row) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            x_out[base + c] = x_in[base + c] + d * (b[base + c] - sum[c]);
        }
    }
}

template <int K>
__global__ void vc_coarse_gemv_block_kernel(int n, const float* __restrict__ ainv,
                                            const float* __restrict__ b,
                                            float* __restrict__ x) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const float* arow = ainv + static_cast<size_t>(row) * n;
    float sum[K];
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0f;
    for (int j = 0; j < n; ++j) {
        const float a = arow[j];
        const long base = static_cast<long>(j) * K;
#pragma unroll
        for (int c = 0; c < K; ++c) sum[c] += a * b[base + c];
    }
    const long out = static_cast<long>(row) * K;
#pragma unroll
    for (int c = 0; c < K; ++c) x[out + c] = sum[c];
}

template <typename F>
void dispatch_block_k(int k, F&& f) {
    switch (k) {
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: throw std::invalid_argument("V-cycle block apply supports k in {2, 4, 8}");
    }
}

}  // namespace

void launch_vc_jacobi_zero(int n, const float* dinv, const float* b, float* x,
                           cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    vc_jacobi_zero_kernel<<<blocks, kBlock, 0, stream>>>(n, dinv, b, x);
}

void launch_vc_residual(int n, const int* row_ptr, const int* col_idx, const float* values,
                        const float* x, const float* b, float* r, cudaStream_t stream) {
    const int blocks = (n * kTpr + kBlock - 1) / kBlock;
    vc_residual_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, values, x, b, r);
}

void launch_vc_restrict(int n_coarse, const int* r_row_ptr, const int* r_col_idx,
                        const float* r_fine, float* b_coarse, cudaStream_t stream) {
    const int blocks = (n_coarse + kBlock - 1) / kBlock;
    vc_restrict_kernel<<<blocks, kBlock, 0, stream>>>(n_coarse, r_row_ptr, r_col_idx, r_fine,
                                                      b_coarse);
}

void launch_vc_prolongate(int n, const int* aggregates, const float* x_coarse, float* x_fine,
                          cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    vc_prolongate_kernel<<<blocks, kBlock, 0, stream>>>(n, aggregates, x_coarse, x_fine);
}

void launch_vc_postsweep(int n, const int* row_ptr, const int* col_idx, const float* values,
                         const float* dinv, const float* b, const float* x_in, float* x_out,
                         cudaStream_t stream) {
    const int blocks = (n * kTpr + kBlock - 1) / kBlock;
    vc_postsweep_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, values, dinv, b,
                                                       x_in, x_out);
}

void launch_vc_coarse_gemv(int n, const float* ainv, const float* b, float* x,
                           cudaStream_t stream) {
    const int blocks = (n + kBlock - 1) / kBlock;
    vc_coarse_gemv_kernel<<<blocks, kBlock, 0, stream>>>(n, ainv, b, x);
}

void launch_vc_jacobi_zero_block(int n, int k, const float* dinv, const float* b, float* x,
                                 cudaStream_t stream) {
    const long total = static_cast<long>(n) * k;
    const long blocks = (total + kBlock - 1) / kBlock;
    vc_jacobi_zero_block_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
        total, k, dinv, b, x);
}

void launch_vc_residual_block(int n, int k, const int* row_ptr, const int* col_idx,
                              const float* values, const float* x, const float* b, float* r,
                              cudaStream_t stream) {
    dispatch_block_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        const int blocks = (n * block_tpr<K>() + kBlock - 1) / kBlock;
        vc_residual_block_kernel<K>
            <<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, values, x, b, r);
    });
}

void launch_vc_restrict_block(int n_coarse, int k, const int* r_row_ptr, const int* r_col_idx,
                              const float* r_fine, float* b_coarse, cudaStream_t stream) {
    dispatch_block_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        const int blocks = (n_coarse + kBlock - 1) / kBlock;
        vc_restrict_block_kernel<K>
            <<<blocks, kBlock, 0, stream>>>(n_coarse, r_row_ptr, r_col_idx, r_fine, b_coarse);
    });
}

void launch_vc_prolongate_block(int n, int k, const int* aggregates, const float* x_coarse,
                                float* x_fine, cudaStream_t stream) {
    const long total = static_cast<long>(n) * k;
    const long blocks = (total + kBlock - 1) / kBlock;
    vc_prolongate_block_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
        total, k, aggregates, x_coarse, x_fine);
}

void launch_vc_postsweep_block(int n, int k, const int* row_ptr, const int* col_idx,
                               const float* values, const float* dinv, const float* b,
                               const float* x_in, float* x_out, cudaStream_t stream) {
    dispatch_block_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        const int blocks = (n * block_tpr<K>() + kBlock - 1) / kBlock;
        vc_postsweep_block_kernel<K>
            <<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, values, dinv, b, x_in, x_out);
    });
}

void launch_vc_coarse_gemv_block(int n, int k, const float* ainv, const float* b, float* x,
                                 cudaStream_t stream) {
    dispatch_block_k(k, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        const int blocks = (n + kBlock - 1) / kBlock;
        vc_coarse_gemv_block_kernel<K><<<blocks, kBlock, 0, stream>>>(n, ainv, b, x);
    });
}

NativeVCycle::NativeVCycle() : generation_(++g_vcycle_generation) {}

NativeVCycle::~NativeVCycle() {
    for (Level& lvl : levels_) {
        cudaFree(lvl.row_ptr);
        cudaFree(lvl.col_idx);
        cudaFree(lvl.values);
        cudaFree(lvl.dinv);
        cudaFree(lvl.r_row_ptr);
        cudaFree(lvl.r_col_idx);
        cudaFree(lvl.aggregates);
        cudaFree(lvl.b);
        cudaFree(lvl.x);
        cudaFree(lvl.r);
        cudaFree(lvl.bk);
        cudaFree(lvl.xk);
        cudaFree(lvl.rk);
    }
    cudaFree(coarse_ainv_);
    cudaFree(coarse_b_);
    cudaFree(coarse_x_);
    cudaFree(coarse_bk_);
    cudaFree(coarse_xk_);
}

void NativeVCycle::ensure_block_buffers(int k) {
    if (block_k_ >= k) return;
    for (Level& lvl : levels_) {
        cudaFree(lvl.bk);
        cudaFree(lvl.xk);
        cudaFree(lvl.rk);
        const size_t bytes = static_cast<size_t>(lvl.n) * k * sizeof(float);
        check_cuda(cudaMalloc(&lvl.bk, bytes), "level bk");
        check_cuda(cudaMalloc(&lvl.xk, bytes), "level xk");
        check_cuda(cudaMalloc(&lvl.rk, bytes), "level rk");
    }
    cudaFree(coarse_bk_);
    cudaFree(coarse_xk_);
    const size_t cbytes = static_cast<size_t>(coarse_n_) * k * sizeof(float);
    check_cuda(cudaMalloc(&coarse_bk_, cbytes), "coarse bk");
    check_cuda(cudaMalloc(&coarse_xk_, cbytes), "coarse xk");
    block_k_ = k;
}

void NativeVCycle::add_level(int n_rows, int nnz, int n_coarse, const int* row_ptr,
                             const int* col_idx, const float* values, const float* dinv,
                             const int* r_row_ptr, const int* r_col_idx,
                             const int* aggregates) {
    if (finalized_) throw std::runtime_error("NativeVCycle: add_level after finalize");
    Level lvl;
    lvl.n = n_rows;
    lvl.nnz = nnz;
    lvl.n_coarse = n_coarse;
    lvl.row_ptr = device_copy(row_ptr, static_cast<size_t>(n_rows) + 1, "level row_ptr");
    lvl.col_idx = device_copy(col_idx, static_cast<size_t>(nnz), "level col_idx");
    lvl.values = device_copy(values, static_cast<size_t>(nnz), "level values");
    lvl.dinv = device_copy(dinv, static_cast<size_t>(n_rows), "level dinv");
    lvl.r_row_ptr = device_copy(r_row_ptr, static_cast<size_t>(n_coarse) + 1, "level R ptr");
    lvl.r_col_idx = device_copy(r_col_idx, static_cast<size_t>(n_rows), "level R idx");
    lvl.aggregates = device_copy(aggregates, static_cast<size_t>(n_rows), "level agg");
    check_cuda(cudaMalloc(&lvl.b, static_cast<size_t>(n_rows) * sizeof(float)), "level b");
    check_cuda(cudaMalloc(&lvl.x, static_cast<size_t>(n_rows) * sizeof(float)), "level x");
    check_cuda(cudaMalloc(&lvl.r, static_cast<size_t>(n_rows) * sizeof(float)), "level r");
    levels_.push_back(lvl);
}

void NativeVCycle::set_coarse(int n, const float* ainv) {
    if (finalized_) throw std::runtime_error("NativeVCycle: set_coarse after finalize");
    coarse_n_ = n;
    coarse_ainv_ = device_copy(ainv, static_cast<size_t>(n) * n, "coarse ainv");
    check_cuda(cudaMalloc(&coarse_b_, static_cast<size_t>(n) * sizeof(float)), "coarse b");
    check_cuda(cudaMalloc(&coarse_x_, static_cast<size_t>(n) * sizeof(float)), "coarse x");
}

void NativeVCycle::finalize() {
    if (coarse_ainv_ == nullptr) {
        throw std::runtime_error("NativeVCycle: finalize without set_coarse");
    }
    for (size_t i = 0; i < levels_.size(); ++i) {
        const int expected = (i + 1 < levels_.size()) ? levels_[i + 1].n : coarse_n_;
        if (levels_[i].n_coarse != expected) {
            throw std::runtime_error("NativeVCycle: inconsistent level dimensions");
        }
    }
    finalized_ = true;
}

void NativeVCycle::apply(int n, const float* b, float* x, cudaStream_t stream) {
    if (!finalized_) throw std::runtime_error("NativeVCycle: apply before finalize");
    if (levels_.empty()) {
        if (n != coarse_n_) throw std::runtime_error("NativeVCycle: size mismatch");
        launch_vc_coarse_gemv(coarse_n_, coarse_ainv_, b, x, stream);
        return;
    }
    if (n != levels_[0].n) throw std::runtime_error("NativeVCycle: size mismatch");

    const int n_levels = static_cast<int>(levels_.size());
    for (int i = 0; i < n_levels; ++i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? b : lvl.b;
        launch_vc_jacobi_zero(lvl.n, lvl.dinv, bi, lvl.x, stream);
        launch_vc_residual(lvl.n, lvl.row_ptr, lvl.col_idx, lvl.values, lvl.x, bi, lvl.r,
                           stream);
        float* next_b = (i + 1 < n_levels) ? levels_[i + 1].b : coarse_b_;
        launch_vc_restrict(lvl.n_coarse, lvl.r_row_ptr, lvl.r_col_idx, lvl.r, next_b, stream);
    }

    launch_vc_coarse_gemv(coarse_n_, coarse_ainv_, coarse_b_, coarse_x_, stream);

    for (int i = n_levels - 1; i >= 0; --i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? b : lvl.b;
        // Below the top of the up-sweep, a level's final smoothed x was written into its
        // r buffer by the out-of-place post-sweep.
        const float* xc = (i + 1 < n_levels) ? levels_[i + 1].r : coarse_x_;
        launch_vc_prolongate(lvl.n, lvl.aggregates, xc, lvl.x, stream);
        float* out = (i == 0) ? x : lvl.r;
        launch_vc_postsweep(lvl.n, lvl.row_ptr, lvl.col_idx, lvl.values, lvl.dinv, bi, lvl.x,
                            out, stream);
    }
}

// Same cycle as apply(), k columns in lockstep over (n, k) row-major operands. The
// first call for a given k allocates the block work vectors (callers warm up before
// CUDA graph capture, matching the single-RHS pool-warmup convention).
void NativeVCycle::apply_block(int n, int k, const float* B, float* X, cudaStream_t stream) {
    if (!finalized_) throw std::runtime_error("NativeVCycle: apply_block before finalize");
    ensure_block_buffers(k);
    if (levels_.empty()) {
        if (n != coarse_n_) throw std::runtime_error("NativeVCycle: size mismatch");
        launch_vc_coarse_gemv_block(coarse_n_, k, coarse_ainv_, B, X, stream);
        return;
    }
    if (n != levels_[0].n) throw std::runtime_error("NativeVCycle: size mismatch");

    const int n_levels = static_cast<int>(levels_.size());
    for (int i = 0; i < n_levels; ++i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? B : lvl.bk;
        launch_vc_jacobi_zero_block(lvl.n, k, lvl.dinv, bi, lvl.xk, stream);
        launch_vc_residual_block(lvl.n, k, lvl.row_ptr, lvl.col_idx, lvl.values, lvl.xk, bi,
                                 lvl.rk, stream);
        float* next_b = (i + 1 < n_levels) ? levels_[i + 1].bk : coarse_bk_;
        launch_vc_restrict_block(lvl.n_coarse, k, lvl.r_row_ptr, lvl.r_col_idx, lvl.rk, next_b,
                                 stream);
    }

    launch_vc_coarse_gemv_block(coarse_n_, k, coarse_ainv_, coarse_bk_, coarse_xk_, stream);

    for (int i = n_levels - 1; i >= 0; --i) {
        Level& lvl = levels_[i];
        const float* bi = (i == 0) ? B : lvl.bk;
        const float* xc = (i + 1 < n_levels) ? levels_[i + 1].rk : coarse_xk_;
        launch_vc_prolongate_block(lvl.n, k, lvl.aggregates, xc, lvl.xk, stream);
        float* out = (i == 0) ? X : lvl.rk;
        launch_vc_postsweep_block(lvl.n, k, lvl.row_ptr, lvl.col_idx, lvl.values, lvl.dinv, bi,
                                  lvl.xk, out, stream);
    }
}
