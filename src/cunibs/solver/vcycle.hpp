#pragma once
#include "common.hpp"

#include <utility>
#include <vector>

namespace vcycle_detail {

// Move-only owner of one cudaMalloc'd block. Level setup takes a dozen allocations in a
// row, so wrapping them keeps a failure part way through from leaking the earlier ones.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(T* ptr) noexcept : ptr_(ptr) {}
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.release()) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) reset(other.release());
        return *this;
    }

    T* get() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    T* release() noexcept { return std::exchange(ptr_, nullptr); }

    void reset(T* ptr = nullptr) noexcept {
        // A free during interpreter teardown, after the context is gone, reports an error
        // there is nothing useful to do with.
        if (ptr_ != nullptr) cudaFree(ptr_);
        ptr_ = ptr;
    }

private:
    T* ptr_ = nullptr;
};

}  // namespace vcycle_detail

// One zero-initial-guess aggregation-AMG V-cycle: l1-Jacobi smoothing, unsmoothed
// (piecewise-constant) transfers, dense coarse solve. The hierarchy operators are built
// outside in build_native_vcycle (aggregation, Galerkin products, l1-Jacobi diagonals,
// restriction order, dense coarse inverse); this class owns device copies of everything
// it touches so a captured CUDA graph stays valid for the object's lifetime.
//
// Per level l (finest = 0, all but the coarsest): A_l as fp32 CSR, dinv_l = omega /
// guard(d_l) with the l1 diagonal d and the smoother relaxation factor already folded
// in, the restriction CSR (coarse row -> fine indices in stable-sorted order), and the
// fine->aggregate map for prolongation.
// The coarsest level is only the precomputed dense inverse.
//
// apply() is stream-ordered and CUDA-graph-capturable (no allocations, no host syncs).
// generation() keys PcgAmgSolver's cached CG graph, which embeds pointers into the buffers
// owned here.
class NativeVCycle {
public:
    NativeVCycle();

    NativeVCycle(const NativeVCycle&) = delete;
    NativeVCycle& operator=(const NativeVCycle&) = delete;

    // All pointers are device memory; contents are copied into solver-owned buffers.
    // Each fine row belongs to exactly one aggregate, so r_col_idx holds n_rows entries.
    void add_level(int n_rows, int nnz, int n_coarse, const int* row_ptr, const int* col_idx,
                   const float* values, const float* dinv, const int* r_row_ptr,
                   const int* r_col_idx, const int* aggregates);
    // ainv is the dense inverse of the coarsest-level matrix, row-major (n x n).
    void set_coarse(int n, const float* ainv);
    void finalize();

    void apply(int n, const float* b, float* x, cudaStream_t stream);
    // Block variant: B and X are row-major (n, k), k in {2, 4, 8}. Must be graph-capturable
    // after a first (warm-up) call.
    void apply_block(int n, int k, const float* B, float* X, cudaStream_t stream);
    int generation() const { return generation_; }
    // Coarsening levels, excluding the coarsest (dense-inverse) one.
    int n_levels() const { return static_cast<int>(levels_.size()); }

private:
    template <typename T>
    using Buffer = vcycle_detail::DeviceBuffer<T>;

    struct Level {
        int n = 0;
        int n_coarse = 0;
        Buffer<int> row_ptr;
        Buffer<int> col_idx;
        Buffer<float> values;
        Buffer<float> dinv;
        Buffer<int> r_row_ptr;
        Buffer<int> r_col_idx;
        Buffer<int> aggregates;
        Buffer<float> b;  // level RHS (restricted residual); null at level 0, which reads
                          // the caller's b directly
        Buffer<float> x;  // pre-smoothed iterate, then corrected in place
        Buffer<float> r;  // residual on the way down; the Jacobi post-sweep is
                          // out-of-place, so on the way up this holds the level's
                          // final smoothed x (level 0 writes the caller's x instead)
        // (n, k) row-major work buffers for apply_block, sized for block_k_.
        Buffer<float> bk;
        Buffer<float> xk;
        Buffer<float> rk;
    };

    void ensure_block_buffers(int k);
    void check_ready(int n, const char* what) const;
    // K = 1 walks the single-RHS buffers, K > 1 the (n, K) row-major ones.
    template <int K>
    void run_cycle(const float* b, float* x, cudaStream_t stream);

    std::vector<Level> levels_;
    int coarse_n_ = 0;
    Buffer<float> coarse_ainv_;
    Buffer<float> coarse_b_;
    Buffer<float> coarse_x_;
    Buffer<float> coarse_bk_;
    Buffer<float> coarse_xk_;
    int block_k_ = 0;
    int generation_ = 0;
    bool finalized_ = false;
};
