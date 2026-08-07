#pragma once
#include "common.hpp"

#include <cuda_fp16.h>

#include <utility>
#include <vector>

// One zero-initial-guess aggregation-AMG V-cycle: Chebyshev smoothing over the l1-Jacobi
// diagonal, smoothed-aggregation transfers, dense coarse solve. The hierarchy operators are
// built outside in build_native_vcycle (aggregation, Galerkin products, l1-Jacobi diagonals,
// prolongators, dense coarse inverse); this class owns device copies of everything it touches
// so a captured CUDA graph stays valid for the object's lifetime.
//
// Per level l (finest = 0, all but the coarsest): A_l as fp16 CSR with a per-row fp32 scale,
// dinv_l = 1 / guard(d_l) with the l1 diagonal d (the relaxation is in the Chebyshev
// coefficients, not here), and the smoothed prolongator P_l with its transpose R_l, both fp32
// CSR. The coarsest level is only the precomputed dense inverse.
//
// The operator is stored as a_ij / max_j|a_ij| in fp16, and the row total is scaled back up after
// the reduction. The row scale is not optional: the stiffness spans more dynamic range than fp16
// covers, and unscaled fp16 costs iterations on some meshes. Callers hand over fp32; add_level
// converts.
//
// apply() is stream-ordered and CUDA-graph-capturable (no allocations, no host syncs).
// generation() keys PcgAmgSolver's cached CG graph, which embeds pointers into the buffers
// owned here.

// Longest Chebyshev recurrence the per-level coefficient arrays hold. Each degree adds one
// SpMV-shaped kernel per level per sweep.
inline constexpr int kMaxSmootherDegree = 8;

class NativeVCycle {
public:
    NativeVCycle();

    NativeVCycle(const NativeVCycle&) = delete;
    NativeVCycle& operator=(const NativeVCycle&) = delete;

    // Chebyshev shape. Must be called before the first add_level.
    void set_smoother(int degree, float lower_ratio);
    // All pointers are device memory; contents are copied into solver-owned buffers.
    // P is (n_rows, n_coarse) CSR and R is P^T, so both hold p_nnz entries.
    void add_level(int n_rows, int nnz, int n_coarse, int p_nnz, const int* row_ptr,
                   const int* col_idx, const float* values, const float* dinv,
                   const int* p_row_ptr, const int* p_col_idx, const float* p_values,
                   const int* r_row_ptr, const int* r_col_idx, const float* r_values);
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
    using Buffer = DeviceBuffer<T>;

    // Host-side Chebyshev coefficients. Step 0 is alpha0 * D^-1 b; step s > 0 is
    //     x_out = c_cur[s-1] x_cur + c_prev[s-1] x_prev + alpha[s-1] D^-1 (b - A x_cur).
    // They depend only on the smoother interval, which the l1 diagonal fixes to
    // [1 / lower_ratio, 1] on every level, so one set serves the whole hierarchy.
    struct Cheby {
        float alpha0 = 0.0f;
        float c_cur[kMaxSmootherDegree] = {};
        float c_prev[kMaxSmootherDegree] = {};
        float alpha[kMaxSmootherDegree] = {};
    };

    struct Level {
        int n = 0;
        int n_coarse = 0;
        // Threads cooperating on one row in this level's SpMV-shaped kernels, from the operator's
        // own shape (see choose_tpr). Never a function of k, so a placement's result does not
        // depend on the block width it was solved at.
        int tpr = 0;
        Buffer<int> row_ptr;
        Buffer<int> col_idx;
        Buffer<__half> values;    // a_ij / row_scale[i]
        Buffer<float> row_scale;  // max_j |a_ij|, or 1 for an empty row
        Buffer<float> dinv;
        // P (n, n_coarse) and R = P^T (n_coarse, n), both CSR with indices sorted at setup so
        // the transfer kernels' row sums have a fixed order.
        Buffer<int> p_row_ptr;
        Buffer<int> p_col_idx;
        Buffer<float> p_values;
        Buffer<int> r_row_ptr;
        Buffer<int> r_col_idx;
        Buffer<float> r_values;
        Buffer<float> b;  // level RHS (restricted residual); null at level 0, which reads
                          // the caller's b directly
        // The smoother's two ping-pong buffers, also holding the residual between sweeps.
        // Every Chebyshev step gathers x_cur across the whole vector, so it cannot write the
        // buffer it reads and a sweep alternates between these two; two suffice because the
        // step kernel's x_prev read is per-row, so its output may alias x_prev.
        Buffer<float> x;
        Buffer<float> r;
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
    Cheby cheby_;
    int degree_ = 1;
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
