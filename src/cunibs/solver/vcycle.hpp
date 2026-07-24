#pragma once
#include <cuda_runtime.h>

#include <vector>

// Common interface for the fp32 preconditioner apply inside the mixed-precision PCG.
// Implementations must be stream-ordered and CUDA-graph-capturable (no allocations,
// no host syncs inside apply). generation() keys the cached CG graph: any value change
// means previously captured graphs referencing this preconditioner must be recaptured.
class FloatPrecond {
public:
    virtual ~FloatPrecond() = default;
    virtual void apply(int n, const float* b, float* x, cudaStream_t stream) = 0;
    virtual int generation() const = 0;
};

// Native replication of the AMGx AGGREGATION/JACOBI_L1/V/DENSE_LU preconditioner apply
// (one zero-initial-guess V-cycle). The hierarchy operators are built outside (Galerkin
// products, l1-Jacobi diagonals, restriction order, dense coarse inverse) from the
// aggregate maps exported by the AMGx fork; this class owns device copies of everything
// it touches so a captured CUDA graph stays valid for the object's lifetime.
//
// Per level l (finest = 0, all but the coarsest): A_l as fp32 CSR, dinv_l = omega /
// guard(d_l) with the l1 diagonal d and the smoother relaxation factor already folded
// in, the restriction CSR (coarse row -> stably-ordered fine indices, matching the
// fork's computeRestrictionOperator), and the fine->aggregate map for prolongation.
// The coarsest level is only the precomputed dense inverse.
class NativeVCycle : public FloatPrecond {
public:
    NativeVCycle();
    ~NativeVCycle() override;

    NativeVCycle(const NativeVCycle&) = delete;
    NativeVCycle& operator=(const NativeVCycle&) = delete;

    // All pointers are device memory; contents are copied into solver-owned buffers.
    void add_level(int n_rows, int nnz, int n_coarse, const int* row_ptr, const int* col_idx,
                   const float* values, const float* dinv, const int* r_row_ptr,
                   const int* r_col_idx, const int* aggregates);
    // ainv is the dense inverse of the coarsest-level matrix, row-major (n x n).
    void set_coarse(int n, const float* ainv);
    void finalize();

    void apply(int n, const float* b, float* x, cudaStream_t stream) override;
    // Block variant: B and X are row-major (n, k). Must be graph-capturable after a
    // first (warm-up) call. Native-only: PcgAmgSolver::solve_mixed_block takes a
    // NativeVCycle directly, so this is not part of the FloatPrecond interface.
    void apply_block(int n, int k, const float* B, float* X, cudaStream_t stream);
    int generation() const override { return generation_; }

private:
    void ensure_block_buffers(int k);

    struct Level {
        int n = 0;
        int nnz = 0;
        int n_coarse = 0;
        int* row_ptr = nullptr;
        int* col_idx = nullptr;
        float* values = nullptr;
        float* dinv = nullptr;
        int* r_row_ptr = nullptr;
        int* r_col_idx = nullptr;
        int* aggregates = nullptr;
        float* b = nullptr;  // level RHS (restricted residual); unused at level 0
        float* x = nullptr;  // pre-smoothed iterate, then corrected in place
        float* r = nullptr;  // residual on the way down; the Jacobi post-sweep is
                             // out-of-place, so on the way up this holds the level's
                             // final smoothed x (level 0 writes the caller's x instead)
        // (n, k) row-major work buffers for apply_block, sized for block_k_.
        float* bk = nullptr;
        float* xk = nullptr;
        float* rk = nullptr;
    };

    std::vector<Level> levels_;
    int coarse_n_ = 0;
    float* coarse_ainv_ = nullptr;
    float* coarse_b_ = nullptr;
    float* coarse_x_ = nullptr;
    float* coarse_bk_ = nullptr;
    float* coarse_xk_ = nullptr;
    int block_k_ = 0;
    int generation_ = 0;
    bool finalized_ = false;
};

void launch_vc_jacobi_zero(int n, const float* dinv, const float* b, float* x,
                           cudaStream_t stream);
void launch_vc_residual(int n, const int* row_ptr, const int* col_idx, const float* values,
                        const float* x, const float* b, float* r, cudaStream_t stream);
void launch_vc_restrict(int n_coarse, const int* r_row_ptr, const int* r_col_idx,
                        const float* r_fine, float* b_coarse, cudaStream_t stream);
void launch_vc_prolongate(int n, const int* aggregates, const float* x_coarse, float* x_fine,
                          cudaStream_t stream);
void launch_vc_postsweep(int n, const int* row_ptr, const int* col_idx, const float* values,
                         const float* dinv, const float* b, const float* x_in, float* x_out,
                         cudaStream_t stream);
void launch_vc_coarse_gemv(int n, const float* ainv, const float* b, float* x,
                           cudaStream_t stream);

// Block (k-RHS, row-major (n, k)) variants; k in {2, 4, 8}.
void launch_vc_jacobi_zero_block(int n, int k, const float* dinv, const float* b, float* x,
                                 cudaStream_t stream);
void launch_vc_residual_block(int n, int k, const int* row_ptr, const int* col_idx,
                              const float* values, const float* x, const float* b, float* r,
                              cudaStream_t stream);
void launch_vc_restrict_block(int n_coarse, int k, const int* r_row_ptr, const int* r_col_idx,
                              const float* r_fine, float* b_coarse, cudaStream_t stream);
void launch_vc_prolongate_block(int n, int k, const int* aggregates, const float* x_coarse,
                                float* x_fine, cudaStream_t stream);
void launch_vc_postsweep_block(int n, int k, const int* row_ptr, const int* col_idx,
                               const float* values, const float* dinv, const float* b,
                               const float* x_in, float* x_out, cudaStream_t stream);
void launch_vc_coarse_gemv_block(int n, int k, const float* ainv, const float* b, float* x,
                                 cudaStream_t stream);
