#pragma once

#include <cstdint>
#include <vector>

#include <cublas_v2.h>

#include "common.hpp"
#include "vcycle.hpp"

struct PcgResult {
    int iterations = 0;
    double relative_residual = 0.0;
};

struct PcgBlockResult {
    int iterations = 0;                     // lockstep iterations run
    std::vector<double> relative_residual;  // per-column final relative residual
};

class PcgAmgSolver {
public:
    PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx, const double* values);
    ~PcgAmgSolver();

    PcgAmgSolver(const PcgAmgSolver&) = delete;
    PcgAmgSolver& operator=(const PcgAmgSolver&) = delete;

    void update_values(const double* values, cudaStream_t stream);
    PcgResult solve_mixed(NativeVCycle& preconditioner, const double* b, double* x,
                          double tolerance, int max_iters, cudaStream_t stream,
                          const double* x0 = nullptr);
    // Block solve: k independent CG chains in lockstep over row-major (n, k) operands,
    // sharing every stiffness-matrix read (block SpMV + block V-cycle). Stops when the
    // worst column reaches tolerance; per-column residuals are reported so callers can
    // fall back per column. k in {2, 4, 8}.
    PcgBlockResult solve_mixed_block(NativeVCycle& preconditioner, const double* B, double* X,
                                     int k, double tolerance, int max_iters,
                                     cudaStream_t stream, const double* X0 = nullptr);

private:
    void ensure_block_buffers(int k);
    int n_ = 0;
    int nnz_ = 0;
    int* row_ptr_ = nullptr;
    int* col_idx_ = nullptr;
    double* values_ = nullptr;
    double* r_ = nullptr;
    double* p_ = nullptr;
    double* ap_ = nullptr;
    double* x_int_ = nullptr;
    float* rf_ = nullptr;
    float* zf_ = nullptr;
    // CG scalars kept on-device (device-pointer-mode cuBLAS): [rz, pap, alpha, neg_alpha, norm,
    // beta]. Only the residual norm is copied back, into pinned host memory, once/iter.
    double* scalars_ = nullptr;
    // Per-block partial sums for the fused deterministic reductions (‖r‖², r·z).
    double* partials_ = nullptr;
    double* h_norm_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    // solve_mixed runs on this internal, capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream; b/x are handed off via join_event_. The iteration body
    // only touches solver-owned buffers (x_int_, not the caller's x), so the captured graph is
    // reused across solves as long as the preconditioner identity/generation is unchanged.
    cudaStream_t solve_stream_ = nullptr;
    cudaEvent_t join_event_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    const NativeVCycle* captured_precond_ = nullptr;
    int captured_precond_gen_ = 0;
    // Block-solve state: (n, k) row-major work buffers (lazily sized to the largest k
    // seen) and a separate cached graph keyed additionally on k.
    int block_k_ = 0;
    double* R_blk_ = nullptr;
    double* P_blk_ = nullptr;
    double* AP_blk_ = nullptr;
    double* X_int_blk_ = nullptr;
    float* RF_blk_ = nullptr;
    float* ZF_blk_ = nullptr;
    // Layout: [rz | pap | alpha | neg_alpha | norm | beta], each k wide.
    double* scalars_blk_ = nullptr;
    double* partials_blk_ = nullptr;
    double* h_norms_blk_ = nullptr;  // pinned, k residual norms + k reference norms
    cudaGraph_t block_graph_ = nullptr;
    cudaGraphExec_t block_graph_exec_ = nullptr;
    const NativeVCycle* block_captured_precond_ = nullptr;
    int block_captured_gen_ = 0;
    int block_captured_k_ = 0;
};

void launch_double_to_float(const double* in, float* out, int n, cudaStream_t stream);
void launch_float_to_double(const float* in, double* out, int n, cudaStream_t stream);
void launch_cg_alpha(const double* rz, const double* pap, double* alpha, double* neg_alpha,
                     cudaStream_t stream);
void launch_cg_update_p(const double* beta, const float* zf, double* p, int n,
                        cudaStream_t stream);
void launch_csrmv_f64(int n, const int* row_ptr, const int* col_idx, const double* vals,
                      const double* x, double* y, cudaStream_t stream);
int cg_partials_size(int n);
void launch_cg_update_xr_norm(const double* alpha, const double* neg_alpha, const double* p,
                              const double* ap, double* x, double* r, float* rf,
                              double* partials, double* norm_sq, int n, cudaStream_t stream);
void launch_cg_cast_dot_beta(const float* zf, const double* r, double* partials,
                             double* rz, double* beta, int n, cudaStream_t stream);

// Block CG launchers (block_cg.cu); all dense operands row-major (n, k), k in {2, 4, 8}.
int bcg_partials_blocks(int n);
void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const int* col_idx,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream);
void launch_bcg_dot(int n, int k, const double* x, const double* y, double* partials,
                    double* out, cudaStream_t stream);
void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream);
void launch_bcg_alpha(int k, const double* rz, const double* pap, double* alpha,
                      double* neg_alpha, cudaStream_t stream);
void launch_bcg_update_xr_norm(int n, int k, const double* alpha, const double* neg_alpha,
                               const double* p, const double* ap, double* x, double* r,
                               float* rf, double* partials, double* norms,
                               cudaStream_t stream);
void launch_bcg_cast_dot_beta(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, double* beta,
                              cudaStream_t stream);
void launch_bcg_cast_dot_init(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, cudaStream_t stream);
void launch_bcg_update_p(int n, int k, const double* beta, const float* zf, double* p,
                         cudaStream_t stream);
void launch_bcg_d2f(std::int64_t n_total, const double* in, float* out, cudaStream_t stream);
void launch_bcg_f2d(std::int64_t n_total, const float* in, double* out, cudaStream_t stream);
void launch_bcg_residual(std::int64_t n_total, const double* b, const double* ap, double* r,
                         cudaStream_t stream);
