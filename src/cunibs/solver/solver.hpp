#pragma once
#include <string>

#include <amgx_c.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

struct PcgResult {
    int iterations = 0;
    double relative_residual = 0.0;
};

class AMGXSolver {
public:
    explicit AMGXSolver(const std::string& config);
    ~AMGXSolver();

    AMGXSolver(const AMGXSolver&) = delete;
    AMGXSolver& operator=(const AMGXSolver&) = delete;

    void setup(int n, int nnz, const int* row_ptr, const int* col_idx, const double* values);

    // Swap only the matrix values (fixed sparsity), then rebuild the numeric hierarchy. The
    // aggregation graph is reused when the config sets structure_reuse_levels, so this is far
    // cheaper than setup(); update_coefficients without a following resetup freezes the
    // preconditioner (used by the conductivity UQ sweep).
    void update_coefficients(int nnz, const double* values);
    void resetup();

    void solve(int n, const double* b, double* x, cudaStream_t stream);
    void apply(int n, const double* b, double* x);
    int iterations() const;

private:
    int n_ = 0;
    AMGX_config_handle cfg_ = nullptr;
    AMGX_matrix_handle A_ = nullptr;
    AMGX_vector_handle b_ = nullptr;
    AMGX_vector_handle x_ = nullptr;
    AMGX_solver_handle solver_ = nullptr;
};

class AMGXFloatSolver {
public:
    explicit AMGXFloatSolver(const std::string& config);
    ~AMGXFloatSolver();

    AMGXFloatSolver(const AMGXFloatSolver&) = delete;
    AMGXFloatSolver& operator=(const AMGXFloatSolver&) = delete;

    void setup(int n, int nnz, const int* row_ptr, const int* col_idx, const float* values);
    void apply(int n, const float* b, float* x);
    int iterations() const;
    // Bumped on every setup(): a captured CUDA graph embeds pointers into this solver's hierarchy
    // buffers, so PcgAmgSolver keys its cached graph on (solver address, generation) and recaptures
    // after any re-setup.
    int generation() const { return generation_; }

private:
    int n_ = 0;
    int generation_ = 0;
    AMGX_config_handle cfg_ = nullptr;
    AMGX_matrix_handle A_ = nullptr;
    AMGX_vector_handle b_ = nullptr;
    AMGX_vector_handle x_ = nullptr;
    AMGX_solver_handle solver_ = nullptr;
};

class PcgAmgSolver {
public:
    PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx, const double* values);
    ~PcgAmgSolver();

    PcgAmgSolver(const PcgAmgSolver&) = delete;
    PcgAmgSolver& operator=(const PcgAmgSolver&) = delete;

    void update_values(const double* values, cudaStream_t stream);
    PcgResult solve(AMGXSolver& preconditioner, const double* b, double* x, double tolerance,
                    int max_iters);
    PcgResult solve_mixed(AMGXFloatSolver& preconditioner, const double* b, double* x,
                          double tolerance, int max_iters, cudaStream_t stream,
                          const double* x0 = nullptr);

private:
    int n_ = 0;
    int nnz_ = 0;
    int* row_ptr_ = nullptr;
    int* col_idx_ = nullptr;
    double* values_ = nullptr;
    double* r_ = nullptr;
    double* z_ = nullptr;
    double* p_ = nullptr;
    double* ap_ = nullptr;
    double* x_int_ = nullptr;
    float* rf_ = nullptr;
    float* zf_ = nullptr;
    void* spmv_buffer_ = nullptr;
    // CG scalars kept on-device (device-pointer-mode cuBLAS): [rz, rz_next, pap, alpha, neg_alpha,
    // norm, beta, one]. Only the residual norm is copied back, into pinned host memory, once/iter.
    double* scalars_ = nullptr;
    // Per-block partial sums for the fused deterministic reductions (‖r‖², r·z).
    double* partials_ = nullptr;
    double* h_norm_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    cusparseHandle_t sparse_ = nullptr;
    cusparseSpMatDescr_t mat_ = nullptr;
    cusparseDnVecDescr_t p_vec_ = nullptr;
    cusparseDnVecDescr_t ap_vec_ = nullptr;
    // solve_mixed runs on this internal, capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream; b/x are handed off via join_event_. The iteration body
    // only touches solver-owned buffers (x_int_, not the caller's x), so the captured graph is
    // reused across solves as long as the preconditioner identity/generation is unchanged.
    cudaStream_t solve_stream_ = nullptr;
    cudaEvent_t join_event_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    const AMGXFloatSolver* captured_precond_ = nullptr;
    int captured_precond_gen_ = 0;
};

PcgResult pcg_amg_solve(int n, int nnz, const int* row_ptr, const int* col_idx,
                        const double* values, AMGXSolver& preconditioner, const double* b,
                        double* x, double tolerance, int max_iters);

void launch_double_to_float(const double* in, float* out, int n, cudaStream_t stream);
void launch_float_to_double(const float* in, double* out, int n, cudaStream_t stream);
void launch_cg_alpha(const double* rz, const double* pap, double* alpha, double* neg_alpha,
                     cudaStream_t stream);
void launch_cg_beta(const double* rz_next, double* rz, double* beta, cudaStream_t stream);
void launch_cg_update_xr(const double* alpha, const double* neg_alpha, const double* p,
                         const double* ap, double* x, double* r, float* rf, int n,
                         cudaStream_t stream);
void launch_cg_update_p(const double* beta, const double* z, double* p, int n,
                        cudaStream_t stream);
void launch_csrmv_f64(int n, const int* row_ptr, const int* col_idx, const double* vals,
                      const double* x, double* y, cudaStream_t stream);
int cg_partials_size(int n);
void launch_cg_update_xr_norm(const double* alpha, const double* neg_alpha, const double* p,
                              const double* ap, double* x, double* r, float* rf,
                              double* partials, double* norm_sq, int n, cudaStream_t stream);
void launch_cg_cast_dot_beta(const float* zf, double* z, const double* r, double* partials,
                             double* rz, double* rz_next, double* beta, int n,
                             cudaStream_t stream);
