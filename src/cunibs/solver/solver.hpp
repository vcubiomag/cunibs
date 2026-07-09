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

    void solve(int n, const double* b, double* x);
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

private:
    int n_ = 0;
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

    void update_values(const double* values);
    PcgResult solve(AMGXSolver& preconditioner, const double* b, double* x, double tolerance,
                    int max_iters);
    PcgResult solve_mixed(AMGXFloatSolver& preconditioner, const double* b, double* x,
                          double tolerance, int max_iters);

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
    float* rf_ = nullptr;
    float* zf_ = nullptr;
    void* spmv_buffer_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    cusparseHandle_t sparse_ = nullptr;
    cusparseSpMatDescr_t mat_ = nullptr;
    cusparseDnVecDescr_t p_vec_ = nullptr;
    cusparseDnVecDescr_t ap_vec_ = nullptr;
};

PcgResult pcg_amg_solve(int n, int nnz, const int* row_ptr, const int* col_idx,
                        const double* values, AMGXSolver& preconditioner, const double* b,
                        double* x, double tolerance, int max_iters);

void launch_double_to_float(const double* in, float* out, int n, cudaStream_t stream);
void launch_float_to_double(const float* in, double* out, int n, cudaStream_t stream);
