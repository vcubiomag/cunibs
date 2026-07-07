#pragma once
#include <string>

#include <amgx_c.h>

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

private:
    int n_ = 0;
    AMGX_config_handle cfg_ = nullptr;
    AMGX_matrix_handle A_ = nullptr;
    AMGX_vector_handle b_ = nullptr;
    AMGX_vector_handle x_ = nullptr;
    AMGX_solver_handle solver_ = nullptr;
};
