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

    void solve(int n, const double* b, double* x);

private:
    AMGX_config_handle cfg_ = nullptr;
    AMGX_matrix_handle A_ = nullptr;
    AMGX_vector_handle b_ = nullptr;
    AMGX_vector_handle x_ = nullptr;
    AMGX_solver_handle solver_ = nullptr;
};
