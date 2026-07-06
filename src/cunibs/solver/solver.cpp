#include "solver.hpp"

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

void check(AMGX_RC rc, const char* what) {
    if (rc != AMGX_RC_OK) {
        char msg[4096];
        AMGX_get_error_string(rc, msg, sizeof(msg));
        throw std::runtime_error(std::string("AMGx ") + what + ": " + msg);
    }
}

extern "C" void amgx_print_filter(const char* msg, int length) {
    std::string s(msg, static_cast<size_t>(length));
    if (s.rfind("AMGX version", 0) == 0 ||
        s.rfind("Built on", 0) == 0 ||
        s.rfind("Compiled with CUDA Runtime", 0) == 0) {
        return;
    }
    std::fwrite(msg, 1, static_cast<size_t>(length), stderr);
}

// AMGx initialization is process-global. Keep it alive until process exit.
void initialize_amgx_once() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        AMGX_register_print_callback(amgx_print_filter);
        check(AMGX_initialize(), "initialize");
    });
}

constexpr AMGX_Mode kMode = AMGX_mode_dDDI;

// AMGx binds its temporary device memory pool to the resources object, so per-solver resources make
// concurrent solvers corrupt that pool ("trying to free non-empty temporary device pool"). ADM keeps
// multiple solvers alive at once, so the resources is a shared process-global singleton.
AMGX_resources_handle shared_resources() {
    static AMGX_config_handle rsc_cfg = nullptr;
    static AMGX_resources_handle rsc = nullptr;
    static std::once_flag flag;
    std::call_once(flag, [] {
        initialize_amgx_once();
        check(AMGX_config_create(&rsc_cfg, "config_version=2"), "config_create(resources)");
        check(AMGX_resources_create_simple(&rsc, rsc_cfg), "resources_create_simple");
    });
    return rsc;
}

}  // namespace

AMGXSolver::AMGXSolver(const std::string& config) {
    AMGX_resources_handle rsc = shared_resources();
    check(AMGX_config_create(&cfg_, config.c_str()), "config_create");
    check(AMGX_matrix_create(&A_, rsc, kMode), "matrix_create");
    check(AMGX_vector_create(&b_, rsc, kMode), "vector_create(b)");
    check(AMGX_vector_create(&x_, rsc, kMode), "vector_create(x)");
    check(AMGX_solver_create(&solver_, rsc, kMode, cfg_), "solver_create");
}

AMGXSolver::~AMGXSolver() {
    // AMGx destroy functions return errors, but this destructor cannot report them.
    // The shared resources (and the global init) are intentionally never destroyed.
    if (solver_) AMGX_solver_destroy(solver_);
    if (x_) AMGX_vector_destroy(x_);
    if (b_) AMGX_vector_destroy(b_);
    if (A_) AMGX_matrix_destroy(A_);
    if (cfg_) AMGX_config_destroy(cfg_);
}

void AMGXSolver::setup(int n, int nnz, const int* row_ptr, const int* col_idx,
                       const double* values) {
    check(AMGX_matrix_upload_all(A_, n, nnz, 1, 1, row_ptr, col_idx, values, nullptr),
          "matrix_upload_all");
    check(AMGX_solver_setup(solver_, A_), "solver_setup");
}

void AMGXSolver::solve(int n, const double* b, double* x) {
    check(AMGX_vector_upload(b_, n, 1, b), "vector_upload(b)");
    check(AMGX_vector_set_zero(x_, n, 1), "vector_set_zero(x)");
    check(AMGX_solver_solve(solver_, b_, x_), "solver_solve");

    AMGX_SOLVE_STATUS status;
    check(AMGX_solver_get_status(solver_, &status), "solver_get_status");
    if (status != AMGX_SOLVE_SUCCESS) {
        int iters = -1;
        AMGX_solver_get_iterations_number(solver_, &iters);
        throw std::runtime_error("AMGx solve did not converge (status=" +
                                 std::to_string(status) + ", iterations=" +
                                 std::to_string(iters) + ")");
    }
    check(AMGX_vector_download(x_, x), "vector_download(x)");
}
