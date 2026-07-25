#include "solver.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(AMGX_RC rc, const char* what) {
    if (rc != AMGX_RC_OK) {
        char msg[4096];
        AMGX_get_error_string(rc, msg, sizeof(msg));
        throw std::runtime_error(std::string("AMGx ") + what + ": " + msg);
    }
}

void check_cuda(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(rc));
    }
}

void check_cublas(cublasStatus_t rc, const char* what) {
    if (rc != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS ") + what + ": status " +
                                 std::to_string(rc));
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
constexpr AMGX_Mode kFloatMode = AMGX_mode_dFFI;

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
    n_ = n;
    check(AMGX_matrix_upload_all(A_, n, nnz, 1, 1, row_ptr, col_idx, values, nullptr),
          "matrix_upload_all");
    check(AMGX_solver_setup(solver_, A_), "solver_setup");
}

void AMGXSolver::update_coefficients(int nnz, const double* values) {
    check(AMGX_matrix_replace_coefficients(A_, n_, nnz, values, nullptr),
          "matrix_replace_coefficients");
}

void AMGXSolver::resetup() {
    check(AMGX_solver_resetup(solver_, A_), "solver_resetup");
}

void AMGXSolver::solve(int n, const double* b, double* x, cudaStream_t stream) {
    check(AMGX_set_thread_stream(reinterpret_cast<void*>(stream)), "set_thread_stream");
    check(AMGX_vector_upload(b_, n, 1, b), "vector_upload(b)");
    check(AMGX_vector_set_zero(x_, n, 1), "vector_set_zero(x)");
    check(AMGX_solver_solve_with_0_initial_guess(solver_, b_, x_), "solver_solve");

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

int AMGXSolver::iterations() const {
    int iters = -1;
    check(AMGX_solver_get_iterations_number(solver_, &iters), "solver_get_iterations_number");
    return iters;
}

AMGXFloatSolver::AMGXFloatSolver(const std::string& config) {
    AMGX_resources_handle rsc = shared_resources();
    check(AMGX_config_create(&cfg_, config.c_str()), "config_create(float)");
    check(AMGX_matrix_create(&A_, rsc, kFloatMode), "matrix_create(float)");
    check(AMGX_vector_create(&b_, rsc, kFloatMode), "vector_create(float b)");
    check(AMGX_vector_create(&x_, rsc, kFloatMode), "vector_create(float x)");
    check(AMGX_solver_create(&solver_, rsc, kFloatMode, cfg_), "solver_create(float)");
}

AMGXFloatSolver::~AMGXFloatSolver() {
    if (solver_) AMGX_solver_destroy(solver_);
    if (x_) AMGX_vector_destroy(x_);
    if (b_) AMGX_vector_destroy(b_);
    if (A_) AMGX_matrix_destroy(A_);
    if (cfg_) AMGX_config_destroy(cfg_);
}

void AMGXFloatSolver::setup(int n, int nnz, const int* row_ptr, const int* col_idx,
                            const float* values) {
    check(AMGX_matrix_upload_all(A_, n, nnz, 1, 1, row_ptr, col_idx, values, nullptr),
          "matrix_upload_all(float)");
    check(AMGX_solver_setup(solver_, A_), "solver_setup(float)");
}

int AMGXFloatSolver::amg_num_levels() const {
    int n_levels = 0;
    check(AMGX_solver_get_amg_num_levels(solver_, &n_levels), "solver_get_amg_num_levels");
    return n_levels;
}

void AMGXFloatSolver::amg_level_dims(int level, int* n_rows, int* n_nz, int* n_coarse) const {
    check(AMGX_solver_get_amg_level_dims(solver_, level, n_rows, n_nz, n_coarse),
          "solver_get_amg_level_dims");
}

void AMGXFloatSolver::download_aggregates(int level, int* aggregates) const {
    check(AMGX_solver_download_amg_aggregates(solver_, level, aggregates),
          "solver_download_amg_aggregates");
}

PcgAmgSolver::PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx,
                           const double* values)
    : n_(n), nnz_(nnz) {
    try {
        check_cublas(cublasCreate(&blas_), "create");
        check_cuda(cudaMalloc(&row_ptr_, static_cast<size_t>(n_ + 1) * sizeof(int)),
                   "malloc(row_ptr)");
        check_cuda(cudaMalloc(&col_idx_, static_cast<size_t>(nnz_) * sizeof(int)),
                   "malloc(col_idx)");
        check_cuda(cudaMalloc(&values_, static_cast<size_t>(nnz_) * sizeof(double)),
                   "malloc(values)");
        check_cuda(cudaMalloc(&r_, static_cast<size_t>(n_) * sizeof(double)), "malloc(r)");
        check_cuda(cudaMalloc(&p_, static_cast<size_t>(n_) * sizeof(double)), "malloc(p)");
        check_cuda(cudaMalloc(&ap_, static_cast<size_t>(n_) * sizeof(double)), "malloc(ap)");
        check_cuda(cudaMalloc(&x_int_, static_cast<size_t>(n_) * sizeof(double)), "malloc(x_int)");
        check_cuda(cudaMalloc(&rf_, static_cast<size_t>(n_) * sizeof(float)), "malloc(rf)");
        check_cuda(cudaMalloc(&zf_, static_cast<size_t>(n_) * sizeof(float)), "malloc(zf)");
        check_cuda(cudaMemcpy(row_ptr_, row_ptr, static_cast<size_t>(n_ + 1) * sizeof(int),
                              cudaMemcpyDeviceToDevice),
                   "copy(row_ptr)");
        check_cuda(cudaMemcpy(col_idx_, col_idx, static_cast<size_t>(nnz_) * sizeof(int),
                              cudaMemcpyDeviceToDevice),
                   "copy(col_idx)");
        update_values(values, nullptr);

        check_cuda(cudaMalloc(&scalars_, 6 * sizeof(double)), "malloc(scalars)");
        check_cuda(cudaMalloc(&partials_,
                              static_cast<size_t>(cg_partials_size(n_)) * sizeof(double)),
                   "malloc(partials)");
        check_cuda(cudaMallocHost(&h_norm_, sizeof(double)), "mallocHost(norm)");
        check_cuda(cudaStreamCreateWithFlags(&solve_stream_, cudaStreamNonBlocking),
                   "create(solve_stream)");
        check_cuda(cudaEventCreateWithFlags(&join_event_, cudaEventDisableTiming),
                   "create(join_event)");
    } catch (...) {
        this->~PcgAmgSolver();
        throw;
    }
}

PcgAmgSolver::~PcgAmgSolver() {
    // AMGx's per-thread stream may still point at solve_stream_ (bound during solve_mixed's graph
    // path). Reset it to the default before destroying solve_stream_ so a later AMGx op on this
    // thread (e.g. another solver's setup thrust calls) does not dereference a destroyed stream.
    AMGX_set_thread_stream(nullptr);
    if (block_graph_exec_) cudaGraphExecDestroy(block_graph_exec_);
    if (block_graph_) cudaGraphDestroy(block_graph_);
    if (h_norms_blk_) cudaFreeHost(h_norms_blk_);
    cudaFree(partials_blk_);
    cudaFree(scalars_blk_);
    cudaFree(ZF_blk_);
    cudaFree(RF_blk_);
    cudaFree(X_int_blk_);
    cudaFree(AP_blk_);
    cudaFree(P_blk_);
    cudaFree(R_blk_);
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (join_event_) cudaEventDestroy(join_event_);
    if (solve_stream_) cudaStreamDestroy(solve_stream_);
    if (h_norm_) cudaFreeHost(h_norm_);
    if (partials_) cudaFree(partials_);
    if (scalars_) cudaFree(scalars_);
    if (x_int_) cudaFree(x_int_);
    if (ap_) cudaFree(ap_);
    if (p_) cudaFree(p_);
    if (r_) cudaFree(r_);
    if (zf_) cudaFree(zf_);
    if (rf_) cudaFree(rf_);
    if (values_) cudaFree(values_);
    if (col_idx_) cudaFree(col_idx_);
    if (row_ptr_) cudaFree(row_ptr_);
    if (blas_) cublasDestroy(blas_);
}

void PcgAmgSolver::update_values(const double* values, cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(values_, values, static_cast<size_t>(nnz_) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream),
               "copy(values)");
}

void PcgAmgSolver::ensure_block_buffers(int k) {
    if (block_k_ >= k) return;
    cudaFree(R_blk_);
    cudaFree(P_blk_);
    cudaFree(AP_blk_);
    cudaFree(X_int_blk_);
    cudaFree(RF_blk_);
    cudaFree(ZF_blk_);
    cudaFree(scalars_blk_);
    cudaFree(partials_blk_);
    if (h_norms_blk_) cudaFreeHost(h_norms_blk_);
    const size_t nk = static_cast<size_t>(n_) * k;
    check_cuda(cudaMalloc(&R_blk_, nk * sizeof(double)), "malloc(R_blk)");
    check_cuda(cudaMalloc(&P_blk_, nk * sizeof(double)), "malloc(P_blk)");
    check_cuda(cudaMalloc(&AP_blk_, nk * sizeof(double)), "malloc(AP_blk)");
    check_cuda(cudaMalloc(&X_int_blk_, nk * sizeof(double)), "malloc(X_int_blk)");
    check_cuda(cudaMalloc(&RF_blk_, nk * sizeof(float)), "malloc(RF_blk)");
    check_cuda(cudaMalloc(&ZF_blk_, nk * sizeof(float)), "malloc(ZF_blk)");
    check_cuda(cudaMalloc(&scalars_blk_, static_cast<size_t>(6) * k * sizeof(double)),
               "malloc(scalars_blk)");
    check_cuda(cudaMalloc(&partials_blk_,
                          static_cast<size_t>(k) * bcg_partials_blocks(n_) * sizeof(double)),
               "malloc(partials_blk)");
    check_cuda(cudaMallocHost(&h_norms_blk_, static_cast<size_t>(2) * k * sizeof(double)),
               "mallocHost(norms_blk)");
    block_k_ = k;
}

PcgBlockResult PcgAmgSolver::solve_mixed_block(NativeVCycle& preconditioner, const double* B,
                                               double* X, int k, double tolerance,
                                               int max_iters, cudaStream_t stream,
                                               const double* X0) {
    ensure_block_buffers(k);
    // The solve runs on an internal capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream; the AMGx fork routes its kernels onto that stream so
    // the preconditioner apply is capturable too.
    cudaStream_t s = solve_stream_;
    check_cuda(cudaEventRecord(join_event_, stream), "block:record_in");
    check_cuda(cudaStreamWaitEvent(s, join_event_, 0), "block:wait_in");
    check(AMGX_set_thread_stream(reinterpret_cast<void*>(s)), "set_thread_stream(block)");

    const size_t nk = static_cast<size_t>(n_) * k;
    double* const d_rz = scalars_blk_ + 0 * k;
    double* const d_pap = scalars_blk_ + 1 * k;
    double* const d_alpha = scalars_blk_ + 2 * k;
    double* const d_neg_alpha = scalars_blk_ + 3 * k;
    double* const d_norm = scalars_blk_ + 4 * k;
    double* const d_beta = scalars_blk_ + 5 * k;
    double* const h_norm = h_norms_blk_;
    double* const h_ref = h_norms_blk_ + k;

    // Reference norms ‖b_c‖ (convergence is measured against b, matching solve_mixed).
    launch_bcg_norm2(n_, k, B, partials_blk_, d_norm, s);
    check_cuda(cudaMemcpyAsync(h_ref, d_norm, static_cast<size_t>(k) * sizeof(double),
                               cudaMemcpyDeviceToHost, s),
               "block:copy(ref_norms)");
    check_cuda(cudaStreamSynchronize(s), "block:sync(ref_norms)");
    std::vector<double> norm_ref(k);
    for (int c = 0; c < k; ++c) {
        norm_ref[c] = std::sqrt(h_ref[c]);
        if (norm_ref[c] == 0.0) {
            throw std::invalid_argument("solve_mixed_block: zero RHS column");
        }
    }

    if (X0 != nullptr) {
        check_cuda(cudaMemcpyAsync(X_int_blk_, X0, nk * sizeof(double),
                                   cudaMemcpyDeviceToDevice, s),
                   "block:copy(X0)");
        launch_bcsrmv_f64_block(n_, k, row_ptr_, col_idx_, values_, X_int_blk_, AP_blk_, s);
        launch_bcg_residual(static_cast<long>(nk), B, AP_blk_, R_blk_, s);
    } else {
        check_cuda(cudaMemsetAsync(X_int_blk_, 0, nk * sizeof(double), s), "block:memset(X)");
        check_cuda(cudaMemcpyAsync(R_blk_, B, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
                   "block:copy(B,R)");
    }

    launch_bcg_d2f(static_cast<long>(nk), R_blk_, RF_blk_, s);
    check_cuda(cudaStreamSynchronize(s), "block:sync(precond_in0)");
    preconditioner.apply_block(n_, k, RF_blk_, ZF_blk_, s);
    launch_bcg_cast_dot_init(n_, k, ZF_blk_, R_blk_, partials_blk_, d_rz, s);
    launch_bcg_f2d(static_cast<long>(nk), ZF_blk_, P_blk_, s);

    auto run_body = [&]() {
        launch_bcsrmv_f64_block(n_, k, row_ptr_, col_idx_, values_, P_blk_, AP_blk_, s);
        launch_bcg_dot(n_, k, P_blk_, AP_blk_, partials_blk_, d_pap, s);
        launch_bcg_alpha(k, d_rz, d_pap, d_alpha, d_neg_alpha, s);
        launch_bcg_update_xr_norm(n_, k, d_alpha, d_neg_alpha, P_blk_, AP_blk_, X_int_blk_,
                                  R_blk_, RF_blk_, partials_blk_, d_norm, s);
        check_cuda(cudaMemcpyAsync(h_norm, d_norm, static_cast<size_t>(k) * sizeof(double),
                                   cudaMemcpyDeviceToHost, s),
                   "block:copy(norms)");
        preconditioner.apply_block(n_, k, RF_blk_, ZF_blk_, s);
        launch_bcg_cast_dot_beta(n_, k, ZF_blk_, R_blk_, partials_blk_, d_rz, d_beta, s);
        launch_bcg_update_p(n_, k, d_beta, ZF_blk_, P_blk_, s);
    };

    if (block_graph_exec_ != nullptr &&
        (block_captured_precond_ != &preconditioner ||
         block_captured_gen_ != preconditioner.generation() || block_captured_k_ != k)) {
        cudaGraphExecDestroy(block_graph_exec_);
        block_graph_exec_ = nullptr;
        if (block_graph_) {
            cudaGraphDestroy(block_graph_);
            block_graph_ = nullptr;
        }
    }

    std::vector<double> rel(k, 0.0);
    bool have_graph = block_graph_exec_ != nullptr;
    bool capture_failed = false;
    int it = 1;
    int result_iters = max_iters;
    for (; it <= max_iters; ++it) {
        if (have_graph) {
            check_cuda(cudaGraphLaunch(block_graph_exec_, s), "block:graph_launch");
        } else if (it == 1 || capture_failed) {
            run_body();  // iter 1 warms pools/lazy block buffers so capture sees no allocs
        } else {
            cudaError_t cerr = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            run_body();
            cudaError_t eerr = cudaStreamEndCapture(s, &block_graph_);
            if (cerr == cudaSuccess && eerr == cudaSuccess && block_graph_ != nullptr &&
                cudaGraphInstantiate(&block_graph_exec_, block_graph_, 0) == cudaSuccess) {
                have_graph = true;
                block_captured_precond_ = &preconditioner;
                block_captured_gen_ = preconditioner.generation();
                block_captured_k_ = k;
                check_cuda(cudaGraphLaunch(block_graph_exec_, s), "block:graph_launch_first");
            } else {
                if (block_graph_) {
                    cudaGraphDestroy(block_graph_);
                    block_graph_ = nullptr;
                }
                capture_failed = true;
                run_body();
            }
        }
        check_cuda(cudaStreamSynchronize(s), "block:sync(iter)");
        double worst = 0.0;
        for (int c = 0; c < k; ++c) {
            rel[c] = std::sqrt(h_norm[c]) / norm_ref[c];
            if (rel[c] > worst) worst = rel[c];
        }
        if (worst <= tolerance) {
            result_iters = it;
            break;
        }
    }
    check_cuda(cudaMemcpyAsync(X, X_int_blk_, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
               "block:copy(X_out)");
    check_cuda(cudaStreamSynchronize(s), "block:sync(X_out)");
    check(AMGX_set_thread_stream(nullptr), "reset_thread_stream(block)");
    PcgBlockResult result;
    result.iterations = result_iters;
    result.relative_residual = std::move(rel);
    return result;
}

PcgResult PcgAmgSolver::solve_mixed(NativeVCycle& preconditioner, const double* b,
                                    double* x, double tolerance, int max_iters,
                                    cudaStream_t stream, const double* x0) {
    // The solve runs on an internal capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream; the AMGx fork routes its kernels onto that stream so the
    // preconditioner apply is capturable too. If capture is invalidated at runtime the loop falls
    // back to direct execution.
    cudaStream_t s = solve_stream_;
    check_cuda(cudaEventRecord(join_event_, stream), "graph:record_in");
    check_cuda(cudaStreamWaitEvent(s, join_event_, 0), "graph:wait_in");
    check_cublas(cublasSetStream(blas_, s), "set_stream(blas)");
    check(AMGX_set_thread_stream(reinterpret_cast<void*>(s)), "set_thread_stream");
    double* const d_rz = scalars_ + 0;
    double* const d_pap = scalars_ + 1;
    double* const d_alpha = scalars_ + 2;
    double* const d_neg_alpha = scalars_ + 3;
    double* const d_norm = scalars_ + 4;
    double* const d_beta = scalars_ + 5;

    // Convergence is measured against ‖b‖, not the warm residual ‖r0‖, so a warm start (x0 != null)
    // still drives to the same 1e-6-of-field criterion instead of stopping early relative to its
    // small initial residual. Setup uses host-pointer mode since it runs outside the captured loop.
    check_cublas(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST), "set_pointer_mode(host)");
    // The loop works on the solver-owned x_int_ (not the caller's x) so the captured graph contains
    // no per-call pointers and can be replayed across solves; the result is copied out at the end.
    if (x0 != nullptr) {
        check_cublas(cublasDcopy(blas_, n_, x0, 1, x_int_, 1), "copy(x0,x_int)");
    } else {
        check_cuda(cudaMemsetAsync(x_int_, 0, static_cast<size_t>(n_) * sizeof(double), s),
                   "memset(x_int)");
    }
    check_cublas(cublasDcopy(blas_, n_, b, 1, r_, 1), "copy(b,r)");
    double norm_ref = 0.0;
    check_cublas(cublasDnrm2(blas_, n_, b, 1, &norm_ref), "nrm2(b)");
    if (norm_ref == 0.0) {
        check_cuda(cudaMemsetAsync(x, 0, static_cast<size_t>(n_) * sizeof(double), s),
                   "memset(x_out)");
        check_cuda(cudaStreamSynchronize(s), "sync(x_out0)");
        check(AMGX_set_thread_stream(nullptr), "reset_thread_stream");
        return {0, 0.0};
    }
    if (x0 != nullptr) {
        // r0 = b - A x0
        launch_csrmv_f64(n_, row_ptr_, col_idx_, values_, x_int_, ap_, s);
        const double neg_one = -1.0;
        check_cublas(cublasDaxpy(blas_, n_, &neg_one, ap_, 1, r_, 1), "axpy(r0)");
        double norm_r0 = 0.0;
        check_cublas(cublasDnrm2(blas_, n_, r_, 1, &norm_r0), "nrm2(r0)");
        if (norm_r0 / norm_ref <= tolerance) {
            check_cuda(cudaMemcpyAsync(x, x_int_, static_cast<size_t>(n_) * sizeof(double),
                                       cudaMemcpyDeviceToDevice, s),
                       "copy(x_int,x)");
            check_cuda(cudaStreamSynchronize(s), "sync(x_out)");
            check(AMGX_set_thread_stream(nullptr), "reset_thread_stream");
            return {0, norm_r0 / norm_ref};
        }
    }
    // The captured loop needs device-pointer mode so its reductions land in scalars_.
    check_cublas(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_DEVICE), "set_pointer_mode(device)");

    launch_double_to_float(r_, rf_, n_, s);
    check_cuda(cudaStreamSynchronize(s), "sync(precond_in0)");
    preconditioner.apply(n_, rf_, zf_, s);
    // p0 = z0 = (double)zf directly; the fp64 z vector is never materialized (the loop
    // consumes zf on the fly in cast_dot_beta and update_p).
    launch_float_to_double(zf_, p_, n_, s);
    check_cublas(cublasDdot(blas_, n_, r_, 1, p_, 1, d_rz), "dot(r,z)");

    // Identical every iteration (fixed buffers updated in place), so it is captured once and
    // replayed; the residual readback is inside the body but the host convergence test stays outside.
    auto run_body = [&]() {
        // p·(Ap) stays a separate cuBLAS pass: folding it into the SpMV epilogue makes the
        // block-wide reduction tree stall the SpMV's memory pipeline.
        launch_csrmv_f64(n_, row_ptr_, col_idx_, values_, p_, ap_, s);
        check_cublas(cublasDdot(blas_, n_, p_, 1, ap_, 1, d_pap), "dot(p,ap)");
        launch_cg_alpha(d_rz, d_pap, d_alpha, d_neg_alpha, s);
        // x += α p; r -= α ap; rf = (float)r; d_norm = ‖r‖² (host takes the sqrt)
        launch_cg_update_xr_norm(d_alpha, d_neg_alpha, p_, ap_, x_int_, r_, rf_, partials_,
                                 d_norm, n_, s);
        check_cuda(cudaMemcpyAsync(h_norm_, d_norm, sizeof(double), cudaMemcpyDeviceToHost, s),
                   "copy(norm)");
        preconditioner.apply(n_, rf_, zf_, s);
        // rz' = r·(double)zf; beta = rz'/rz; rz <- rz' (no fp64 z vector)
        launch_cg_cast_dot_beta(zf_, r_, partials_, d_rz, d_beta, n_, s);
        launch_cg_update_p(d_beta, zf_, p_, n_, s);  // p = β p + (double)zf
    };

    // The body references only solver-owned buffers plus the preconditioner's hierarchy, so a graph
    // captured in an earlier solve stays valid until the preconditioner is re-setup (or replaced).
    if (graph_exec_ != nullptr && (captured_precond_ != &preconditioner ||
                                   captured_precond_gen_ != preconditioner.generation())) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
        if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
    }

    double rel = 0.0;
    bool have_graph = graph_exec_ != nullptr;
    bool capture_failed = false;
    int it = 1;
    int result_iters = max_iters;
    for (; it <= max_iters; ++it) {
        if (have_graph) {
            check_cuda(cudaGraphLaunch(graph_exec_, s), "graph:launch");
        } else if (it == 1 || capture_failed) {
            run_body();  // iter 1 warms the AMGx pool so capture sees no allocation
        } else {
            cudaError_t cerr = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            run_body();
            cudaError_t eerr = cudaStreamEndCapture(s, &graph_);
            if (cerr == cudaSuccess && eerr == cudaSuccess && graph_ != nullptr &&
                cudaGraphInstantiate(&graph_exec_, graph_, 0) == cudaSuccess) {
                have_graph = true;
                captured_precond_ = &preconditioner;
                captured_precond_gen_ = preconditioner.generation();
                check_cuda(cudaGraphLaunch(graph_exec_, s), "graph:launch_first");
            } else {
                // Capture was invalidated (records but does not execute): run this iteration
                // directly and stop attempting capture for the rest of the solve.
                if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
                capture_failed = true;
                run_body();
            }
        }
        check_cuda(cudaStreamSynchronize(s), "sync(iter)");
        rel = std::sqrt(*h_norm_) / norm_ref;
        if (rel <= tolerance) {
            result_iters = it;
            break;
        }
    }
    check_cuda(cudaMemcpyAsync(x, x_int_, static_cast<size_t>(n_) * sizeof(double),
                               cudaMemcpyDeviceToDevice, s),
               "copy(x_int,x)");
    check_cuda(cudaStreamSynchronize(s), "sync(x_out)");
    // solve_stream_ is destroyed with this solver, so reset AMGx's per-thread stream to the default
    // rather than leave it dangling for a later AMGx op on this thread.
    check(AMGX_set_thread_stream(nullptr), "reset_thread_stream");
    return {result_iters, rel};
}
