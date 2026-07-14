#include "solver.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>
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

void check_cusparse(cusparseStatus_t rc, const char* what) {
    if (rc != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSPARSE ") + what + ": status " +
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

void AMGXSolver::apply(int n, const double* b, double* x) {
    check(AMGX_vector_upload(b_, n, 1, b), "vector_upload(b)");
    check(AMGX_vector_set_zero(x_, n, 1), "vector_set_zero(x)");
    check(AMGX_solver_solve_with_0_initial_guess(solver_, b_, x_), "solver_apply");
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
    n_ = n;
    check(AMGX_matrix_upload_all(A_, n, nnz, 1, 1, row_ptr, col_idx, values, nullptr),
          "matrix_upload_all(float)");
    check(AMGX_solver_setup(solver_, A_), "solver_setup(float)");
}

void AMGXFloatSolver::apply(int n, const float* b, float* x) {
    check(AMGX_vector_upload(b_, n, 1, b), "vector_upload(float b)");
    check(AMGX_vector_set_zero(x_, n, 1), "vector_set_zero(float x)");
    check(AMGX_solver_solve_with_0_initial_guess(solver_, b_, x_), "solver_apply(float)");
    check(AMGX_vector_download(x_, x), "vector_download(float x)");
}

int AMGXFloatSolver::iterations() const {
    int iters = -1;
    check(AMGX_solver_get_iterations_number(solver_, &iters),
          "solver_get_iterations_number(float)");
    return iters;
}

PcgAmgSolver::PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx,
                           const double* values)
    : n_(n), nnz_(nnz) {
    try {
        check_cublas(cublasCreate(&blas_), "create");
        check_cusparse(cusparseCreate(&sparse_), "create");
        check_cuda(cudaMalloc(&row_ptr_, static_cast<size_t>(n_ + 1) * sizeof(int)),
                   "malloc(row_ptr)");
        check_cuda(cudaMalloc(&col_idx_, static_cast<size_t>(nnz_) * sizeof(int)),
                   "malloc(col_idx)");
        check_cuda(cudaMalloc(&values_, static_cast<size_t>(nnz_) * sizeof(double)),
                   "malloc(values)");
        check_cuda(cudaMalloc(&r_, static_cast<size_t>(n_) * sizeof(double)), "malloc(r)");
        check_cuda(cudaMalloc(&z_, static_cast<size_t>(n_) * sizeof(double)), "malloc(z)");
        check_cuda(cudaMalloc(&p_, static_cast<size_t>(n_) * sizeof(double)), "malloc(p)");
        check_cuda(cudaMalloc(&ap_, static_cast<size_t>(n_) * sizeof(double)), "malloc(ap)");
        check_cuda(cudaMalloc(&rf_, static_cast<size_t>(n_) * sizeof(float)), "malloc(rf)");
        check_cuda(cudaMalloc(&zf_, static_cast<size_t>(n_) * sizeof(float)), "malloc(zf)");
        check_cuda(cudaMemcpy(row_ptr_, row_ptr, static_cast<size_t>(n_ + 1) * sizeof(int),
                              cudaMemcpyDeviceToDevice),
                   "copy(row_ptr)");
        check_cuda(cudaMemcpy(col_idx_, col_idx, static_cast<size_t>(nnz_) * sizeof(int),
                              cudaMemcpyDeviceToDevice),
                   "copy(col_idx)");
        update_values(values, nullptr);

        check_cusparse(cusparseCreateCsr(&mat_, n_, n_, nnz_, row_ptr_, col_idx_, values_,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                       "create_csr");
        check_cusparse(cusparseCreateDnVec(&p_vec_, n_, p_, CUDA_R_64F), "create_p_vec");
        check_cusparse(cusparseCreateDnVec(&ap_vec_, n_, ap_, CUDA_R_64F), "create_ap_vec");

        const double one = 1.0;
        const double zero = 0.0;
        size_t spmv_buffer_size = 0;
        check_cusparse(cusparseSpMV_bufferSize(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                               mat_, p_vec_, &zero, ap_vec_, CUDA_R_64F,
                                               CUSPARSE_SPMV_CSR_ALG1, &spmv_buffer_size),
                       "spmv_buffer_size");
        check_cuda(cudaMalloc(&spmv_buffer_, spmv_buffer_size), "malloc(spmv_buffer)");

        check_cuda(cudaMalloc(&scalars_, 8 * sizeof(double)), "malloc(scalars)");
        check_cuda(cudaMallocHost(&h_norm_, sizeof(double)), "mallocHost(norm)");
        const double host_one = 1.0;
        check_cuda(cudaMemcpy(scalars_ + 7, &host_one, sizeof(double), cudaMemcpyHostToDevice),
                   "copy(one)");
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
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (join_event_) cudaEventDestroy(join_event_);
    if (solve_stream_) cudaStreamDestroy(solve_stream_);
    if (h_norm_) cudaFreeHost(h_norm_);
    if (scalars_) cudaFree(scalars_);
    if (spmv_buffer_) cudaFree(spmv_buffer_);
    if (ap_vec_) cusparseDestroyDnVec(ap_vec_);
    if (p_vec_) cusparseDestroyDnVec(p_vec_);
    if (mat_) cusparseDestroySpMat(mat_);
    if (ap_) cudaFree(ap_);
    if (p_) cudaFree(p_);
    if (z_) cudaFree(z_);
    if (r_) cudaFree(r_);
    if (zf_) cudaFree(zf_);
    if (rf_) cudaFree(rf_);
    if (values_) cudaFree(values_);
    if (col_idx_) cudaFree(col_idx_);
    if (row_ptr_) cudaFree(row_ptr_);
    if (sparse_) cusparseDestroy(sparse_);
    if (blas_) cublasDestroy(blas_);
}

void PcgAmgSolver::update_values(const double* values, cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(values_, values, static_cast<size_t>(nnz_) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream),
               "copy(values)");
}

PcgResult PcgAmgSolver::solve(AMGXSolver& preconditioner, const double* b, double* x,
                              double tolerance, int max_iters) {
    // solve_mixed leaves the shared cuBLAS handle in device-pointer mode; these reductions use host
    // pointers, so restore host mode.
    check_cublas(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST), "set_pointer_mode(host)");
    check_cuda(cudaMemset(x, 0, static_cast<size_t>(n_) * sizeof(double)), "memset(x)");
    check_cublas(cublasDcopy(blas_, n_, b, 1, r_, 1), "copy(b,r)");

    double norm0 = 0.0;
    check_cublas(cublasDnrm2(blas_, n_, r_, 1, &norm0), "nrm2(r0)");
    if (norm0 == 0.0) {
        return {0, 0.0};
    }

    preconditioner.apply(n_, r_, z_);
    check_cublas(cublasDcopy(blas_, n_, z_, 1, p_, 1), "copy(z,p)");

    double rz = 0.0;
    check_cublas(cublasDdot(blas_, n_, r_, 1, z_, 1, &rz), "dot(r,z)");

    const double one = 1.0;
    const double zero = 0.0;
    for (int it = 1; it <= max_iters; ++it) {
        check_cusparse(cusparseSpMV(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_, p_vec_,
                                    &zero, ap_vec_, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1,
                                    spmv_buffer_),
                       "spmv");
        double pap = 0.0;
        check_cublas(cublasDdot(blas_, n_, p_, 1, ap_, 1, &pap), "dot(p,ap)");
        const double alpha = rz / pap;
        check_cublas(cublasDaxpy(blas_, n_, &alpha, p_, 1, x, 1), "axpy(x)");
        const double neg_alpha = -alpha;
        check_cublas(cublasDaxpy(blas_, n_, &neg_alpha, ap_, 1, r_, 1), "axpy(r)");

        double norm = 0.0;
        check_cublas(cublasDnrm2(blas_, n_, r_, 1, &norm), "nrm2(r)");
        const double rel = norm / norm0;
        if (rel <= tolerance) {
            return {it, rel};
        }

        preconditioner.apply(n_, r_, z_);
        double rz_next = 0.0;
        check_cublas(cublasDdot(blas_, n_, r_, 1, z_, 1, &rz_next), "dot(r,z)");
        const double beta = rz_next / rz;
        check_cublas(cublasDscal(blas_, n_, &beta, p_, 1), "scal(p)");
        check_cublas(cublasDaxpy(blas_, n_, &one, z_, 1, p_, 1), "axpy(p)");
        rz = rz_next;
    }

    double norm = 0.0;
    check_cublas(cublasDnrm2(blas_, n_, r_, 1, &norm), "nrm2(r_final)");
    return {max_iters, norm / norm0};
}

PcgResult PcgAmgSolver::solve_mixed(AMGXFloatSolver& preconditioner, const double* b,
                                    double* x, double tolerance, int max_iters,
                                    cudaStream_t stream, const double* x0) {
    // Default-on (CUNIBS_GRAPH=0 forces the direct path). The solve runs on an internal
    // capture-capable stream because the caller's is usually the un-capturable legacy default stream;
    // the AMGx fork routes its kernels onto that stream so the preconditioner apply is capturable too.
    // If capture is invalidated at runtime the loop falls back to direct execution.
    const char* graph_env = getenv("CUNIBS_GRAPH");
    const bool use_graph = (graph_env == nullptr) || (graph_env[0] != '0');
    cudaStream_t s = use_graph ? solve_stream_ : stream;
    if (use_graph) {
        check_cuda(cudaEventRecord(join_event_, stream), "graph:record_in");
        check_cuda(cudaStreamWaitEvent(s, join_event_, 0), "graph:wait_in");
    }
    check_cublas(cublasSetStream(blas_, s), "set_stream(blas)");
    check_cusparse(cusparseSetStream(sparse_, s), "set_stream(sparse)");
    check(AMGX_set_thread_stream(reinterpret_cast<void*>(s)), "set_thread_stream");
    double* const d_rz = scalars_ + 0;
    double* const d_rz_next = scalars_ + 1;
    double* const d_pap = scalars_ + 2;
    double* const d_alpha = scalars_ + 3;
    double* const d_neg_alpha = scalars_ + 4;
    double* const d_norm = scalars_ + 5;
    double* const d_beta = scalars_ + 6;
    const double one = 1.0;
    const double zero = 0.0;

    // Convergence is measured against ‖b‖, not the warm residual ‖r0‖, so a warm start (x0 != null)
    // still drives to the same 1e-6-of-field criterion instead of stopping early relative to its
    // small initial residual. With x0 == 0 this is the old path exactly (r0 = b, so ‖b‖ was the
    // reference either way). Setup uses host-pointer mode since it runs outside the captured loop.
    check_cublas(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST), "set_pointer_mode(host)");
    if (x0 != nullptr) {
        check_cublas(cublasDcopy(blas_, n_, x0, 1, x, 1), "copy(x0,x)");
    } else {
        check_cuda(cudaMemsetAsync(x, 0, static_cast<size_t>(n_) * sizeof(double), s), "memset(x)");
    }
    check_cublas(cublasDcopy(blas_, n_, b, 1, r_, 1), "copy(b,r)");
    double norm_ref = 0.0;
    check_cublas(cublasDnrm2(blas_, n_, b, 1, &norm_ref), "nrm2(b)");
    if (norm_ref == 0.0) {
        if (use_graph) { check(AMGX_set_thread_stream(nullptr), "reset_thread_stream"); }
        return {0, 0.0};
    }
    if (x0 != nullptr) {
        // r0 = b - A x0. Retarget the p-vector descriptor at x to reuse the cached SpMV plan/buffer,
        // then restore it to p_ before the captured loop reads it.
        check_cusparse(cusparseDnVecSetValues(p_vec_, x), "spmv:set_x0");
        check_cusparse(cusparseSpMV(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_, p_vec_,
                                    &zero, ap_vec_, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, spmv_buffer_),
                       "spmv(Ax0)");
        check_cusparse(cusparseDnVecSetValues(p_vec_, p_), "spmv:restore_p");
        const double neg_one = -1.0;
        check_cublas(cublasDaxpy(blas_, n_, &neg_one, ap_, 1, r_, 1), "axpy(r0)");
        double norm_r0 = 0.0;
        check_cublas(cublasDnrm2(blas_, n_, r_, 1, &norm_r0), "nrm2(r0)");
        if (norm_r0 / norm_ref <= tolerance) {
            if (use_graph) { check(AMGX_set_thread_stream(nullptr), "reset_thread_stream"); }
            return {0, norm_r0 / norm_ref};
        }
    }
    // The captured loop needs device-pointer mode so its reductions land in scalars_.
    check_cublas(cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_DEVICE), "set_pointer_mode(device)");

    launch_double_to_float(r_, rf_, n_, s);
    check_cuda(cudaStreamSynchronize(s), "sync(precond_in0)");
    preconditioner.apply(n_, rf_, zf_);
    launch_float_to_double(zf_, z_, n_, s);
    check_cublas(cublasDcopy(blas_, n_, z_, 1, p_, 1), "copy(z,p)");
    check_cublas(cublasDdot(blas_, n_, r_, 1, z_, 1, d_rz), "dot(r,z)");

    // Identical every iteration (fixed buffers updated in place), so it is captured once and
    // replayed; the residual readback is inside the body but the host convergence test stays outside.
    auto run_body = [&]() {
        check_cusparse(cusparseSpMV(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_, p_vec_,
                                    &zero, ap_vec_, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, spmv_buffer_),
                       "spmv");
        check_cublas(cublasDdot(blas_, n_, p_, 1, ap_, 1, d_pap), "dot(p,ap)");
        launch_cg_alpha(d_rz, d_pap, d_alpha, d_neg_alpha, s);
        // x += α p; r -= α ap; rf = (float)r
        launch_cg_update_xr(d_alpha, d_neg_alpha, p_, ap_, x, r_, rf_, n_, s);
        check_cublas(cublasDnrm2(blas_, n_, r_, 1, d_norm), "nrm2(r)");
        check_cuda(cudaMemcpyAsync(h_norm_, d_norm, sizeof(double), cudaMemcpyDeviceToHost, s),
                   "copy(norm)");
        preconditioner.apply(n_, rf_, zf_);
        launch_float_to_double(zf_, z_, n_, s);
        check_cublas(cublasDdot(blas_, n_, r_, 1, z_, 1, d_rz_next), "dot(r,z)");
        launch_cg_beta(d_rz_next, d_rz, d_beta, s);  // beta = rz_next/rz; rz <- rz_next
        launch_cg_update_p(d_beta, z_, p_, n_, s);  // p = β p + z
    };

    // The graph body writes to the caller's x, which changes per call, so recapture each solve.
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }

    double rel = 0.0;
    bool have_graph = false;
    bool capture_failed = !use_graph;  // when off, always execute the body directly (no capture)
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
        rel = *h_norm_ / norm_ref;
        if (rel <= tolerance) {
            result_iters = it;
            break;
        }
    }
    // solve_stream_ is destroyed with this solver, so reset AMGx's per-thread stream to the default
    // rather than leave it dangling for a later AMGx op on this thread.
    if (use_graph) {
        check(AMGX_set_thread_stream(nullptr), "reset_thread_stream");
    }
    return {result_iters, rel};
}

PcgResult pcg_amg_solve(int n, int nnz, const int* row_ptr, const int* col_idx,
                        const double* values, AMGXSolver& preconditioner, const double* b,
                        double* x, double tolerance, int max_iters) {
    PcgAmgSolver solver(n, nnz, row_ptr, col_idx, values);
    return solver.solve(preconditioner, b, x, tolerance, max_iters);
}
