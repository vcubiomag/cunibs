#include "solver.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t rc, const char* what) { ::check_cuda(rc, "solver", what); }

void check_cublas(cublasStatus_t rc, const char* what) {
    if (rc != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS ") + what + ": status " +
                                 std::to_string(rc));
    }
}

}  // namespace

namespace {

template <typename T>
DeviceBuffer<T> alloc(std::size_t count, const char* what) {
    return device_alloc<T>(count, "solver", what);
}

template <typename T>
DeviceBuffer<T> clone(const T* src, std::size_t count, const char* what) {
    return device_clone<T>(src, count, "solver", what);
}

}  // namespace

// Every member is a move-only owner, so a throw part way through releases what was already
// taken and the destructor is the implicit one.
PcgAmgSolver::PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx,
                           const double* values)
    : n_(n), nnz_(nnz) {
    if (n_ <= 0) throw std::invalid_argument("PcgAmgSolver: n must be positive");
    if (nnz_ < 0) throw std::invalid_argument("PcgAmgSolver: nnz must not be negative");

    check_cublas(cublasCreate(&blas_.h), "create");
    row_ptr_ = clone(row_ptr, static_cast<size_t>(n_) + 1, "row_ptr");
    col_idx_ = clone(col_idx, static_cast<size_t>(nnz_), "col_idx");
    values_ = clone(values, static_cast<size_t>(nnz_), "values");
    r_ = alloc<double>(n_, "r");
    p_ = alloc<double>(n_, "p");
    ap_ = alloc<double>(n_, "ap");
    x_int_ = alloc<double>(n_, "x_int");
    rf_ = alloc<float>(n_, "rf");
    zf_ = alloc<float>(n_, "zf");
    scalars_ = alloc<double>(6, "scalars");
    partials_ = alloc<double>(static_cast<size_t>(cg_partials_size(n_)), "partials");
    h_norm_ = pinned_alloc<double>(1, "solver", "norm");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create(solve_stream)");
    solve_stream_.reset(stream);
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create(join_event)");
    join_event_.reset(event);
}

PcgAmgSolver::~PcgAmgSolver() = default;

void PcgAmgSolver::update_values(const double* values, cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(values_.get(), values, static_cast<size_t>(nnz_) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream),
               "copy(values)");
}

void PcgAmgSolver::ensure_block_buffers(int k) {
    if (block_k_ >= k) return;
    // Drop the width before reallocating: if an allocation below throws, the old buffers are
    // already released and a retry at the same k must not take the early-out above.
    block_k_ = 0;
    const size_t nk = static_cast<size_t>(n_) * k;
    r_blk_ = alloc<double>(nk, "R_blk");
    p_blk_ = alloc<double>(nk, "P_blk");
    ap_blk_ = alloc<double>(nk, "AP_blk");
    x_int_blk_ = alloc<double>(nk, "X_int_blk");
    rf_blk_ = alloc<float>(nk, "RF_blk");
    zf_blk_ = alloc<float>(nk, "ZF_blk");
    scalars_blk_ = alloc<double>(static_cast<size_t>(6) * k, "scalars_blk");
    partials_blk_ =
        alloc<double>(static_cast<size_t>(k) * bcg_partials_blocks(n_), "partials_blk");
    h_norms_blk_ = pinned_alloc<double>(static_cast<size_t>(2) * k, "solver", "norms_blk");
    block_k_ = k;
}

PcgBlockResult PcgAmgSolver::solve_mixed_block(NativeVCycle& preconditioner, const double* B,
                                               double* X, int k, double tolerance,
                                               int max_iters, cudaStream_t stream,
                                               const double* X0) {
    if (max_iters <= 0) throw std::invalid_argument("solve_mixed_block: max_iters must be > 0");
    ensure_block_buffers(k);
    // The solve runs on an internal capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream.
    cudaStream_t s = solve_stream_.get();
    check_cuda(cudaEventRecord(join_event_.get(), stream), "block:record_in");
    check_cuda(cudaStreamWaitEvent(s, join_event_.get(), 0), "block:wait_in");

    const size_t nk = static_cast<size_t>(n_) * k;
    const std::int64_t nk_signed = static_cast<std::int64_t>(nk);
    double* const scalars = scalars_blk_.get();
    double* const d_rz = scalars + 0 * k;
    double* const d_pap = scalars + 1 * k;
    double* const d_alpha = scalars + 2 * k;
    double* const d_neg_alpha = scalars + 3 * k;
    double* const d_norm = scalars + 4 * k;
    double* const d_beta = scalars + 5 * k;
    double* const h_norm = h_norms_blk_.get();
    double* const h_ref = h_norms_blk_.get() + k;
    double* const partials = partials_blk_.get();
    double* const R = r_blk_.get();
    double* const P = p_blk_.get();
    double* const AP = ap_blk_.get();
    double* const X_int = x_int_blk_.get();
    float* const RF = rf_blk_.get();
    float* const ZF = zf_blk_.get();

    // Reference norms ‖b_c‖ (convergence is measured against b, matching solve_mixed).
    launch_bcg_norm2(n_, k, B, partials, d_norm, s);
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
        check_cuda(cudaMemcpyAsync(X_int, X0, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
                   "block:copy(X0)");
        launch_bcsrmv_f64_block(n_, k, row_ptr_.get(), col_idx_.get(), values_.get(), X_int, AP,
                                s);
        launch_bcg_residual(nk_signed, B, AP, R, s);
    } else {
        check_cuda(cudaMemsetAsync(X_int, 0, nk * sizeof(double), s), "block:memset(X)");
        check_cuda(cudaMemcpyAsync(R, B, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
                   "block:copy(B,R)");
    }

    launch_bcg_d2f(nk_signed, R, RF, s);
    preconditioner.apply_block(n_, k, RF, ZF, s);
    launch_bcg_cast_dot_init(n_, k, ZF, R, partials, d_rz, s);
    launch_bcg_f2d(nk_signed, ZF, P, s);

    auto run_body = [&]() {
        launch_bcsrmv_f64_block(n_, k, row_ptr_.get(), col_idx_.get(), values_.get(), P, AP, s);
        launch_bcg_dot(n_, k, P, AP, partials, d_pap, s);
        launch_bcg_alpha(k, d_rz, d_pap, d_alpha, d_neg_alpha, s);
        launch_bcg_update_xr_norm(n_, k, d_alpha, d_neg_alpha, P, AP, X_int, R, RF, partials,
                                  d_norm, s);
        check_cuda(cudaMemcpyAsync(h_norm, d_norm, static_cast<size_t>(k) * sizeof(double),
                                   cudaMemcpyDeviceToHost, s),
                   "block:copy(norms)");
        preconditioner.apply_block(n_, k, RF, ZF, s);
        launch_bcg_cast_dot_beta(n_, k, ZF, R, partials, d_rz, d_beta, s);
        launch_bcg_update_p(n_, k, d_beta, ZF, P, s);
    };

    const GraphCache::Key key{&preconditioner, preconditioner.generation(), k};
    std::vector<double> rel(k, 0.0);
    int result_iters = max_iters;
    for (int it = 1; it <= max_iters; ++it) {
        block_graph_.run(s, key, it, run_body);
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
    check_cuda(cudaMemcpyAsync(X, X_int, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
               "block:copy(X_out)");
    check_cuda(cudaStreamSynchronize(s), "block:sync(X_out)");
    PcgBlockResult result;
    result.iterations = result_iters;
    result.relative_residual = std::move(rel);
    return result;
}

PcgResult PcgAmgSolver::solve_mixed(NativeVCycle& preconditioner, const double* b,
                                    double* x, double tolerance, int max_iters,
                                    cudaStream_t stream, const double* x0) {
    // The solve runs on an internal capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream. If capture is invalidated at runtime the loop falls
    // back to direct execution.
    if (max_iters <= 0) throw std::invalid_argument("solve_mixed: max_iters must be > 0");
    cudaStream_t s = solve_stream_.get();
    cublasHandle_t blas = blas_.h;
    check_cuda(cudaEventRecord(join_event_.get(), stream), "graph:record_in");
    check_cuda(cudaStreamWaitEvent(s, join_event_.get(), 0), "graph:wait_in");
    check_cublas(cublasSetStream(blas, s), "set_stream(blas)");
    double* const d_rz = scalars_.get() + 0;
    double* const d_pap = scalars_.get() + 1;
    double* const d_alpha = scalars_.get() + 2;
    double* const d_neg_alpha = scalars_.get() + 3;
    double* const d_norm = scalars_.get() + 4;
    double* const d_beta = scalars_.get() + 5;
    double* const r = r_.get();
    double* const p = p_.get();
    double* const ap = ap_.get();
    double* const x_int = x_int_.get();
    float* const rf = rf_.get();
    float* const zf = zf_.get();
    double* const partials = partials_.get();

    // Convergence is measured against ‖b‖, not the warm residual ‖r0‖, so a warm start (x0 != null)
    // still drives to the same 1e-6-of-field criterion instead of stopping early relative to its
    // small initial residual. Setup uses host-pointer mode since it runs outside the captured loop.
    check_cublas(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "set_pointer_mode(host)");
    // The loop works on the solver-owned x_int_ (not the caller's x) so the captured graph contains
    // no per-call pointers and can be replayed across solves; the result is copied out at the end.
    if (x0 != nullptr) {
        check_cublas(cublasDcopy(blas, n_, x0, 1, x_int, 1), "copy(x0,x_int)");
    } else {
        check_cuda(cudaMemsetAsync(x_int, 0, static_cast<size_t>(n_) * sizeof(double), s),
                   "memset(x_int)");
    }
    check_cublas(cublasDcopy(blas, n_, b, 1, r, 1), "copy(b,r)");
    double norm_ref = 0.0;
    check_cublas(cublasDnrm2(blas, n_, b, 1, &norm_ref), "nrm2(b)");
    if (norm_ref == 0.0) {
        check_cuda(cudaMemsetAsync(x, 0, static_cast<size_t>(n_) * sizeof(double), s),
                   "memset(x_out)");
        check_cuda(cudaStreamSynchronize(s), "sync(x_out0)");
        return {0, 0.0};
    }
    if (x0 != nullptr) {
        // r0 = b - A x0
        launch_csrmv_f64(n_, row_ptr_.get(), col_idx_.get(), values_.get(), x_int, ap, s);
        const double neg_one = -1.0;
        check_cublas(cublasDaxpy(blas, n_, &neg_one, ap, 1, r, 1), "axpy(r0)");
        double norm_r0 = 0.0;
        check_cublas(cublasDnrm2(blas, n_, r, 1, &norm_r0), "nrm2(r0)");
        if (norm_r0 / norm_ref <= tolerance) {
            check_cuda(cudaMemcpyAsync(x, x_int, static_cast<size_t>(n_) * sizeof(double),
                                       cudaMemcpyDeviceToDevice, s),
                       "copy(x_int,x)");
            check_cuda(cudaStreamSynchronize(s), "sync(x_out)");
            return {0, norm_r0 / norm_ref};
        }
    }
    // The captured loop needs device-pointer mode so its reductions land in scalars_.
    check_cublas(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE), "set_pointer_mode(device)");

    launch_double_to_float(r, rf, n_, s);
    preconditioner.apply(n_, rf, zf, s);
    // p0 = z0 = (double)zf directly; the fp64 z vector is never materialized (the loop
    // consumes zf on the fly in cast_dot_beta and update_p).
    launch_float_to_double(zf, p, n_, s);
    check_cublas(cublasDdot(blas, n_, r, 1, p, 1, d_rz), "dot(r,z)");

    // Identical every iteration (fixed buffers updated in place), so it is captured once and
    // replayed; the residual readback is inside the body but the host convergence test stays outside.
    auto run_body = [&]() {
        // p·(Ap) stays a separate cuBLAS pass: folding it into the SpMV epilogue makes the
        // block-wide reduction tree stall the SpMV's memory pipeline.
        launch_csrmv_f64(n_, row_ptr_.get(), col_idx_.get(), values_.get(), p, ap, s);
        check_cublas(cublasDdot(blas, n_, p, 1, ap, 1, d_pap), "dot(p,ap)");
        launch_cg_alpha(d_rz, d_pap, d_alpha, d_neg_alpha, s);
        // x += α p; r -= α ap; rf = (float)r; d_norm = ‖r‖² (host takes the sqrt)
        launch_cg_update_xr_norm(d_alpha, d_neg_alpha, p, ap, x_int, r, rf, partials, d_norm, n_,
                                 s);
        check_cuda(
            cudaMemcpyAsync(h_norm_.get(), d_norm, sizeof(double), cudaMemcpyDeviceToHost, s),
            "copy(norm)");
        preconditioner.apply(n_, rf, zf, s);
        // rz' = r·(double)zf; beta = rz'/rz; rz <- rz' (no fp64 z vector)
        launch_cg_cast_dot_beta(zf, r, partials, d_rz, d_beta, n_, s);
        launch_cg_update_p(d_beta, zf, p, n_, s);  // p = β p + (double)zf
    };

    // The body references only solver-owned buffers plus the preconditioner's hierarchy, so a graph
    // captured in an earlier solve stays valid until the preconditioner is re-setup (or replaced).
    const GraphCache::Key key{&preconditioner, preconditioner.generation(), 1};
    double rel = 0.0;
    int result_iters = max_iters;
    for (int it = 1; it <= max_iters; ++it) {
        graph_.run(s, key, it, run_body);
        check_cuda(cudaStreamSynchronize(s), "sync(iter)");
        rel = std::sqrt(*h_norm_.get()) / norm_ref;
        if (rel <= tolerance) {
            result_iters = it;
            break;
        }
    }
    check_cuda(cudaMemcpyAsync(x, x_int, static_cast<size_t>(n_) * sizeof(double),
                               cudaMemcpyDeviceToDevice, s),
               "copy(x_int,x)");
    check_cuda(cudaStreamSynchronize(s), "sync(x_out)");
    return {result_iters, rel};
}
