#include "amg/solver.hpp"

#include "amg/cg_kernels.hpp"
#include "amg/vcycle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t rc, const char* what) { ::check_cuda(rc, "solver", what); }

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

    row_ptr_ = clone(row_ptr, static_cast<size_t>(n_) + 1, "row_ptr");
    slots_ = build_slot_csr(n_, row_ptr, col_idx, "solver");
    values_ = alloc<double>(static_cast<size_t>(slots_.n_slots), "values");
    launch_slot_scatter(n_, row_ptr_.get(), slots_.ptr.get(), slots_.cols.get(), values,
                        values_.get(), nullptr);
    r_ = alloc<double>(n_, "r");
    p_ = alloc<double>(n_, "p");
    ap_ = alloc<double>(n_, "ap");
    x_int_ = alloc<double>(n_, "x_int");
    rf_ = alloc<float>(n_, "rf");
    zf_ = alloc<float>(n_, "zf");
    scalars_ = alloc<double>(8, "scalars");
    partials_ = alloc<double>(static_cast<size_t>(2) * bcg_partials_blocks(n_), "partials");
    h_norm_ = pinned_alloc<double>(1, "solver", "norm");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create(solve_stream)");
    solve_stream_.reset(stream);
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create(join_event)");
    join_event_.reset(event);

    // The deflation support: rows whose sum is more than a rounding residue of their l1 norm. On a
    // grounded conductivity operator that is the ground node's neighbours. Chosen once, from the
    // values this was built with: it is a property of the pattern, and update_values keeps it.
    // Everything here is on the legacy stream the clones above are ordered on, which a
    // non-blocking solve_stream_ does not wait for, so it is drained before any solve can start.
    constexpr double kSupportTol = 1e-10;
    {
        DeviceBuffer<double> sums = alloc<double>(n_, "row sums");
        DeviceBuffer<double> l1 = alloc<double>(n_, "row l1");
        launch_bcg_row_sums(n_, slots_.ptr.get(), values_.get(), sums.get(), l1.get(), nullptr);
        std::vector<double> h_sums(n_), h_l1(n_);
        check_cuda(cudaMemcpy(h_sums.data(), sums.get(), n_ * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "copy(row sums)");
        check_cuda(cudaMemcpy(h_l1.data(), l1.get(), n_ * sizeof(double), cudaMemcpyDeviceToHost),
                   "copy(row l1)");
        std::vector<int> rows;
        for (int i = 0; i < n_; ++i) {
            if (std::fabs(h_sums[i]) > kSupportTol * h_l1[i]) rows.push_back(i);
        }
        defl_n_ = static_cast<int>(rows.size());
        defl_rows_ = alloc<int>(std::max<size_t>(rows.size(), 1), "deflation rows");
        if (!rows.empty()) {
            check_cuda(cudaMemcpy(defl_rows_.get(), rows.data(), rows.size() * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       "copy(deflation rows)");
        }
    }
    defl_aw_ = alloc<double>(std::max<size_t>(defl_n_, 1) * kMaxStageBlock, "deflation aw");
    defl_inv_e_ = alloc<double>(kMaxStageBlock, "deflation inv_e");
    defl_affine_ = true;
    deflation_from_values(nullptr);
    check_cuda(cudaStreamSynchronize(nullptr), "sync(deflation setup)");
}

PcgAmgSolver::~PcgAmgSolver() = default;

Deflation PcgAmgSolver::deflation() const {
    return Deflation{defl_n_, defl_rows_.get(), defl_aw_.get(), defl_inv_e_.get()};
}

void PcgAmgSolver::deflation_from_values(cudaStream_t stream) {
    if (!defl_affine_) return;
    // A low-half slot's value is zero, so a row sum over slots is the row sum over entries.
    launch_bcg_defl_from_values(defl_n_, defl_rows_.get(), slots_.ptr.get(), values_.get(),
                                defl_aw_.get(), defl_inv_e_.get(), stream);
    check_cuda(cudaGetLastError(), "deflation");
    defl_affine_ = false;
}

void PcgAmgSolver::update_values(const double* values, cudaStream_t stream) {
    launch_slot_scatter(n_, row_ptr_.get(), slots_.ptr.get(), slots_.cols.get(), values,
                        values_.get(), stream);
    check_cuda(cudaGetLastError(), "scatter(values)");
    // A solve then waits on its caller's stream, which is this one, before touching either.
    defl_affine_ = true;
    deflation_from_values(stream);
}

void PcgAmgSolver::set_affine_terms(int n_terms, const double* term_values) {
    affine_ = build_affine_slots(n_, nnz_, row_ptr_.get(), slots_.ptr.get(), slots_.cols.get(),
                                 term_values, n_terms, "solver");
    affine_aw_t_ =
        alloc<double>(static_cast<size_t>(n_terms) * std::max(defl_n_, 1), "affine aw_t");
    launch_term_row_sums(defl_n_, defl_rows_.get(), row_ptr_.get(), term_values, nnz_, n_terms,
                         affine_aw_t_.get(), nullptr);
    affine_sigma_ = alloc<double>(static_cast<size_t>(kMaxStageBlock) * n_terms, "affine sigma");
    affine_graph_ = std::make_unique<GraphCache>();
    check_cuda(cudaGetLastError(), "affine terms");
    check_cuda(cudaStreamSynchronize(nullptr), "sync(affine terms)");
}

void PcgAmgSolver::apply_operator(int k, bool affine, const double* x, double* y,
                                  cudaStream_t stream) {
    if (affine) {
        const AffineOperator op{affine_.n_terms,         affine_.order.get(),
                                affine_.seg_ptr.get(),   affine_.seg_start.get(),
                                affine_.seg_term.get(),  affine_.cols.get(),
                                affine_.vals.get()};
        launch_affine_spmv_block(n_, k, op, affine_sigma_.get(), x, y, stream);
    } else {
        launch_bcsrmv_f64_block(n_, k, slots_.ptr.get(), slots_.cols.get(), values_.get(), x, y,
                                stream);
    }
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
    scalars_blk_ = alloc<double>(static_cast<size_t>(10) * k + 1, "scalars_blk");
    partials_blk_ =
        alloc<double>(static_cast<size_t>(2) * k * bcg_partials_blocks(n_), "partials_blk");
    h_norms_blk_ = pinned_alloc<double>(static_cast<size_t>(2) * k, "solver", "norms_blk");
    block_k_ = k;
}

PcgBlockResult PcgAmgSolver::solve_mixed_block(NativeVCycle& preconditioner, const double* B,
                                               double* X, int k, double tolerance,
                                               int max_iters, cudaStream_t stream,
                                               const double* X0) {
    return solve_block(preconditioner, B, X, k, tolerance, max_iters, stream, X0, false);
}

PcgBlockResult PcgAmgSolver::solve_affine_block(NativeVCycle& preconditioner, const double* B,
                                                double* X, int k, const double* sigma,
                                                double tolerance, int max_iters,
                                                cudaStream_t stream, const double* X0) {
    if (affine_.n_terms == 0) {
        throw std::runtime_error("solve_affine_block: no affine terms set");
    }
    if (k < 1 || k > kMaxStageBlock) throw std::invalid_argument("solve_affine_block: bad k");
    // Columns past k repeat the last, so the deflation state for every held column is defined.
    const int t = affine_.n_terms;
    std::vector<double> padded(static_cast<size_t>(kMaxStageBlock) * t);
    for (int c = 0; c < kMaxStageBlock; ++c) {
        const int from = std::min(c, k - 1);
        for (int j = 0; j < t; ++j) padded[c * t + j] = sigma[from * t + j];
    }
    check_cuda(cudaMemcpyAsync(affine_sigma_.get(), padded.data(), padded.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "copy(sigma)");
    launch_bcg_defl_affine(defl_n_, t, affine_aw_t_.get(), affine_sigma_.get(), defl_aw_.get(),
                           defl_inv_e_.get(), stream);
    check_cuda(cudaGetLastError(), "affine deflation");
    defl_affine_ = true;
    // The copy reads a host vector that dies with this call, so it has to land before returning.
    check_cuda(cudaStreamSynchronize(stream), "sync(sigma)");
    return solve_block(preconditioner, B, X, k, tolerance, max_iters, stream, X0, true);
}

PcgBlockResult PcgAmgSolver::solve_block(NativeVCycle& preconditioner, const double* B,
                                         double* X, int k, double tolerance, int max_iters,
                                         cudaStream_t stream, const double* X0, bool affine) {
    if (max_iters <= 0) throw std::invalid_argument("solve_mixed_block: max_iters must be > 0");
    ensure_block_buffers(k);
    if (!affine) deflation_from_values(stream);
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
    double* const d_rsum = scalars + 5 * k;
    double* const d_beta = scalars + 6 * k;
    double* const d_shift = scalars + 7 * k;
    double* const d_ref_sq = scalars + 8 * k;
    double* const d_converged = scalars + 9 * k;
    double* const d_tolerance = scalars + 10 * k;
    const Deflation defl = deflation();
    double* const h_norm = h_norms_blk_.get();
    double* const h_ref = h_norms_blk_.get() + k;
    double* const partials = partials_blk_.get();
    double* const R = r_blk_.get();
    double* const P = p_blk_.get();
    double* const AP = ap_blk_.get();
    double* const X_int = x_int_blk_.get();
    float* const RF = rf_blk_.get();
    float* const ZF = zf_blk_.get();

    // Reference norms ‖b_c‖ (convergence is measured against b, matching solve_mixed). The
    // squares stay on the device for the per-column freeze; the host copy drives the block's
    // stopping test and the reported residuals.
    launch_bcg_norm2(n_, k, B, partials, d_ref_sq, s);
    // The freeze reads the tolerance from the device: a captured body replays whatever it was
    // launched with, and the graph is shared by every solve at this width whatever its tolerance.
    check_cuda(cudaMemcpyAsync(d_tolerance, &tolerance, sizeof(double), cudaMemcpyHostToDevice, s),
               "block:copy(tolerance)");
    check_cuda(cudaMemcpyAsync(h_ref, d_ref_sq, static_cast<size_t>(k) * sizeof(double),
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
    // No column starts frozen; the mask is recomputed at the end of each iteration body.
    check_cuda(cudaMemsetAsync(d_converged, 0, static_cast<size_t>(k) * sizeof(double), s),
               "block:memset(converged)");

    if (X0 != nullptr) {
        check_cuda(cudaMemcpyAsync(X_int, X0, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
                   "block:copy(X0)");
        apply_operator(k, affine, X_int, AP, s);
        launch_bcg_residual(nk_signed, B, AP, R, s);
    } else {
        check_cuda(cudaMemsetAsync(X_int, 0, nk * sizeof(double), s), "block:memset(X)");
        check_cuda(cudaMemcpyAsync(R, B, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
                   "block:copy(B,R)");
    }

    launch_bcg_deflate_start(n_, k, defl, X_int, R, RF, partials, d_rsum, s);
    preconditioner.apply_block(n_, k, RF, ZF, s);
    launch_bcg_cast_dot_init(n_, k, ZF, R, defl, d_rsum, partials, d_rz, d_shift, P, s);

    auto run_body = [&]() {
        apply_operator(k, affine, P, AP, s);
        launch_bcg_dot(n_, k, P, AP, partials, d_pap, s);
        launch_bcg_alpha(k, d_rz, d_pap, d_converged, d_alpha, d_neg_alpha, s);
        launch_bcg_update_r_norm(n_, k, d_neg_alpha, AP, R, RF, partials, d_norm, s);
        launch_bcg_mark_converged(k, d_norm, d_ref_sq, d_tolerance, d_converged, s);
        check_cuda(cudaMemcpyAsync(h_norm, d_norm, static_cast<size_t>(k) * sizeof(double),
                                   cudaMemcpyDeviceToHost, s),
                   "block:copy(norms)");
        preconditioner.apply_block(n_, k, RF, ZF, s);
        launch_bcg_cast_dot_beta(n_, k, ZF, R, defl, d_converged, d_rsum, partials, d_rz, d_beta,
                                 d_shift, s);
        launch_bcg_update_p(n_, k, d_alpha, d_beta, d_shift, ZF, P, X_int, s);
    };

    const GraphCache::Key key{&preconditioner, preconditioner.generation(), k};
    std::vector<double> rel(k, 0.0);
    // max_iters doubles as the default for a column that never reaches tolerance and as the
    // value a column crossing on the last iteration would be given.
    std::vector<int> col_iters(k, max_iters);
    int result_iters = max_iters;
    for (int it = 1; it <= max_iters; ++it) {
        (affine ? *affine_graph_ : block_graph_).run(s, key, it, run_body);
        check_cuda(cudaStreamSynchronize(s), "block:sync(iter)");
        double worst = 0.0;
        for (int c = 0; c < k; ++c) {
            rel[c] = std::sqrt(h_norm[c]) / norm_ref[c];
            if (rel[c] > worst) worst = rel[c];
            // First crossing only: launch_bcg_mark_converged froze the column on the same
            // expression, so rel[c] holds from here on and would otherwise re-trigger.
            if (col_iters[c] == max_iters && rel[c] <= tolerance) col_iters[c] = it;
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
    result.column_iterations = std::move(col_iters);
    return result;
}

PcgResult PcgAmgSolver::solve_mixed(NativeVCycle& preconditioner, const double* b,
                                    double* x, double tolerance, int max_iters,
                                    cudaStream_t stream, const double* x0) {
    // The solve runs on an internal capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream. If capture is invalidated at runtime the loop falls
    // back to direct execution.
    if (max_iters <= 0) throw std::invalid_argument("solve_mixed: max_iters must be > 0");
    deflation_from_values(stream);
    cudaStream_t s = solve_stream_.get();
    check_cuda(cudaEventRecord(join_event_.get(), stream), "graph:record_in");
    check_cuda(cudaStreamWaitEvent(s, join_event_.get(), 0), "graph:wait_in");
    double* const d_rz = scalars_.get() + 0;
    double* const d_pap = scalars_.get() + 1;
    double* const d_alpha = scalars_.get() + 2;
    double* const d_neg_alpha = scalars_.get() + 3;
    double* const d_norm = scalars_.get() + 4;
    double* const d_rsum = scalars_.get() + 5;
    double* const d_beta = scalars_.get() + 6;
    double* const d_shift = scalars_.get() + 7;
    const Deflation defl = deflation();
    double* const r = r_.get();
    double* const p = p_.get();
    double* const ap = ap_.get();
    double* const x_int = x_int_.get();
    float* const rf = rf_.get();
    float* const zf = zf_.get();
    double* const partials = partials_.get();

    // Every reduction here runs the same k = 1 block kernels solve_mixed_block does, so the two
    // paths sum in the same order: a column that misses tolerance and is retried through this
    // solver comes back with the numbers the block path would have produced. A vendor BLAS dot or
    // norm, whose association order is unspecified, could not stand in.
    auto host_norm = [&](const double* v, const char* what) {
        launch_bcg_norm2(n_, 1, v, partials, d_norm, s);
        check_cuda(
            cudaMemcpyAsync(h_norm_.get(), d_norm, sizeof(double), cudaMemcpyDeviceToHost, s),
            what);
        check_cuda(cudaStreamSynchronize(s), what);
        return std::sqrt(*h_norm_.get());
    };

    // Convergence is measured against ‖b‖, not the warm residual ‖r0‖, so a warm start (x0 != null)
    // still drives to the same 1e-6-of-field criterion instead of stopping early relative to its
    // small initial residual.
    const double norm_ref = host_norm(b, "setup:norm(b)");
    if (norm_ref == 0.0) {
        check_cuda(cudaMemsetAsync(x, 0, static_cast<size_t>(n_) * sizeof(double), s),
                   "memset(x_out)");
        check_cuda(cudaStreamSynchronize(s), "sync(x_out0)");
        return {0, 0.0};
    }
    // The loop works on the solver-owned x_int_ (not the caller's x) so the captured graph contains
    // no per-call pointers and can be replayed across solves; the result is copied out at the end.
    if (x0 != nullptr) {
        check_cuda(cudaMemcpyAsync(x_int, x0, static_cast<size_t>(n_) * sizeof(double),
                                   cudaMemcpyDeviceToDevice, s),
                   "copy(x0,x_int)");
        // r0 = b - A x0
        launch_bcsrmv_f64_block(n_, 1, slots_.ptr.get(), slots_.cols.get(), values_.get(), x_int,
                                ap, s);
        launch_bcg_residual(n_, b, ap, r, s);
        const double norm_r0 = host_norm(r, "setup:norm(r0)");
        if (norm_r0 / norm_ref <= tolerance) {
            check_cuda(cudaMemcpyAsync(x, x_int, static_cast<size_t>(n_) * sizeof(double),
                                       cudaMemcpyDeviceToDevice, s),
                       "copy(x_int,x)");
            check_cuda(cudaStreamSynchronize(s), "sync(x_out)");
            return {0, norm_r0 / norm_ref};
        }
    } else {
        check_cuda(cudaMemsetAsync(x_int, 0, static_cast<size_t>(n_) * sizeof(double), s),
                   "memset(x_int)");
        check_cuda(cudaMemcpyAsync(r, b, static_cast<size_t>(n_) * sizeof(double),
                                   cudaMemcpyDeviceToDevice, s),
                   "copy(b,r)");
    }
    launch_bcg_deflate_start(n_, 1, defl, x_int, r, rf, partials, d_rsum, s);
    preconditioner.apply(n_, rf, zf, s);
    // p0 = z0 = (double)zf + shift directly; the fp64 z vector is never materialized (the loop
    // consumes zf on the fly in cast_dot_beta and update_p).
    launch_bcg_cast_dot_init(n_, 1, zf, r, defl, d_rsum, partials, d_rz, d_shift, p, s);

    // Identical every iteration (fixed buffers updated in place), so it is captured once and
    // replayed; the residual readback is inside the body but the host convergence test stays outside.
    auto run_body = [&]() {
        // p·(Ap) stays a separate pass: folding it into the SpMV epilogue would make the
        // block-wide reduction tree stall the SpMV's memory pipeline.
        launch_bcsrmv_f64_block(n_, 1, slots_.ptr.get(), slots_.cols.get(), values_.get(), p, ap,
                                s);
        launch_bcg_dot(n_, 1, p, ap, partials, d_pap, s);
        launch_bcg_alpha(1, d_rz, d_pap, nullptr, d_alpha, d_neg_alpha, s);
        // r -= α ap; rf = (float)r; d_norm = ‖r‖² (host takes the sqrt), d_rsum = Σr
        launch_bcg_update_r_norm(n_, 1, d_neg_alpha, ap, r, rf, partials, d_norm, s);
        check_cuda(
            cudaMemcpyAsync(h_norm_.get(), d_norm, sizeof(double), cudaMemcpyDeviceToHost, s),
            "copy(norm)");
        preconditioner.apply(n_, rf, zf, s);
        // z = (double)zf + shift; rz' = r·z; beta = rz'/rz; rz <- rz' (no fp64 z vector)
        launch_bcg_cast_dot_beta(n_, 1, zf, r, defl, nullptr, d_rsum, partials, d_rz, d_beta,
                                 d_shift, s);
        // x += α p; p = β p + z
        launch_bcg_update_p(n_, 1, d_alpha, d_beta, d_shift, zf, p, x_int, s);
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
