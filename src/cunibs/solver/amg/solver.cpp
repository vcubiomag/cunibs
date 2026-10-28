#include "amg/solver.hpp"

#include "amg/vcycle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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
DeviceBuffer<T> clone(const T* src, std::size_t count, cudaStream_t stream, const char* what) {
    return device_clone<T>(src, count, stream, "solver", what);
}

// A row whose sum is more than this fraction of its l1 norm is a deflation support row, and keeps
// its sum in the difference form. On a grounded conductivity operator these are the ground node's
// neighbours; every other row sums to a rounding residue.
constexpr double kSupportTol = 1e-10;

// How far a column's fp32 recurrence may fall below the true residual it last restarted from before
// it is checked against the fp64 operator. The recurrence tracks the true residual to a few percent
// over six decades, so at the default tolerance a solve is one stage with one check. A tighter
// tolerance is reached in stages, each restarted from the true residual.
//
// Restarting rather than replacing r and keeping p (the reliable update of van der Vorst and Ye) is
// deliberate. The reliable update matches the fp64 iteration count at 1e-9 and 1e-12 where restarts
// cost 5-30% more, but it needs checkpoints every few decades, which cost a fp64 SpMV each at the
// default tolerance too, and it diverges once a small system's fp32 Krylov space is exhausted.
constexpr double kStageDrop = 1e-6;

// Restarts a column may take before its true residual is reported as it stands.
constexpr int kMaxRestarts = 8;

constexpr double kUnfrozen = 0.0;

}  // namespace

// Every member is a move-only owner, so a throw part way through releases what was already
// taken and the destructor is the implicit one.
PcgAmgSolver::PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx,
                           const double* values, cudaStream_t stream)
    : n_(n), nnz_(nnz) {
    if (n_ <= 0) throw std::invalid_argument("PcgAmgSolver: n must be positive");
    if (nnz_ < 0) throw std::invalid_argument("PcgAmgSolver: nnz must not be negative");

    row_ptr_ = clone(row_ptr, static_cast<size_t>(n_) + 1, stream, "row_ptr");
    slots_ = build_slot_csr(n_, row_ptr, col_idx, stream, "solver");
    values_ = alloc<double>(static_cast<size_t>(slots_.n_slots), "values");
    launch_slot_scatter(n_, row_ptr_.get(), slots_.ptr.get(), slots_.cols.get(), values,
                        values_.get(), stream);
    values32_ = alloc<float>(static_cast<size_t>(slots_.n_slots), "values32");
    launch_diff_values(n_, slots_.ptr.get(), slots_.cols.get(), values_.get(), kSupportTol,
                       values32_.get(), stream);

    cudaStream_t solve_stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&solve_stream, cudaStreamNonBlocking),
               "create(solve_stream)");
    solve_stream_.reset(solve_stream);
    cudaEvent_t event = nullptr;
    check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create(join_event)");
    join_event_.reset(event);

    // The deflation support: rows whose sum is more than a rounding residue of their l1 norm. On a
    // grounded conductivity operator that is the ground node's neighbours. Chosen once, from the
    // values this was built with: it is a property of the pattern, and update_values keeps it.
    // Everything here is on the caller's stream, which a non-blocking solve_stream_ does not
    // wait for, so it is drained before any solve can start.
    {
        const ScratchBuffer<int> rows =
            scratch_alloc<int>(static_cast<size_t>(n_), stream, "solver", "deflation rows");
        defl_n_ = select_deflation_rows(n_, slots_.ptr.get(), values_.get(), kSupportTol,
                                        rows.get(), stream);
        defl_rows_ = alloc<int>(std::max(defl_n_, 1), "deflation rows");
        check_cuda(cudaMemcpyAsync(defl_rows_.get(), rows.get(), defl_n_ * sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream),
                   "copy(deflation rows)");
    }
    defl_aw_ = alloc<double>(std::max<size_t>(defl_n_, 1) * kMaxStageBlock, "deflation aw");
    defl_inv_e_ = alloc<double>(kMaxStageBlock, "deflation inv_e");
    defl_affine_ = true;
    deflation_from_values(stream);
    check_cuda(cudaStreamSynchronize(stream), "sync(deflation setup)");
}

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
    const std::lock_guard lock(mutex_);
    launch_slot_scatter(n_, row_ptr_.get(), slots_.ptr.get(), slots_.cols.get(), values,
                        values_.get(), stream);
    launch_diff_values(n_, slots_.ptr.get(), slots_.cols.get(), values_.get(), kSupportTol,
                       values32_.get(), stream);
    check_cuda(cudaGetLastError(), "scatter(values)");
    // A solve then waits on its caller's stream, which is this one, before touching either.
    defl_affine_ = true;
    deflation_from_values(stream);
}

// The solver's pattern and deflation rows are fixed at construction, so the parts built here stay
// valid against it for both objects' lifetimes.
AffineTerms::AffineTerms(const PcgAmgSolver& solver, int n_terms, const double* term_values,
                         cudaStream_t stream)
    : solver_(&solver),
      n_(solver.n_),
      slots_(build_affine_slots(solver.n_, solver.nnz_, solver.row_ptr_.get(),
                                solver.slots_.ptr.get(), solver.slots_.cols.get(), term_values,
                                n_terms, stream, "affine terms")) {
    vals32_ = alloc<float>(static_cast<size_t>(slots_.n_slots), "affine vals32");
    launch_diff_values(slots_.n_segs, slots_.seg_start.get(), slots_.cols.get(),
                       slots_.vals.get(), kSupportTol, vals32_.get(), stream);
    aw_t_ = alloc<double>(static_cast<size_t>(n_terms) * std::max(solver.defl_n_, 1),
                          "affine aw_t");
    launch_term_row_sums(solver.defl_n_, solver.defl_rows_.get(), solver.row_ptr_.get(),
                         term_values, solver.nnz_, n_terms, aw_t_.get(), stream);
    sigma_ = alloc<double>(static_cast<size_t>(kMaxStageBlock) * n_terms, "affine sigma");
    sigma32_ = alloc<float>(static_cast<size_t>(kMaxStageBlock) * n_terms, "affine sigma32");
    check_cuda(cudaGetLastError(), "affine terms");
    check_cuda(cudaStreamSynchronize(stream), "sync(affine terms)");
}

void AffineTerms::values(const double* sigma, double* out, cudaStream_t stream) const {
    const int t = slots_.n_terms;
    // A copy from pageable memory has read sigma by the time it returns, so the caller's array
    // need not outlive the call.
    const ScratchBuffer<double> d_sigma =
        scratch_alloc<double>(static_cast<size_t>(t), stream, "affine terms", "sigma");
    check_cuda(cudaMemcpyAsync(d_sigma.get(), sigma, t * sizeof(double), cudaMemcpyHostToDevice,
                               stream),
               "copy(values sigma)");
    const PcgAmgSolver& s = *solver_;
    launch_affine_values(s.n_, s.row_ptr_.get(), s.slots_.ptr.get(), s.slots_.cols.get(), slots_,
                         d_sigma.get(), out, stream);
    check_cuda(cudaGetLastError(), "affine values");
}

void AffineTerms::matmul(int term, const double* x, double* y, int m, cudaStream_t stream) const {
    if (term < 0 || term >= slots_.n_terms) {
        throw std::invalid_argument("AffineTerms::matmul: no such term");
    }
    launch_affine_term_matmul(n_, slots_, term, x, y, m, stream);
    check_cuda(cudaGetLastError(), "affine term matmul");
}

void PcgAmgSolver::export_csr(int* row_ptr, int* col_idx, double* values, cudaStream_t stream) {
    const std::lock_guard lock(mutex_);
    check_cuda(cudaMemcpyAsync(row_ptr, row_ptr_.get(),
                               (static_cast<size_t>(n_) + 1) * sizeof(int),
                               cudaMemcpyDeviceToDevice, stream),
               "copy(export row_ptr)");
    if (col_idx != nullptr || values != nullptr) {
        launch_slot_gather(n_, row_ptr_.get(), slots_.ptr.get(), slots_.cols.get(), values_.get(),
                           col_idx, values, stream);
    }
    check_cuda(cudaGetLastError(), "export csr");
}

void PcgAmgSolver::apply_operator(int k, const AffineTerms* terms, const double* x, double* y,
                                  cudaStream_t stream) {
    if (terms != nullptr) {
        const AffineSlots& a = terms->slots_;
        const AffineOperator op{a.n_terms,          a.order.get(),     a.seg_ptr.get(),
                                a.seg_start.get(),  a.seg_term.get(),  a.cols.get(),
                                a.vals.get()};
        launch_affine_spmv_block(n_, k, op, terms->sigma_.get(), x, y, stream);
    } else {
        launch_bcsrmv_f64_block(n_, k, slots_.ptr.get(), slots_.cols.get(), values_.get(), x, y,
                                stream);
    }
}

void PcgAmgSolver::apply_operator(int k, const AffineTerms* terms, const float* x, float* y,
                                  cudaStream_t stream) {
    if (terms != nullptr) {
        const AffineSlots& a = terms->slots_;
        const AffineOperator op{a.n_terms,          a.order.get(),     a.seg_ptr.get(),
                                a.seg_start.get(),  a.seg_term.get(),  a.cols.get(),
                                a.vals.get()};
        launch_affine_diff_block(n_, k, op, terms->vals32_.get(), terms->sigma32_.get(), x, y,
                                 stream);
    } else {
        launch_bcsrmv_diff_block(n_, k, slots_.ptr.get(), slots_.cols.get(), values32_.get(), x,
                                 y, stream);
    }
}

void PcgAmgSolver::ensure_block_buffers(int k) {
    if (block_k_ >= k) return;
    // Drop the width before reallocating: if an allocation below throws, the old buffers are
    // already released and a retry at the same k must not take the early-out above.
    block_k_ = 0;
    const size_t nk = static_cast<size_t>(n_) * k;
    r_blk_ = alloc<float>(nk, "R_blk");
    p_blk_ = alloc<float>(nk, "P_blk");
    ap_blk_ = alloc<float>(nk, "AP_blk");
    zf_blk_ = alloc<float>(nk, "ZF_blk");
    x_int_blk_ = alloc<double>(nk, "X_int_blk");
    ap64_blk_ = alloc<double>(nk, "AP64_blk");
    scalars_blk_ = alloc<double>(static_cast<size_t>(12) * k + 1, "scalars_blk");
    partials_blk_ =
        alloc<double>(static_cast<size_t>(2) * k * bcg_partials_blocks(n_), "partials_blk");
    h_norms_blk_ = pinned_alloc<double>(static_cast<size_t>(3) * k, "solver", "norms_blk");
    block_k_ = k;
    ++block_buffers_;
}

PcgBlockResult PcgAmgSolver::solve_mixed_block(NativeVCycle& preconditioner, const double* B,
                                               double* X, int k, double tolerance,
                                               int max_iters, cudaStream_t stream,
                                               const double* X0) {
    const std::scoped_lock lock(mutex_, preconditioner.mutex());
    return solve_block(preconditioner, B, X, k, tolerance, max_iters, stream, X0, Path::kBlock);
}

PcgBlockResult PcgAmgSolver::solve_affine_block(NativeVCycle& preconditioner, AffineTerms& terms,
                                                const double* B, double* X, int k,
                                                const double* sigma, double tolerance,
                                                int max_iters, cudaStream_t stream,
                                                const double* X0) {
    const std::scoped_lock lock(mutex_, preconditioner.mutex());
    if (terms.solver_ != this) {
        throw std::invalid_argument("solve_affine_block: the terms were built for another solver");
    }
    if (!is_block_width(k)) {
        throw std::invalid_argument("solve_affine_block supports k in {1, 2, 4, 8}");
    }
    // Columns past k repeat the last, so the deflation state for every held column is defined.
    const int t = terms.n_terms();
    std::vector<double> padded(static_cast<size_t>(kMaxStageBlock) * t);
    for (int c = 0; c < kMaxStageBlock; ++c) {
        const int from = std::min(c, k - 1);
        for (int j = 0; j < t; ++j) padded[c * t + j] = sigma[from * t + j];
    }
    check_cuda(cudaMemcpyAsync(terms.sigma_.get(), padded.data(), padded.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "copy(sigma)");
    // Pageable, so the copy has read the host array by the time it returns.
    const std::vector<float> padded32(padded.begin(), padded.end());
    check_cuda(cudaMemcpyAsync(terms.sigma32_.get(), padded32.data(),
                               padded32.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
               "copy(sigma32)");
    launch_bcg_defl_affine(defl_n_, t, terms.aw_t_.get(), terms.sigma_.get(), defl_aw_.get(),
                           defl_inv_e_.get(), stream);
    check_cuda(cudaGetLastError(), "affine deflation");
    defl_affine_ = true;
    return solve_block(preconditioner, B, X, k, tolerance, max_iters, stream, X0, Path::kAffine,
                       &terms);
}

PcgBlockResult PcgAmgSolver::solve_block(NativeVCycle& preconditioner, const double* B,
                                         double* X, int k, double tolerance, int max_iters,
                                         cudaStream_t stream, const double* X0, Path path,
                                         AffineTerms* terms) {
    if (max_iters <= 0) throw std::invalid_argument("solve_mixed_block: max_iters must be > 0");
    if (!is_block_width(k)) {
        throw std::invalid_argument("solve_mixed_block supports k in {1, 2, 4, 8}");
    }
    const bool affine = path == Path::kAffine;
    const bool single = path == Path::kSingle;
    ensure_block_buffers(k);
    if (!affine) deflation_from_values(stream);
    cudaStream_t s = solve_stream_.get();
    check_cuda(cudaEventRecord(join_event_.get(), stream), "block:record_in");
    check_cuda(cudaStreamWaitEvent(s, join_event_.get(), 0), "block:wait_in");

    const size_t nk = static_cast<size_t>(n_) * k;
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
    double* const d_true_sq = scalars + 10 * k;
    double* const d_write = scalars + 11 * k;
    double* const d_tolerance = scalars + 12 * k;
    const Deflation defl = deflation();
    double* const h_norm = h_norms_blk_.get();
    double* const h_ref = h_norms_blk_.get() + k;
    double* const h_true = h_norms_blk_.get() + 2 * k;
    double* const partials = partials_blk_.get();
    float* const R = r_blk_.get();
    float* const P = p_blk_.get();
    float* const AP = ap_blk_.get();
    float* const ZF = zf_blk_.get();
    double* const X_int = x_int_blk_.get();
    double* const AP64 = ap64_blk_.get();

    // Reference norms ‖b_c‖: convergence is measured against b, not the warm residual, so a warm
    // start still drives to the same criterion. The squares stay on the device for the per-column
    // freeze; the host copy drives the block's stopping test and the reported residuals.
    launch_bcg_norm2(n_, k, B, partials, d_ref_sq, s);
    // The freeze reads the tolerance from the device: a captured body replays whatever it was
    // launched with, and the graph is shared by every solve at this width whatever its tolerance.
    check_cuda(cudaMemcpyAsync(d_tolerance, &tolerance, sizeof(double), cudaMemcpyHostToDevice, s),
               "block:copy(tolerance)");
    check_cuda(cudaMemsetAsync(d_write, 0, static_cast<size_t>(k) * sizeof(double), s),
               "block:memset(write)");
    check_cuda(cudaMemcpyAsync(h_ref, d_ref_sq, static_cast<size_t>(k) * sizeof(double),
                               cudaMemcpyDeviceToHost, s),
               "block:copy(ref_norms)");
    check_cuda(cudaStreamSynchronize(s), "block:sync(ref_norms)");
    // A zero column's solution is exactly zero. It starts frozen at x = 0 with an infinite
    // reference, so the freeze test keeps it there, and the other columns run as they would alone.
    std::vector<double> norm_ref(k);
    std::vector<double> frozen(k, 0.0);
    int n_zero = 0;
    for (int c = 0; c < k; ++c) {
        norm_ref[c] = std::sqrt(h_ref[c]);
        if (norm_ref[c] != 0.0) continue;
        frozen[c] = 1.0;
        h_ref[c] = std::numeric_limits<double>::infinity();
        ++n_zero;
    }
    if (n_zero == k) {
        check_cuda(cudaMemsetAsync(X, 0, nk * sizeof(double), s), "memset(x_out)");
        check_cuda(cudaStreamSynchronize(s), "sync(x_out0)");
        return {0, std::vector<double>(k, 0.0), std::vector<int>(k, 0)};
    }
    check_cuda(cudaMemcpyAsync(d_converged, frozen.data(), static_cast<size_t>(k) * sizeof(double),
                               cudaMemcpyHostToDevice, s),
               "block:copy(converged)");
    if (n_zero) {
        check_cuda(cudaMemcpyAsync(d_ref_sq, h_ref, static_cast<size_t>(k) * sizeof(double),
                                   cudaMemcpyHostToDevice, s),
                   "block:copy(ref_norms)");
    }

    if (X0 != nullptr) {
        check_cuda(cudaMemcpyAsync(X_int, X0, nk * sizeof(double), cudaMemcpyDeviceToDevice, s),
                   "block:copy(X0)");
        for (int c = 0; c < k; ++c) {
            if (!frozen[c]) continue;
            check_cuda(cudaMemset2DAsync(X_int + c, k * sizeof(double), 0, sizeof(double), n_, s),
                       "block:zero(X0 column)");
        }
        apply_operator(k, terms, X_int, AP64, s);
        launch_bcg_residual(n_, k, B, AP64, nullptr, R, partials, d_true_sq, s);
        check_cuda(cudaMemcpyAsync(h_true, d_true_sq, static_cast<size_t>(k) * sizeof(double),
                                   cudaMemcpyDeviceToHost, s),
                   "copy(norm r0)");
        check_cuda(cudaStreamSynchronize(s), "sync(norm r0)");
        if (single) {
            const double rel0 = std::sqrt(*h_true) / norm_ref[0];
            if (rel0 <= tolerance) {
                check_cuda(cudaMemcpyAsync(X, X_int, nk * sizeof(double),
                                           cudaMemcpyDeviceToDevice, s),
                           "copy(x_int,x)");
                check_cuda(cudaStreamSynchronize(s), "sync(x_out)");
                return {0, {rel0}, {0}};
            }
        }
    } else {
        check_cuda(cudaMemsetAsync(X_int, 0, nk * sizeof(double), s), "block:memset(X)");
        launch_bcg_residual(n_, k, B, nullptr, nullptr, R, partials, d_true_sq, s);
    }

    launch_bcg_deflate_start(n_, k, defl, X_int, R, partials, d_rsum, s);
    preconditioner.apply_block(n_, k, R, ZF, s);
    launch_bcg_cast_dot_init(n_, k, ZF, R, defl, d_rsum, partials, d_rz, d_shift, P, s);

    auto run_body = [&]() {
        apply_operator(k, terms, P, AP, s);
        launch_bcg_dot(n_, k, P, AP, partials, d_pap, s);
        launch_bcg_alpha(k, d_rz, d_pap, d_converged, d_alpha, d_neg_alpha, s);
        launch_bcg_update_r_norm(n_, k, d_neg_alpha, AP, R, partials, d_norm, s);
        launch_bcg_mark_converged(k, d_norm, d_ref_sq, d_tolerance, d_converged, s);
        check_cuda(cudaMemcpyAsync(h_norm, d_norm, static_cast<size_t>(k) * sizeof(double),
                                   cudaMemcpyDeviceToHost, s),
                   "block:copy(norms)");
        preconditioner.apply_block(n_, k, R, ZF, s);
        launch_bcg_cast_dot_beta(n_, k, ZF, R, defl, d_converged, d_rsum, partials, d_rz, d_beta,
                                 d_shift, s);
        launch_bcg_update_p(n_, k, d_alpha, d_beta, d_shift, ZF, P, X_int, s);
    };

    const GraphCache::Key key{&preconditioner, preconditioner.generation(), k, block_buffers_};
    std::vector<double> rel(k, 0.0);
    // max_iters for a column that never reaches tolerance.
    std::vector<int> col_iters(k, max_iters);
    // Whether the column's convergence has been settled: zero columns at the start, the rest once
    // their true residual is checked.
    std::vector<bool> done(k, false);
    std::vector<int> restarts(k, 0);
    // Iterations each column has run, which excludes any it spent frozen awaiting its check.
    std::vector<int> ran(k, 0);
    std::vector<bool> waiting(k, false);
    // The true relative residual each column's current stage started from.
    std::vector<double> stage(k, 1.0);
    for (int c = 0; c < k; ++c) {
        if (X0 != nullptr && !frozen[c]) stage[c] = std::sqrt(h_true[c]) / norm_ref[c];
        if (!frozen[c]) continue;
        col_iters[c] = 0;
        done[c] = true;
    }
    const auto checkpoint = [&](int c) { return std::max(tolerance, kStageDrop * stage[c]); };
    std::vector<double> write(k);
    int result_iters = max_iters;
    for (int it = 1; it <= max_iters; ++it) {
        GraphCache& graph = affine ? terms->graph_ : single ? single_graph_ : block_graph_;
        graph.run(s, key, it, run_body);
        check_cuda(cudaStreamSynchronize(s), "block:sync(iter)");
        // A column whose recurrence met tolerance is frozen on the device, so its x cannot move
        // and its check can wait for the block's other columns: a block then pays one fp64 SpMV
        // for columns that converge a few iterations apart. A column at an intermediate
        // checkpoint is still running and is checked at once, so that its arithmetic never
        // depends on who it shares the block with.
        bool check_now = false, all_waiting = true;
        for (int c = 0; c < k; ++c) {
            if (done[c]) continue;
            if (!waiting[c]) ++ran[c];
            rel[c] = std::sqrt(h_norm[c]) / norm_ref[c];
            waiting[c] = rel[c] <= tolerance;
            check_now = check_now || (!waiting[c] && rel[c] <= checkpoint(c));
            all_waiting = all_waiting && waiting[c];
        }
        if (check_now || all_waiting) {
            // A column whose fp32 recurrence reaches its checkpoint is checked on the fp64
            // operator. It is done if that meets tolerance; otherwise it restarts from the true
            // residual, the same start a solve makes, and its next stage is measured from there.
            apply_operator(k, terms, X_int, AP64, s);
            launch_bcg_residual(n_, k, B, AP64, d_write, R, partials, d_true_sq, s);
            check_cuda(cudaMemcpyAsync(h_true, d_true_sq, static_cast<size_t>(k) * sizeof(double),
                                       cudaMemcpyDeviceToHost, s),
                       "block:copy(true norms)");
            check_cuda(cudaStreamSynchronize(s), "block:sync(true norms)");
            bool restart = false;
            for (int c = 0; c < k; ++c) {
                write[c] = 0.0;
                if (done[c] || rel[c] > checkpoint(c)) continue;
                const double true_rel = std::sqrt(h_true[c]) / norm_ref[c];
                if (true_rel <= tolerance || restarts[c] == kMaxRestarts) {
                    rel[c] = true_rel;
                    col_iters[c] = true_rel <= tolerance ? ran[c] : max_iters;
                    done[c] = true;
                    continue;
                }
                write[c] = 1.0;
                ++restarts[c];
                stage[c] = true_rel;
                waiting[c] = false;
                restart = true;
            }
            if (restart) {
                check_cuda(cudaMemcpyAsync(d_write, write.data(),
                                           static_cast<size_t>(k) * sizeof(double),
                                           cudaMemcpyHostToDevice, s),
                           "block:copy(write)");
                launch_bcg_residual(n_, k, B, AP64, d_write, R, partials, d_true_sq, s);
                // zf is dead between iterations, so every column's may be overwritten here.
                preconditioner.apply_block(n_, k, R, ZF, s);
                launch_bcg_restart(n_, k, ZF, R, defl, d_write, partials, d_rsum, d_rz, d_shift,
                                   P, s);
                for (int c = 0; c < k; ++c) {
                    if (write[c] == 0.0) continue;
                    check_cuda(cudaMemcpyAsync(d_converged + c, &kUnfrozen, sizeof(double),
                                               cudaMemcpyHostToDevice, s),
                               "block:copy(unfreeze)");
                }
                check_cuda(cudaMemsetAsync(d_write, 0, static_cast<size_t>(k) * sizeof(double), s),
                           "block:memset(write)");
            }
        }
        if (std::all_of(done.begin(), done.end(), [](bool d) { return d; })) {
            result_iters = it;
            break;
        }
    }
    if (!std::all_of(done.begin(), done.end(), [](bool d) { return d; })) {
        // A column out of iterations reports its fp64 residual, never the fp32 recurrence, which
        // can drift arbitrarily far below it.
        apply_operator(k, terms, X_int, AP64, s);
        launch_bcg_residual(n_, k, B, AP64, d_write, R, partials, d_true_sq, s);
        check_cuda(cudaMemcpyAsync(h_true, d_true_sq, static_cast<size_t>(k) * sizeof(double),
                                   cudaMemcpyDeviceToHost, s),
                   "block:copy(final norms)");
        check_cuda(cudaStreamSynchronize(s), "block:sync(final norms)");
        for (int c = 0; c < k; ++c) {
            if (done[c]) continue;
            rel[c] = std::sqrt(h_true[c]) / norm_ref[c];
            if (rel[c] <= tolerance) col_iters[c] = ran[c];
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
    const std::scoped_lock lock(mutex_, preconditioner.mutex());
    const PcgBlockResult r =
        solve_block(preconditioner, b, x, 1, tolerance, max_iters, stream, x0, Path::kSingle);
    return {r.iterations, r.relative_residual[0]};
}
