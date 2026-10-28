#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "amg/cg_kernels.hpp"
#include "amg/slot_csr.hpp"
#include "core/common.hpp"

// Only ever held by reference or pointer here; solver.cpp includes vcycle.hpp for the calls.
class NativeVCycle;

struct PcgResult {
    int iterations = 0;
    double relative_residual = 0.0;
};

struct PcgBlockResult {
    int iterations = 0;                     // iterations the block ran, i.e. its slowest column
    // Per-column final relative residual, of the fp64 operator.
    std::vector<double> relative_residual;
    // Per-column iterations run until that column met tolerance, not counting any it spent frozen
    // while the block's other columns caught up, so the count it would have taken solving alone.
    // max_iters for a column that never reached it.
    std::vector<int> column_iterations;
};

// A captured CG iteration body, replayed until the preconditioner it was captured against
// changes. The body only touches solver-owned buffers, so a graph captured in an earlier
// solve stays valid across solves; `key` is what identifies "the same body": the
// preconditioner's address and generation, plus the block width.
class GraphCache {
public:
    struct Key {
        const NativeVCycle* precond = nullptr;
        int generation = 0;
        int k = 0;
        // The solver's block buffers the body reads, which ensure_block_buffers replaces.
        int buffers = 0;
        bool operator==(const Key&) const = default;
    };

    GraphCache() = default;
    ~GraphCache() { clear(); }
    GraphCache(const GraphCache&) = delete;
    GraphCache& operator=(const GraphCache&) = delete;

    // Runs `body` once on `stream`, replaying the cached graph when there is one.
    //
    // `iteration` is the caller's 1-based loop counter: the first iteration always runs
    // uncaptured so it can warm allocation pools and lazy buffers, which capture forbids.
    // Capture is attempted on the second, and abandoned for the rest of the solve if it
    // fails: an invalidated capture records without executing, which PcgAmgSolver has to
    // survive rather than propagate.
    template <typename Body>
    void run(cudaStream_t stream, const Key& key, int iteration, Body&& body);

    void clear() noexcept {
        if (exec_) cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }

private:
    // Records `body` into a fresh executable graph. Returns false, with the stream out of capture
    // mode and no error left pending, if anything along the way failed.
    template <typename Body>
    bool capture(cudaStream_t stream, Body& body);

    cudaGraphExec_t exec_ = nullptr;
    Key key_{};
    bool capture_failed_ = false;
};

template <typename Body>
bool GraphCache::capture(cudaStream_t stream, Body& body) {
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    bool recorded = true;
    try {
        body();
    } catch (...) {
        // A call inside an invalidated capture reports the invalidation as its own failure. The
        // capture still has to be ended, or the stream stays in capture mode for good.
        recorded = false;
    }
    cudaGraph_t graph = nullptr;
    const bool ended = cudaStreamEndCapture(stream, &graph) == cudaSuccess && graph != nullptr;
    const bool ok = recorded && ended && cudaGraphInstantiate(&exec_, graph, 0) == cudaSuccess;
    if (graph) cudaGraphDestroy(graph);
    if (!ok) {
        exec_ = nullptr;
        (void)cudaGetLastError();
    }
    return ok;
}

template <typename Body>
void GraphCache::run(cudaStream_t stream, const Key& key, int iteration, Body&& body) {
    // A failed capture is given up on for the rest of that solve but retried by the next one,
    // since whatever invalidated it is usually not a property of this body.
    if (iteration == 1) capture_failed_ = false;
    if (exec_ != nullptr && key_ == key) {
        check_cuda(cudaGraphLaunch(exec_, stream), "solver", "graph_launch");
        return;
    }
    clear();
    // Iteration 1 warms pools and lazy buffers so capture sees no allocation.
    if (iteration == 1 || capture_failed_) {
        body();
        return;
    }
    if (!capture(stream, body)) {
        // The capture recorded without executing, so this iteration still has to run for real.
        capture_failed_ = true;
        body();
        return;
    }
    key_ = key;
    check_cuda(cudaGraphLaunch(exec_, stream), "solver", "graph_launch_first");
}

class PcgAmgSolver;

// An operator affine in per-column coefficients over one solver's pattern,
// A(sigma) = sum_t sigma_t A_t, for PcgAmgSolver::solve_affine_block. A solver can serve any
// number of them; each holds its own parts, the coefficients a block runs with, and its captured
// CG graph, and only the solver it was built on can solve with it.
class AffineTerms {
public:
    // term_values is (n_terms, nnz) in the solver's entry order. Runs on `stream` and
    // synchronises it, so the caller may drop term_values on return.
    AffineTerms(const PcgAmgSolver& solver, int n_terms, const double* term_values,
                cudaStream_t stream);

    AffineTerms(const AffineTerms&) = delete;
    AffineTerms& operator=(const AffineTerms&) = delete;

    int n_terms() const { return slots_.n_terms; }
    // Rows of the solver's matrix.
    int n() const { return n_; }
    // out (nnz, entry order) = sum_t sigma[t] A_t, for host sigma (n_terms).
    void values(const double* sigma, double* out, cudaStream_t stream) const;
    // y (n, m) = A_term x for row-major (n, m) x, each row summed over the part's own nonzeros in
    // column order.
    void matmul(int term, const double* x, double* y, int m, cudaStream_t stream) const;

private:
    friend class PcgAmgSolver;
    const PcgAmgSolver* solver_;
    int n_;
    AffineSlots slots_;
    // slots_.vals in the fp32 difference form the CG iterates with (launch_diff_values).
    DeviceBuffer<float> vals32_;
    // Each part's Aw on the solver's deflation rows (n_terms, defl_n_).
    DeviceBuffer<double> aw_t_;
    // The per-column coefficients a block runs with, [column][term], and their fp32 copy.
    DeviceBuffer<double> sigma_;
    DeviceBuffer<float> sigma32_;
    GraphCache graph_;
};

class PcgAmgSolver {
public:
    // Setup runs on `stream` and synchronises it, so the caller may drop its arrays on return.
    PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx, const double* values,
                 cudaStream_t stream);

    PcgAmgSolver(const PcgAmgSolver&) = delete;
    PcgAmgSolver& operator=(const PcgAmgSolver&) = delete;

    // Dimensions of the matrix this was built on, so callers can validate their arrays.
    int n() const { return n_; }
    int nnz() const { return nnz_; }

    void update_values(const double* values, cudaStream_t stream);
    // The block solve at k = 1, bit for bit, except that a zero b returns a zero x and an x0
    // already within tolerance is returned as it is.
    PcgResult solve_mixed(NativeVCycle& preconditioner, const double* b, double* x,
                          double tolerance, int max_iters, cudaStream_t stream,
                          const double* x0 = nullptr);
    // Block solve: k independent CG chains in lockstep over row-major (n, k) operands, sharing
    // every stiffness-matrix read (block SpMV + block V-cycle). The block runs until its worst
    // column reaches tolerance, but each column freezes at its own, so a column's answer does
    // not depend on who it was batched with, nor does its cost. Per-column residuals and
    // iteration counts are reported so callers can fall back per column. k in {1, 2, 4, 8}.
    PcgBlockResult solve_mixed_block(NativeVCycle& preconditioner, const double* B, double* X,
                                     int k, double tolerance, int max_iters,
                                     cudaStream_t stream, const double* X0 = nullptr);

    // A block solve in which column c's operator is A(sigma[c]) from `terms`, sigma host
    // (k, n_terms): every column a different matrix, sharing each part's reads and the
    // preconditioner. The values update_values sets are left as they are.
    PcgBlockResult solve_affine_block(NativeVCycle& preconditioner, AffineTerms& terms,
                                      const double* B, double* X, int k, const double* sigma,
                                      double tolerance, int max_iters, cudaStream_t stream,
                                      const double* X0 = nullptr);

    // The matrix as the entry CSR it was built from: row_ptr (n + 1), and col_idx and values
    // (nnz each), either of which may be null. Runs on `stream`.
    void export_csr(int* row_ptr, int* col_idx, double* values, cudaStream_t stream);

private:
    friend class AffineTerms;
    void ensure_block_buffers(int k);
    Deflation deflation() const;
    // Sets the deflation state from values_, unless it already holds it.
    void deflation_from_values(cudaStream_t stream);
    // `terms` is the affine operator to apply, or null for values_: in fp64, and in the fp32
    // difference form the iteration uses.
    void apply_operator(int k, const AffineTerms* terms, const double* x, double* y,
                        cudaStream_t stream);
    void apply_operator(int k, const AffineTerms* terms, const float* x, float* y,
                        cudaStream_t stream);
    enum class Path { kSingle, kBlock, kAffine };
    PcgBlockResult solve_block(NativeVCycle& preconditioner, const double* B, double* X, int k,
                               double tolerance, int max_iters, cudaStream_t stream,
                               const double* X0, Path path, AffineTerms* terms = nullptr);

    // Serialises every call that touches the solver's buffers, since the bindings release the GIL.
    std::mutex mutex_;
    int n_ = 0;
    int nnz_ = 0;
    // The entry CSR's row pointer, kept to lay new values into slots, and the slot CSR the
    // SpMV streams. values_ is in slot order.
    DeviceBuffer<int> row_ptr_;
    SlotCsr slots_;
    DeviceBuffer<double> values_;
    // values_ in the fp32 difference form the CG iterates with (launch_diff_values).
    DeviceBuffer<float> values32_;
    // Deflation of the constant vector (block_cg.cu): the rows where A 1 is nonzero, chosen once
    // from the pattern's values, then Aw on them per column and 1/E per column. It holds either
    // values_' state or one affine block's; the flag says which.
    int defl_n_ = 0;
    DeviceBuffer<int> defl_rows_;
    DeviceBuffer<double> defl_aw_;
    DeviceBuffer<double> defl_inv_e_;
    bool defl_affine_ = false;
    // Solves run on this internal, capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream; b/x are handed off via join_event_. The iteration body
    // only touches solver-owned buffers (the internal x, not the caller's), so a captured graph
    // is reused across solves as long as its key holds.
    CudaStream solve_stream_;
    CudaEvent join_event_;
    // The single path keeps its own graph, so a k = 1 retry between wider block solves does not
    // evict theirs.
    GraphCache single_graph_;
    // Block-solve state: (n, k) row-major work buffers, lazily sized to the largest k seen.
    // block_buffers_ counts their allocations and is part of every graph's key.
    int block_k_ = 0;
    int block_buffers_ = 0;
    DeviceBuffer<float> r_blk_;
    DeviceBuffer<float> p_blk_;
    DeviceBuffer<float> ap_blk_;
    DeviceBuffer<float> zf_blk_;
    DeviceBuffer<double> x_int_blk_;
    // A X in fp64, for the residuals convergence is decided on.
    DeviceBuffer<double> ap64_blk_;
    // Layout: [rz | pap | alpha | neg_alpha | norm | rsum | beta | shift | ref_sq | converged |
    // true_sq | write], each k wide, then the solve's tolerance.
    DeviceBuffer<double> scalars_blk_;
    DeviceBuffer<double> partials_blk_;
    // pinned: k recurrence norms, k reference norms, k true-residual norms
    PinnedBuffer<double> h_norms_blk_;
    GraphCache block_graph_;
};
