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
    std::vector<double> relative_residual;  // per-column final relative residual
    // Per-column iteration at which that column reached tolerance and froze, so the count it
    // would have taken solving alone. max_iters for a column that never reached it.
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

    // An operator affine in per-column coefficients over this solver's pattern,
    // A(sigma) = sum_t sigma_t A_t, given as n_terms value arrays in entry order (n_terms, nnz).
    // Kept beside the values update_values sets, which the other solves go on using.
    void set_affine_terms(int n_terms, const double* term_values, cudaStream_t stream);
    int affine_terms() const { return affine_.n_terms; }
    // A block solve in which column c's operator is A(sigma[c]), sigma host (k, n_terms): every
    // column a different matrix, sharing each part's reads and the preconditioner.
    PcgBlockResult solve_affine_block(NativeVCycle& preconditioner, const double* B, double* X,
                                      int k, const double* sigma, double tolerance, int max_iters,
                                      cudaStream_t stream, const double* X0 = nullptr);

private:
    void ensure_block_buffers(int k);
    Deflation deflation() const;
    // Sets the deflation state from values_, unless it already holds it.
    void deflation_from_values(cudaStream_t stream);
    void apply_operator(int k, bool affine, const double* x, double* y, cudaStream_t stream);
    enum class Path { kSingle, kBlock, kAffine };
    PcgBlockResult solve_block(NativeVCycle& preconditioner, const double* B, double* X, int k,
                               double tolerance, int max_iters, cudaStream_t stream,
                               const double* X0, Path path);

    // Serialises every call that touches the solver's buffers, since the bindings release the GIL.
    std::mutex mutex_;
    int n_ = 0;
    int nnz_ = 0;
    // The entry CSR's row pointer, kept to lay new values into slots, and the slot CSR the
    // SpMV streams. values_ is in slot order.
    DeviceBuffer<int> row_ptr_;
    SlotCsr slots_;
    DeviceBuffer<double> values_;
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
    // block_buffers_ counts their allocations and keys both block graphs.
    int block_k_ = 0;
    int block_buffers_ = 0;
    DeviceBuffer<double> r_blk_;
    DeviceBuffer<double> p_blk_;
    DeviceBuffer<double> ap_blk_;
    DeviceBuffer<double> x_int_blk_;
    DeviceBuffer<float> rf_blk_;
    DeviceBuffer<float> zf_blk_;
    // Layout: [rz | pap | alpha | neg_alpha | norm | rsum | beta | shift | ref_sq | converged],
    // each k wide, then the solve's tolerance.
    DeviceBuffer<double> scalars_blk_;
    DeviceBuffer<double> partials_blk_;
    PinnedBuffer<double> h_norms_blk_;  // pinned, k residual norms + k reference norms
    GraphCache block_graph_;
    // The affine operator, its parts' Aw on the deflation rows (n_terms, defl_n_), and the
    // per-column coefficients a block runs with, [column][term]. The graph is its own: the body
    // reads different arrays, and a new set of parts replaces it.
    AffineSlots affine_;
    DeviceBuffer<double> affine_aw_t_;
    DeviceBuffer<double> affine_sigma_;
    GraphCache affine_graph_;
};
