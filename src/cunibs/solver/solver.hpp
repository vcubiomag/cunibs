#pragma once

#include <cstdint>
#include <vector>

#include "common.hpp"

// Only ever held by reference or pointer here; solver.cpp includes vcycle.hpp for the calls.
class NativeVCycle;

struct PcgResult {
    int iterations = 0;
    double relative_residual = 0.0;
};

struct PcgBlockResult {
    int iterations = 0;                     // iterations the block ran, i.e. its slowest column
    std::vector<double> relative_residual;  // per-column final relative residual
};

// A captured CG iteration body, replayed until the preconditioner it was captured against
// changes. The body only touches solver-owned buffers, so a graph captured in an earlier
// solve stays valid across solves; `key` is what identifies "the same body" -- the
// preconditioner's address and generation, plus the block width for the k-RHS solve.
class GraphCache {
public:
    struct Key {
        const NativeVCycle* precond = nullptr;
        int generation = 0;
        int k = 0;
        bool operator==(const Key&) const = default;
    };

    ~GraphCache() { reset(); }
    GraphCache(const GraphCache&) = delete;
    GraphCache& operator=(const GraphCache&) = delete;
    GraphCache() = default;

    // Runs `body` once on `stream`, replaying the cached graph when there is one.
    //
    // `iteration` is the caller's 1-based loop counter: the first iteration always runs
    // uncaptured so it can warm allocation pools and lazy buffers, which capture forbids.
    // Capture is attempted on the second, and abandoned for the rest of the solve if it
    // fails -- an invalidated capture records without executing, which PcgAmgSolver has to
    // survive rather than propagate.
    template <typename Body>
    void run(cudaStream_t stream, const Key& key, int iteration, Body&& body);

private:
    void reset() noexcept;
    // Instantiating costs more than updating, so an unchanged topology reuses the executable
    // graph and only re-instantiates when cudaGraphExecUpdate declines.
    bool adopt(cudaGraph_t graph) noexcept;

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    Key key_{};
    bool capture_failed_ = false;
};

inline void GraphCache::reset() noexcept {
    if (exec_) cudaGraphExecDestroy(exec_);
    if (graph_) cudaGraphDestroy(graph_);
    exec_ = nullptr;
    graph_ = nullptr;
}

inline bool GraphCache::adopt(cudaGraph_t graph) noexcept {
    if (exec_ != nullptr && cudaGraphExecUpdate(exec_, graph, nullptr) == cudaSuccess) {
        if (graph_) cudaGraphDestroy(graph_);
        graph_ = graph;
        return true;
    }
    // Topology moved (or the update was refused for any other reason): start over.
    reset();
    if (cudaGraphInstantiate(&exec_, graph, 0) != cudaSuccess) {
        exec_ = nullptr;
        cudaGraphDestroy(graph);
        return false;
    }
    graph_ = graph;
    return true;
}

template <typename Body>
void GraphCache::run(cudaStream_t stream, const Key& key, int iteration, Body&& body) {
    // A failed capture is given up on for the rest of that solve but retried by the next one,
    // since whatever invalidated it is usually not a property of this body.
    if (iteration == 1) capture_failed_ = false;
    if (exec_ != nullptr && !(key_ == key)) reset();
    if (exec_ != nullptr) {
        check_cuda(cudaGraphLaunch(exec_, stream), "solver", "graph_launch");
        return;
    }
    if (iteration == 1 || capture_failed_) {
        body();  // iteration 1 warms pools and lazy buffers so capture sees no allocation
        return;
    }

    cudaGraph_t captured = nullptr;
    const cudaError_t begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    body();
    const cudaError_t ended = cudaStreamEndCapture(stream, &captured);
    if (begun != cudaSuccess || ended != cudaSuccess || captured == nullptr) {
        // An invalidated capture records without executing, so this iteration still has to
        // run for real. Stop attempting capture for the rest of the solve.
        if (captured) cudaGraphDestroy(captured);
        capture_failed_ = true;
        body();
        return;
    }
    if (!adopt(captured)) {
        capture_failed_ = true;
        body();
        return;
    }
    key_ = key;
    check_cuda(cudaGraphLaunch(exec_, stream), "solver", "graph_launch_first");
}

class PcgAmgSolver {
public:
    PcgAmgSolver(int n, int nnz, const int* row_ptr, const int* col_idx, const double* values);
    ~PcgAmgSolver();

    PcgAmgSolver(const PcgAmgSolver&) = delete;
    PcgAmgSolver& operator=(const PcgAmgSolver&) = delete;

    // Dimensions of the matrix this was built on, so callers can validate their arrays.
    int n() const { return n_; }
    int nnz() const { return nnz_; }

    void update_values(const double* values, cudaStream_t stream);
    PcgResult solve_mixed(NativeVCycle& preconditioner, const double* b, double* x,
                          double tolerance, int max_iters, cudaStream_t stream,
                          const double* x0 = nullptr);
    // Block solve: k independent CG chains in lockstep over row-major (n, k) operands, sharing
    // every stiffness-matrix read (block SpMV + block V-cycle). The block runs until its worst
    // column reaches tolerance, but each column freezes at its own, so a column's answer does
    // not depend on who it was batched with. Per-column residuals are reported so callers can
    // fall back per column. k in {1, 2, 4, 8}.
    PcgBlockResult solve_mixed_block(NativeVCycle& preconditioner, const double* B, double* X,
                                     int k, double tolerance, int max_iters,
                                     cudaStream_t stream, const double* X0 = nullptr);

private:
    void ensure_block_buffers(int k);

    int n_ = 0;
    int nnz_ = 0;
    DeviceBuffer<int> row_ptr_;
    DeviceBuffer<int> col_idx_;
    DeviceBuffer<double> values_;
    DeviceBuffer<double> r_;
    DeviceBuffer<double> p_;
    DeviceBuffer<double> ap_;
    DeviceBuffer<double> x_int_;
    DeviceBuffer<float> rf_;
    DeviceBuffer<float> zf_;
    // CG scalars kept on-device: [rz, pap, alpha, neg_alpha, norm, beta]. Only the residual norm
    // is copied back, into pinned host memory, once/iter.
    DeviceBuffer<double> scalars_;
    // Per-block partial sums for the fused deterministic reductions (‖r‖², r·z).
    DeviceBuffer<double> partials_;
    PinnedBuffer<double> h_norm_;
    // solve_mixed runs on this internal, capture-capable stream because the caller's is usually the
    // un-capturable legacy default stream; b/x are handed off via join_event_. The iteration body
    // only touches solver-owned buffers (x_int_, not the caller's x), so the captured graph is
    // reused across solves as long as the preconditioner identity/generation is unchanged.
    CudaStream solve_stream_;
    CudaEvent join_event_;
    GraphCache graph_;
    // Block-solve state: (n, k) row-major work buffers, lazily sized to the largest k seen.
    int block_k_ = 0;
    DeviceBuffer<double> r_blk_;
    DeviceBuffer<double> p_blk_;
    DeviceBuffer<double> ap_blk_;
    DeviceBuffer<double> x_int_blk_;
    DeviceBuffer<float> rf_blk_;
    DeviceBuffer<float> zf_blk_;
    // Layout: [rz | pap | alpha | neg_alpha | norm | beta | ref_sq | converged], each k wide.
    DeviceBuffer<double> scalars_blk_;
    DeviceBuffer<double> partials_blk_;
    PinnedBuffer<double> h_norms_blk_;  // pinned, k residual norms + k reference norms
    GraphCache block_graph_;
};
