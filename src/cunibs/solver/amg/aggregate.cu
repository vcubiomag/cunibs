#include "amg/aggregate.hpp"

#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

// AMGx defaults (core.cu). Fixed rather than exposed: nothing in the pipeline varies them,
// and round 1 must not apply kMaxUnassignedFraction at all (see select_size4).
constexpr int kMaxMatchingIterations = 15;
constexpr float kMaxUnassignedFraction = 0.05f;

void check_cuda(cudaError_t err, const char* what) { ::check_cuda(err, "aggregate", what); }

__device__ __forceinline__ unsigned int agg_hash(unsigned int a, unsigned int seed) {
    a ^= seed;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return a;
}

// The strongest neighbour of a row: the lexicographic maximum of (weight, column) starting from
// (0, -1). Every row scan below is that same fold, so the tie-break (highest column index wins)
// lives in one place.
//
// One thread per row rather than a warp fraction as in vcycle.cu: consecutive CSR rows are
// contiguous, so a warp already walks a contiguous span and these kernels run near peak
// bandwidth. Splitting a row across lanes would pay in shuffles for coalescing it already has.
struct Best {
    float w;
    int col;
};

// A column is a matching candidate when it is off the diagonal and names a row this level
// actually has. The unsigned compare rejects a negative index in the same test as an oversized
// one: without it a negative column reaches diag/partner/aggregated below their base pointers.
__device__ __forceinline__ bool is_candidate(int col, int row, int n) {
    return col != row && static_cast<unsigned>(col) < static_cast<unsigned>(n);
}

// The `==` drops a NaN weight, which agg_edge_weight_kernel produces when a stored zero meets two
// zero diagonals.
__device__ __forceinline__ void best_take(Best& acc, float w, int col) {
    if (w > acc.w || (w == acc.w && col > acc.col)) {
        acc.w = w;
        acc.col = col;
    }
}

// One atomic per block instead of one per row, folded into the kernel that produces the
// flag so the convergence check costs no extra pass over the row arrays. Integer addition
// is associative, so the order the blocks arrive in does not affect the count. Every thread
// of the block must reach this.
__device__ __forceinline__ void count_block(int predicate, int* __restrict__ counter) {
    const int n_set = __syncthreads_count(predicate);
    if (threadIdx.x == 0 && n_set != 0) atomicAdd(counter, n_set);
}

__global__ __launch_bounds__(kBlock) void agg_init_round1_kernel(int n, int* __restrict__ agg,
                                                                 int* __restrict__ partner,
                                                                 int* __restrict__ strongest) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    agg[i] = i;
    partner[i] = -1;
    strongest[i] = -1;
}

// `strongest` is deliberately not reset here: round 2 starts from round 1's proposals.
__global__ __launch_bounds__(kBlock) void agg_init_round2_kernel(int n, float* __restrict__ wsn,
                                                                 int* __restrict__ aggregated) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    wsn[i] = -1.0f;
    aggregated[i] = -1;
}

__global__ __launch_bounds__(kBlock) void agg_diag_kernel(int n, const int* __restrict__ row_ptr,
                                                          const int* __restrict__ col_idx,
                                                          const float* __restrict__ values,
                                                          float* __restrict__ diag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = 0.0f;
    const int row_e = row_ptr[i + 1];
    for (int j = row_ptr[i]; j < row_e; ++j) {
        if (col_idx[j] == i) {
            d = values[j];
            break;
        }
    }
    diag[i] = d;
}

// w_ij = |a_ij| / max(|a_ii|, |a_jj|), then AMGx's uniform-weight perturbation
// (common_selector.h), which breaks ties between equal-weight edges.
//
// The entries the matching skips, the diagonal and the columns past the row count, are set to -1
// here rather than by a separate fill pass over all nnz: the rows of a valid CSR cover every j in
// [0, nnz) exactly once, so `w` ends up fully written either way.
__global__ __launch_bounds__(kBlock) void agg_edge_weight_kernel(
    int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const float* __restrict__ values, const float* __restrict__ diag, float* __restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d_i = fabsf(diag[i]);
    const int row_e = row_ptr[i + 1];
    for (int j = row_ptr[i]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (!is_candidate(jc, i, n)) {
            w[j] = -1.0f;
            continue;
        }
        const float den = fmaxf(d_i, fabsf(diag[jc]));
        const float ed = fabsf(values[j]) / den;
        const unsigned int lo = static_cast<unsigned int>(min(i, jc));
        const unsigned int hi = static_cast<unsigned int>(max(i, jc));
        const float frac = 1e-5f * static_cast<float>(agg_hash(lo, hi)) / 4294967295.0f;
        w[j] = ed + frac * ed;
    }
}

// Round 1: strongest unassigned neighbour. The absent else-branch is deliberate: leaving a stale
// `strongest` lets a row that momentarily sees no unassigned neighbour still match on its last
// proposal, which is worth a markedly coarser hierarchy.
__global__ __launch_bounds__(kBlock) void agg_find_strongest_nomerge_kernel(
    int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const float* __restrict__ w, const int* __restrict__ partner, int* __restrict__ strongest) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || partner[tid] != -1) return;
    Best best{0.0f, -1};
    const int row_e = row_ptr[tid + 1];
    for (int j = row_ptr[tid]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (!is_candidate(jc, tid, n) || partner[jc] != -1) continue;
        best_take(best, w[j], jc);
    }
    if (best.col != -1) strongest[tid] = best.col;
}

__global__ __launch_bounds__(kBlock) void agg_match_edges_kernel(
    int n, int* __restrict__ partner, int* __restrict__ agg, const int* __restrict__ strongest,
    int* __restrict__ remaining) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int unassigned = 0;
    // A row's own thread is the only writer of partner[tid], so it also knows, without a
    // second pass, whether that slot is still unset once this kernel retires.
    if (tid < n && partner[tid] == -1) {
        const int pm = strongest[tid];
        if (pm != -1 && strongest[pm] == tid) {
            partner[tid] = pm;
            agg[tid] = min(pm, tid);
        } else {
            unassigned = 1;
        }
    }
    count_block(unassigned, remaining);
}

__global__ __launch_bounds__(kBlock) void agg_assign_unassigned_kernel(
    int n, int* __restrict__ partner) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && partner[tid] == -1) partner[tid] = tid;
}

// Round 2: strongest unaggregated neighbour outside the row's own pair, recorded as that
// neighbour's aggregate id. Same deliberate stale-state carry-over as round 1.
__global__ __launch_bounds__(kBlock) void agg_find_strongest_store_kernel(
    int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const float* __restrict__ w, const int* __restrict__ aggregated, const int* __restrict__ agg,
    const int* __restrict__ partner, int* __restrict__ strongest, float* __restrict__ wsn) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || aggregated[tid] != -1) return;
    const int p = partner[tid];
    Best best{0.0f, -1};
    const int row_e = row_ptr[tid + 1];
    for (int j = row_ptr[tid]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (!is_candidate(jc, tid, n)) continue;
        if (aggregated[jc] != -1 || jc == p) continue;
        best_take(best, w[j], jc);
    }
    if (best.col != -1) {
        wsn[tid] = best.w;
        strongest[tid] = agg[best.col];
    }
}

// Both members of a pair commit to one proposal. Race-free despite reading the partner's
// slot: thread tid writes only when wsn[tid] < wsn[p] and thread p only when the reverse
// holds, which is mutually exclusive; when both are negative neither reads the other.
__global__ __launch_bounds__(kBlock) void agg_agree_on_proposal_kernel(
    int n, int* __restrict__ aggregated, int* __restrict__ strongest,
    const float* __restrict__ wsn, const int* __restrict__ partner) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || aggregated[tid] != -1) return;
    const int p = partner[tid];
    const float mine = wsn[tid];
    const float theirs = (p != -1) ? wsn[p] : -1.0f;
    if (mine < 0.0f && theirs < 0.0f) {
        aggregated[tid] = 1;
        strongest[tid] = -1;
    } else if (mine < theirs) {
        strongest[tid] = strongest[p];
    }
}

__global__ __launch_bounds__(kBlock) void agg_match_aggregates_kernel(
    int n, int* __restrict__ agg, int* __restrict__ aggregated,
    const int* __restrict__ strongest, int* __restrict__ remaining) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int unassigned = 0;
    if (tid < n && aggregated[tid] == -1) {
        const int pm = strongest[tid];
        const int mine = agg[tid];
        if (pm != -1 && strongest[pm] == mine) {
            aggregated[tid] = 1;
            agg[tid] = min(pm, mine);
        } else {
            unassigned = 1;
        }
    }
    count_block(unassigned, remaining);
}

__global__ __launch_bounds__(kBlock) void agg_merge_existing_kernel(
    int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const float* __restrict__ w, const int* __restrict__ agg,
    const int* __restrict__ aggregated, int* __restrict__ cand) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || aggregated[tid] != -1) return;
    Best best{0.0f, -1};
    const int row_e = row_ptr[tid + 1];
    for (int j = row_ptr[tid]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (!is_candidate(jc, tid, n) || aggregated[jc] == -1) continue;
        best_take(best, w[j], jc);
    }
    cand[tid] = (best.col != -1) ? agg[best.col] : tid;
}

// Every row agg_merge_existing_kernel considers gets a candidate (its own index if no
// aggregated neighbour), so this assigns all of them and no row can survive to a second pass.
__global__ __launch_bounds__(kBlock) void agg_join_existing_kernel(
    int n, int* __restrict__ agg, int* __restrict__ aggregated, const int* __restrict__ cand) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && aggregated[tid] == -1) {
        agg[tid] = cand[tid];
        aggregated[tid] = 1;
    }
}

// Rows of the same aggregate race to store here, but they all store 1, and the memory model
// guarantees one of the values written lands.
__global__ __launch_bounds__(kBlock) void agg_mark_used_kernel(int n, const int* __restrict__ agg,
                                                               int* __restrict__ used) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) used[agg[tid]] = 1;
}

__global__ __launch_bounds__(kBlock) void agg_relabel_kernel(int n, int* __restrict__ agg,
                                                             const int* __restrict__ excl) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) agg[tid] = excl[agg[tid]];
}

constexpr size_t kArenaAlign = 256;

constexpr size_t align_up(size_t bytes) {
    return (bytes + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
}

// Bump allocator over one device block. Sizing runs it against a null base first, so the
// sizing pass and the pointer pass cannot drift apart.
class Bump {
  public:
    explicit Bump(std::byte* base) : base_(base) {}

    template <typename T>
    T* take(size_t count) {
        std::byte* p = base_ != nullptr ? base_ + used_ : nullptr;
        used_ += align_up(count * sizeof(T));
        return reinterpret_cast<T*>(p);
    }

    size_t bytes() const { return used_; }

  private:
    std::byte* base_;
    size_t used_ = 0;
};

struct Scratch {
    float* diag;
    float* w;
    float* wsn;
    int* partner;
    int* strongest;
    int* aggregated;
    int* cand;
    int* used;
    int* excl;
    int* counter;
    std::byte* cub;
};

Scratch lay_out(Bump& bump, size_t n, size_t nnz, size_t cub_bytes) {
    Scratch s{};
    s.diag = bump.take<float>(n);
    s.w = bump.take<float>(nnz > 0 ? nnz : 1);
    s.wsn = bump.take<float>(n);
    s.partner = bump.take<int>(n);
    s.strongest = bump.take<int>(n);
    s.aggregated = bump.take<int>(n);
    s.cand = bump.take<int>(n);
    // One slot past the flags so the exclusive scan has somewhere to deposit the total.
    s.used = bump.take<int>(n + 1);
    s.excl = bump.take<int>(n + 1);
    s.counter = bump.take<int>(1);
    s.cub = bump.take<std::byte>(cub_bytes);
    return s;
}

// Every scratch buffer shares one stream-ordered allocation: on the small coarse levels a dozen
// separate ones cost more than the aggregation itself.
class DeviceArena {
  public:
    DeviceArena(size_t bytes, cudaStream_t stream) : stream_(stream) {
        check_cuda(cudaMallocAsync(&base_, bytes, stream), "alloc aggregate scratch");
    }
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    ~DeviceArena() {
        if (base_ != nullptr) cudaFreeAsync(base_, stream_);
    }

    std::byte* get() const { return static_cast<std::byte*>(base_); }

  private:
    void* base_ = nullptr;
    cudaStream_t stream_;
};

// The matching loops are data dependent, so each one costs a round trip per iteration.
int read_counter(const int* counter, cudaStream_t stream) {
    int host = 0;
    check_cuda(cudaMemcpyAsync(&host, counter, sizeof(int), cudaMemcpyDeviceToHost, stream),
               "copy counter");
    check_cuda(cudaStreamSynchronize(stream), "sync counter");
    return host;
}

}  // namespace

int select_size4(int n_rows, int nnz, const int* row_ptr, const int* col_idx,
                 const float* values, int* aggregates, cudaStream_t stream) {
    if (n_rows <= 0) return 0;
    if (nnz < 0) throw std::invalid_argument("aggregate: nnz must not be negative");

    // The caller builds each level's operator with asynchronous library calls and never syncs,
    // so read_counter below is the first synchronising point on the whole hierarchy build.
    // Draining here keeps an asynchronous fault from that upstream work out of the aggregation's
    // own error reports.
    check_cuda(cudaStreamSynchronize(stream), "fault from work queued before this call");

    // Nonzero: n_rows > 0 above, and grid_for only returns 0 for an empty range.
    const unsigned blocks = grid_for(n_rows, kBlock);

    size_t cub_bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, static_cast<int*>(nullptr),
                                             static_cast<int*>(nullptr), n_rows + 1, stream),
               "size(renumber scan)");

    Bump sizing(nullptr);
    lay_out(sizing, n_rows, nnz, cub_bytes);
    DeviceArena arena(sizing.bytes(), stream);
    Bump bump(arena.get());
    const Scratch s = lay_out(bump, n_rows, nnz, cub_bytes);

    agg_diag_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, row_ptr, col_idx, values, s.diag);
    if (nnz > 0) {
        agg_edge_weight_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, row_ptr, col_idx, values,
                                                              s.diag, s.w);
    }
    agg_init_round1_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, s.partner,
                                                          s.strongest);
    check_cuda(cudaGetLastError(), "round 1 init launch");

    // Round 1: pairs. AMGx counts unassigned rows over a 3n-length partner array, which makes
    // its `== 0` and `< kMaxUnassignedFraction` exits unreachable here; counting over n and
    // dropping the fraction rule is equivalent, whereas applying it would leave up to 5% of
    // rows self-paired.
    int unassigned = n_rows;
    for (int iter = 0; iter <= kMaxMatchingIterations; ++iter) {
        check_cuda(cudaMemsetAsync(s.counter, 0, sizeof(int), stream), "memset counter");
        agg_find_strongest_nomerge_kernel<<<blocks, kBlock, 0, stream>>>(
            n_rows, row_ptr, col_idx, s.w, s.partner, s.strongest);
        agg_match_edges_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, s.partner, aggregates,
                                                              s.strongest, s.counter);
        check_cuda(cudaGetLastError(), "round 1 launch");
        const int previous = unassigned;
        unassigned = read_counter(s.counter, stream);
        if (unassigned == 0 || unassigned == previous) break;
    }

    agg_assign_unassigned_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, s.partner);
    agg_init_round2_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, s.wsn, s.aggregated);
    check_cuda(cudaGetLastError(), "round 2 init launch");

    // Round 2: merge pairs into quads. Here all four AMGx exits apply, including the
    // kMaxUnassignedFraction one, because `aggregated` really is n-length upstream.
    unassigned = n_rows;
    for (int iter = 0; iter <= kMaxMatchingIterations; ++iter) {
        check_cuda(cudaMemsetAsync(s.counter, 0, sizeof(int), stream), "memset counter");
        agg_find_strongest_store_kernel<<<blocks, kBlock, 0, stream>>>(
            n_rows, row_ptr, col_idx, s.w, s.aggregated, aggregates, s.partner, s.strongest,
            s.wsn);
        agg_agree_on_proposal_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, s.aggregated,
                                                                    s.strongest, s.wsn,
                                                                    s.partner);
        agg_match_aggregates_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates,
                                                                   s.aggregated, s.strongest,
                                                                   s.counter);
        check_cuda(cudaGetLastError(), "round 2 launch");
        const int previous = unassigned;
        unassigned = read_counter(s.counter, stream);
        if (unassigned == 0 || unassigned == previous ||
            static_cast<float>(unassigned) / static_cast<float>(n_rows) <
                kMaxUnassignedFraction) {
            break;
        }
    }

    // Leftovers join their strongest aggregated neighbour. One pass assigns all of them:
    // agg_merge_existing_kernel gives every remaining row a candidate, falling back to the row
    // itself, so there is nothing for a second pass to do and no counter to read back.
    if (unassigned != 0) {
        agg_merge_existing_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, row_ptr, col_idx, s.w,
                                                                 aggregates, s.aggregated,
                                                                 s.cand);
        agg_join_existing_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, s.aggregated,
                                                                s.cand);
        check_cuda(cudaGetLastError(), "leftover merge launch");
    }

    // Order-preserving dense renumbering (agg_selector.cu): mark the labels in use,
    // exclusive-scan the flags, and gather. Surjective by construction. The scan runs over one
    // slot more than there are rows, so excl[n_rows] comes back holding the aggregate count.
    check_cuda(cudaMemsetAsync(s.used, 0, (static_cast<size_t>(n_rows) + 1) * sizeof(int), stream),
               "memset used");
    agg_mark_used_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, s.used);
    check_cuda(cudaGetLastError(), "mark used launch");
    check_cuda(cub::DeviceScan::ExclusiveSum(s.cub, cub_bytes, s.used, s.excl, n_rows + 1, stream),
               "renumber scan");
    agg_relabel_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, s.excl);
    check_cuda(cudaGetLastError(), "relabel launch");

    return read_counter(s.excl + n_rows, stream);
}
