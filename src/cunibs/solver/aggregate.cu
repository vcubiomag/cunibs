#include "aggregate.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int kWarpsPerBlock = kBlock / kWarp;

// AMGx defaults (core.cu:465-466). Fixed rather than exposed: nothing in the pipeline varies
// them, and round 1 must not apply kMaxUnassignedFraction at all (see select_size4).
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

// The strongest neighbour of a row: the lexicographic maximum of (weight, column) starting
// from (0, -1). Every row scan below is that same fold, so the tie-break (highest column
// index wins) lives in one place.
//
// Deliberately one thread per row rather than a warp fraction as in vcycle.cu. Consecutive
// CSR rows are contiguous, so a warp already walks a contiguous span and these kernels
// already run near peak bandwidth; splitting a row across lanes buys tighter per-instruction
// coalescing and pays for it in shuffles. On an 8M-row 7-point stencil that was a 70% loss
// at 8 lanes per row and 11% at 2, and only rows well denser than these levels carry broke
// even.
struct Best {
    float w;
    int col;
};

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
// (common_selector.h:123-125), which is unconditional upstream. Dropping it costs
// bit-exactness at scale: sub-001 lands on 208827 aggregates instead of 208828.
//
// The entries the matching skips, the diagonal and the columns past the row count, are set
// to -1 here rather than by a separate fill pass over all nnz: the rows of a valid CSR cover
// every j in [0, nnz) exactly once, so `w` ends up fully written either way.
__global__ __launch_bounds__(kBlock) void agg_edge_weight_kernel(
    int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const float* __restrict__ values, const float* __restrict__ diag, float* __restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d_i = fabsf(diag[i]);
    const int row_e = row_ptr[i + 1];
    for (int j = row_ptr[i]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (jc == i || jc >= n) {
            w[j] = -1.0f;
            continue;
        }
        const float den = fmaxf(d_i, fabsf(diag[jc]));
        const float ed = fabsf(values[j]) / den;
        const unsigned int lo = static_cast<unsigned int>(i < jc ? i : jc);
        const unsigned int hi = static_cast<unsigned int>(i < jc ? jc : i);
        const float frac = 1e-5f * static_cast<float>(agg_hash(lo, hi)) / 4294967295.0f;
        w[j] = ed + frac * ed;
    }
}

// Round 1: strongest unassigned neighbour. The absent else-branch is deliberate: leaving a
// stale `strongest` lets a row that momentarily sees no unassigned neighbour still match on
// its last proposal. Resetting it here measured 4.34 -> 3.68 coarsening and +35% PCG
// iterations on the test patch.
__global__ __launch_bounds__(kBlock) void agg_find_strongest_nomerge_kernel(
    int n, const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const float* __restrict__ w, const int* __restrict__ partner, int* __restrict__ strongest) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || partner[tid] != -1) return;
    Best best{0.0f, -1};
    const int row_e = row_ptr[tid + 1];
    for (int j = row_ptr[tid]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (jc == tid || jc >= n || partner[jc] != -1) continue;
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
            agg[tid] = (pm > tid) ? tid : pm;
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
        if (jc == tid || jc >= n) continue;
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
            agg[tid] = (pm > mine) ? mine : pm;
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
        if (jc == tid || jc >= n || aggregated[jc] == -1) continue;
        best_take(best, w[j], jc);
    }
    cand[tid] = (best.col != -1) ? agg[best.col] : tid;
}

__global__ __launch_bounds__(kBlock) void agg_join_existing_kernel(
    int n, int* __restrict__ agg, int* __restrict__ aggregated, const int* __restrict__ cand,
    int* __restrict__ remaining) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int unassigned = 0;
    if (tid < n && aggregated[tid] == -1) {
        if (cand[tid] != -1) {
            agg[tid] = cand[tid];
            aggregated[tid] = 1;
        } else {
            unassigned = 1;
        }
    }
    count_block(unassigned, remaining);
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

// Exclusive prefix sum over the used-label flags, in the usual three passes over fixed tiles.
// CUB would do this in one call; hand-rolling it keeps this file free of any dependency
// beyond the CUDA runtime, which is the point of the file. Every add is an integer add, so
// the result does not depend on how the tiles get scheduled.
constexpr int kScanItemsPerThread = 8;
constexpr int kScanTile = kBlock * kScanItemsPerThread;

// Block-wide exclusive scan of one value per thread; `total` receives the block sum. All
// threads must call it, and callers must __syncthreads() between successive calls, which
// share `smem`.
__device__ __forceinline__ int block_exclusive_scan(int value, int& total,
                                                    int* __restrict__ smem) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const int warp = static_cast<int>(threadIdx.x) / kWarp;

    int x = value;
#pragma unroll
    for (int off = 1; off < kWarp; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == kWarp - 1) smem[warp] = x;
    __syncthreads();

    if (warp == 0) {
        int s = (lane < kWarpsPerBlock) ? smem[lane] : 0;
#pragma unroll
        for (int off = 1; off < kWarpsPerBlock; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, off);
            if (lane >= off) s += y;
        }
        if (lane < kWarpsPerBlock) smem[lane] = s;
    }
    __syncthreads();

    total = smem[kWarpsPerBlock - 1];
    return (warp == 0 ? 0 : smem[warp - 1]) + x - value;
}

__global__ __launch_bounds__(kBlock) void agg_scan_tile_sums_kernel(
    int n, const int* __restrict__ in, int* __restrict__ tile_sum) {
    __shared__ int smem[kWarpsPerBlock];
    const int tile = blockIdx.x * kScanTile;
    int sum = 0;
    for (int r = 0; r < kScanItemsPerThread; ++r) {
        const int i = tile + r * kBlock + static_cast<int>(threadIdx.x);
        if (i < n) sum += in[i];
    }
    int total = 0;
    block_exclusive_scan(sum, total, smem);
    if (threadIdx.x == 0) tile_sum[blockIdx.x] = total;
}

// One block walking the tile sums, so the offsets come out in a single pass with no second
// level to scan. A few thousand tiles at most, 256 at a time.
__global__ __launch_bounds__(kBlock) void agg_scan_tile_offsets_kernel(
    int n_tiles, int* __restrict__ tile_sum, int* __restrict__ total_out) {
    __shared__ int smem[kWarpsPerBlock];
    int running = 0;
    for (int base = 0; base < n_tiles; base += kBlock) {
        const int i = base + static_cast<int>(threadIdx.x);
        const int v = (i < n_tiles) ? tile_sum[i] : 0;
        int total = 0;
        const int prefix = block_exclusive_scan(v, total, smem);
        if (i < n_tiles) tile_sum[i] = running + prefix;
        running += total;
        __syncthreads();
    }
    if (threadIdx.x == 0) *total_out = running;
}

__global__ __launch_bounds__(kBlock) void agg_scan_write_kernel(int n, const int* __restrict__ in,
                                                                const int* __restrict__ tile_sum,
                                                                int* __restrict__ out) {
    __shared__ int smem[kWarpsPerBlock];
    const int tile = blockIdx.x * kScanTile;
    int running = tile_sum[blockIdx.x];
    for (int r = 0; r < kScanItemsPerThread; ++r) {
        const int i = tile + r * kBlock + static_cast<int>(threadIdx.x);
        const int v = (i < n) ? in[i] : 0;
        int total = 0;
        const int prefix = block_exclusive_scan(v, total, smem);
        if (i < n) out[i] = running + prefix;
        running += total;
        __syncthreads();
    }
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
    int* tile_sum;
    int* counter;
};

Scratch lay_out(Bump& bump, size_t n, size_t nnz, size_t scan_tiles) {
    Scratch s{};
    s.diag = bump.take<float>(n);
    s.w = bump.take<float>(nnz > 0 ? nnz : 1);
    s.wsn = bump.take<float>(n);
    s.partner = bump.take<int>(n);
    s.strongest = bump.take<int>(n);
    s.aggregated = bump.take<int>(n);
    s.cand = bump.take<int>(n);
    s.used = bump.take<int>(n);
    s.excl = bump.take<int>(n + 1);
    s.tile_sum = bump.take<int>(scan_tiles);
    s.counter = bump.take<int>(1);
    return s;
}

// cudaMalloc synchronises the whole device, so every scratch buffer shares one allocation:
// on the small coarse levels the separate allocations cost more than the aggregation itself.
class DeviceArena {
  public:
    explicit DeviceArena(size_t bytes) {
        check_cuda(cudaMalloc(&base_, bytes), "alloc aggregate scratch");
    }
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    ~DeviceArena() {
        if (base_ != nullptr) cudaFree(base_);
    }

    std::byte* get() const { return static_cast<std::byte*>(base_); }

  private:
    void* base_ = nullptr;
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

    // Both are nonzero: n_rows > 0 above, and grid_for only returns 0 for an empty range.
    const unsigned blocks = grid_for(n_rows, kBlock);
    const unsigned scan_tiles = grid_for(n_rows, kScanTile);

    Bump sizing(nullptr);
    lay_out(sizing, n_rows, nnz, scan_tiles);
    DeviceArena arena(sizing.bytes());
    Bump bump(arena.get());
    const Scratch s = lay_out(bump, n_rows, nnz, scan_tiles);

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

    // Leftovers join their strongest aggregated neighbour. `cand` is always assigned, so this
    // converges in one pass; the cap only guards against an unforeseen input.
    constexpr int kMaxMergePasses = 20;
    for (int pass = 0; unassigned != 0; ++pass) {
        if (pass >= kMaxMergePasses) {
            throw std::runtime_error("aggregate: leftover merge did not terminate");
        }
        check_cuda(cudaMemsetAsync(s.counter, 0, sizeof(int), stream), "memset counter");
        agg_merge_existing_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, row_ptr, col_idx, s.w,
                                                                 aggregates, s.aggregated,
                                                                 s.cand);
        agg_join_existing_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, s.aggregated,
                                                                s.cand, s.counter);
        check_cuda(cudaGetLastError(), "leftover merge launch");
        unassigned = read_counter(s.counter, stream);
    }

    // Order-preserving dense renumbering (agg_selector.cu:20-44): mark the labels in use,
    // exclusive-scan the flags, and gather. Surjective by construction.
    check_cuda(cudaMemsetAsync(s.used, 0, static_cast<size_t>(n_rows) * sizeof(int), stream),
               "memset used");
    agg_mark_used_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, s.used);
    agg_scan_tile_sums_kernel<<<scan_tiles, kBlock, 0, stream>>>(n_rows, s.used, s.tile_sum);
    agg_scan_tile_offsets_kernel<<<1, kBlock, 0, stream>>>(static_cast<int>(scan_tiles),
                                                           s.tile_sum, s.excl + n_rows);
    agg_scan_write_kernel<<<scan_tiles, kBlock, 0, stream>>>(n_rows, s.used, s.tile_sum, s.excl);
    agg_relabel_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, s.excl);
    check_cuda(cudaGetLastError(), "renumber launch");

    return read_counter(s.excl + n_rows, stream);
}
