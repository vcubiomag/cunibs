#include "aggregate.hpp"

#include <stdexcept>
#include <string>

namespace {

constexpr int kBlock = 256;

// AMGx defaults (core.cu:465-466). Fixed rather than exposed: nothing in the pipeline varies
// them, and round 1 must not apply kMaxUnassignedFraction at all (see select_size4).
constexpr int kMaxMatchingIterations = 15;
constexpr float kMaxUnassignedFraction = 0.05f;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("aggregate CUDA error (") + what +
                                 "): " + cudaGetErrorString(err));
    }
}

int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// Owns one device allocation and frees it even if a later step throws.
struct DevBuf {
    void* ptr = nullptr;
    DevBuf() = default;
    explicit DevBuf(size_t bytes, const char* what) {
        check_cuda(cudaMalloc(&ptr, bytes), what);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr = o.ptr;
            o.ptr = nullptr;
        }
        return *this;
    }
    ~DevBuf() {
        if (ptr) cudaFree(ptr);
    }
    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }
};

__device__ unsigned int agg_hash(unsigned int a, unsigned int seed) {
    a ^= seed;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return a;
}

__global__ void agg_fill_f32_kernel(int n, float value, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

__global__ void agg_fill_i32_kernel(int n, int value, int* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

__global__ void agg_iota_kernel(int n, int* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

__global__ void agg_diag_kernel(int n, const int* __restrict__ row_ptr,
                                const int* __restrict__ col_idx,
                                const float* __restrict__ values, float* __restrict__ diag) {
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
__global__ void agg_edge_weight_kernel(int n, const int* __restrict__ row_ptr,
                                       const int* __restrict__ col_idx,
                                       const float* __restrict__ values,
                                       const float* __restrict__ diag, float* __restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d_i = fabsf(diag[i]);
    const int row_e = row_ptr[i + 1];
    for (int j = row_ptr[i]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (jc == i || jc >= n) continue;
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
__global__ void agg_find_strongest_nomerge_kernel(int n, const int* __restrict__ row_ptr,
                                                  const int* __restrict__ col_idx,
                                                  const float* __restrict__ w,
                                                  const int* __restrict__ partner,
                                                  int* __restrict__ strongest) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || partner[tid] != -1) return;
    float best = 0.0f;
    int s = -1;
    const int row_e = row_ptr[tid + 1];
    for (int j = row_ptr[tid]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (jc == tid || jc >= n) continue;
        const float ww = w[j];
        if (partner[jc] == -1 && (ww > best || (ww == best && jc > s))) {
            best = ww;
            s = jc;
        }
    }
    if (s != -1) strongest[tid] = s;
}

__global__ void agg_match_edges_kernel(int n, int* __restrict__ partner, int* __restrict__ agg,
                                       const int* __restrict__ strongest) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || partner[tid] != -1) return;
    const int pm = strongest[tid];
    if (pm == -1) return;
    if (strongest[pm] == tid) {
        partner[tid] = pm;
        agg[tid] = (pm > tid) ? tid : pm;
    }
}

__global__ void agg_assign_unassigned_kernel(int n, int* __restrict__ partner) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && partner[tid] == -1) partner[tid] = tid;
}

// Round 2: strongest unaggregated neighbour outside the row's own pair, recorded as that
// neighbour's aggregate id. Same deliberate stale-state carry-over as round 1.
__global__ void agg_find_strongest_store_kernel(int n, const int* __restrict__ row_ptr,
                                                const int* __restrict__ col_idx,
                                                const float* __restrict__ w,
                                                const int* __restrict__ aggregated,
                                                const int* __restrict__ agg,
                                                const int* __restrict__ partner,
                                                int* __restrict__ strongest,
                                                float* __restrict__ wsn) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || aggregated[tid] != -1) return;
    const int p = partner[tid];
    float best = 0.0f;
    int s = -1;
    const int row_e = row_ptr[tid + 1];
    for (int j = row_ptr[tid]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (jc == tid || jc >= n) continue;
        if (aggregated[jc] != -1 || jc == p) continue;
        const float ww = w[j];
        if (ww > best || (ww == best && jc > s)) {
            best = ww;
            s = jc;
        }
    }
    if (s != -1) {
        wsn[tid] = best;
        strongest[tid] = agg[s];
    }
}

// Both members of a pair commit to one proposal. Race-free despite reading the partner's
// slot: thread tid writes only when wsn[tid] < wsn[p] and thread p only when the reverse
// holds, which is mutually exclusive; when both are negative neither reads the other.
__global__ void agg_agree_on_proposal_kernel(int n, int* __restrict__ aggregated,
                                             int* __restrict__ strongest,
                                             const float* __restrict__ wsn,
                                             const int* __restrict__ partner) {
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

__global__ void agg_match_aggregates_kernel(int n, int* __restrict__ agg,
                                            int* __restrict__ aggregated,
                                            const int* __restrict__ strongest) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || aggregated[tid] != -1) return;
    const int pm = strongest[tid];
    if (pm == -1) return;
    const int mine = agg[tid];
    if (strongest[pm] == mine) {
        aggregated[tid] = 1;
        agg[tid] = (pm > mine) ? mine : pm;
    }
}

__global__ void agg_merge_existing_kernel(int n, const int* __restrict__ row_ptr,
                                          const int* __restrict__ col_idx,
                                          const float* __restrict__ w,
                                          const int* __restrict__ agg,
                                          const int* __restrict__ aggregated,
                                          int* __restrict__ cand) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || aggregated[tid] != -1) return;
    float best = 0.0f;
    int s = -1;
    const int row_e = row_ptr[tid + 1];
    for (int j = row_ptr[tid]; j < row_e; ++j) {
        const int jc = col_idx[j];
        if (jc == tid || jc >= n) continue;
        if (aggregated[jc] == -1) continue;
        const float ww = w[j];
        if (ww > best || (ww == best && jc > s)) {
            best = ww;
            s = jc;
        }
    }
    cand[tid] = (s != -1) ? agg[s] : tid;
}

__global__ void agg_join_existing_kernel(int n, int* __restrict__ agg,
                                         int* __restrict__ aggregated,
                                         const int* __restrict__ cand) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (aggregated[tid] == -1 && cand[tid] != -1) {
        agg[tid] = cand[tid];
        aggregated[tid] = 1;
    }
}

// Integer addition is associative, so the atomic order does not affect the result.
__global__ void agg_count_unset_kernel(int n, const int* __restrict__ flags,
                                       int* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && flags[tid] == -1) atomicAdd(out, 1);
}

__global__ void agg_mark_used_kernel(int n, const int* __restrict__ agg,
                                     int* __restrict__ used) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) used[agg[tid]] = 1;
}

__global__ void agg_relabel_kernel(int n, int* __restrict__ agg, const int* __restrict__ excl) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) agg[tid] = excl[agg[tid]];
}

// Exclusive prefix sum over n flags, in three passes over fixed contiguous chunks. CUB would
// do this in one call; hand-rolling it keeps this file free of any dependency beyond the CUDA
// runtime, which is the point of the file. The adds happen in a fixed order per chunk and the
// chunk offsets are scanned sequentially, so the result is deterministic.
constexpr int kScanChunks = 4096;

__device__ inline void scan_chunk_bounds(int n, int chunk_id, int& begin, int& end) {
    const int per = (n + kScanChunks - 1) / kScanChunks;
    begin = chunk_id * per;
    end = begin + per;
    if (begin > n) begin = n;
    if (end > n) end = n;
}

__global__ void agg_scan_chunk_sums_kernel(int n, const int* __restrict__ in,
                                           int* __restrict__ partial) {
    const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= kScanChunks) return;
    int begin, end;
    scan_chunk_bounds(n, chunk, begin, end);
    int sum = 0;
    for (int i = begin; i < end; ++i) sum += in[i];
    partial[chunk] = sum;
}

// One thread over kScanChunks entries: a few thousand dependent adds, once per level.
__global__ void agg_scan_partials_kernel(int* __restrict__ partial, int* __restrict__ total) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int running = 0;
    for (int i = 0; i < kScanChunks; ++i) {
        const int value = partial[i];
        partial[i] = running;
        running += value;
    }
    *total = running;
}

__global__ void agg_scan_write_kernel(int n, const int* __restrict__ in,
                                      const int* __restrict__ partial, int* __restrict__ out) {
    const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= kScanChunks) return;
    int begin, end;
    scan_chunk_bounds(n, chunk, begin, end);
    int running = partial[chunk];
    for (int i = begin; i < end; ++i) {
        out[i] = running;
        running += in[i];
    }
}

int count_unset(int n, const int* flags, int* d_counter, cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(d_counter, 0, sizeof(int), stream), "memset counter");
    agg_count_unset_kernel<<<grid_for(n), kBlock, 0, stream>>>(n, flags, d_counter);
    check_cuda(cudaGetLastError(), "count_unset launch");
    int host = 0;
    check_cuda(cudaMemcpyAsync(&host, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream),
               "copy counter");
    check_cuda(cudaStreamSynchronize(stream), "sync counter");
    return host;
}

}  // namespace

int select_size4(int n_rows, int nnz, const int* row_ptr, const int* col_idx,
                 const float* values, int* aggregates, cudaStream_t stream) {
    if (n_rows <= 0) return 0;

    const int blocks = grid_for(n_rows);

    DevBuf diag_buf(static_cast<size_t>(n_rows) * sizeof(float), "alloc diag");
    DevBuf w_buf(static_cast<size_t>(nnz > 0 ? nnz : 1) * sizeof(float), "alloc weights");
    DevBuf partner_buf(static_cast<size_t>(n_rows) * sizeof(int), "alloc partner");
    DevBuf strongest_buf(static_cast<size_t>(n_rows) * sizeof(int), "alloc strongest");
    DevBuf wsn_buf(static_cast<size_t>(n_rows) * sizeof(float), "alloc wsn");
    DevBuf aggregated_buf(static_cast<size_t>(n_rows) * sizeof(int), "alloc aggregated");
    DevBuf cand_buf(static_cast<size_t>(n_rows) * sizeof(int), "alloc cand");
    DevBuf used_buf(static_cast<size_t>(n_rows + 1) * sizeof(int), "alloc used");
    DevBuf excl_buf(static_cast<size_t>(n_rows + 1) * sizeof(int), "alloc excl");
    DevBuf counter_buf(sizeof(int), "alloc counter");

    float* diag = diag_buf.as<float>();
    float* w = w_buf.as<float>();
    int* partner = partner_buf.as<int>();
    int* strongest = strongest_buf.as<int>();
    float* wsn = wsn_buf.as<float>();
    int* aggregated = aggregated_buf.as<int>();
    int* cand = cand_buf.as<int>();
    int* used = used_buf.as<int>();
    int* excl = excl_buf.as<int>();
    int* counter = counter_buf.as<int>();

    agg_diag_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, row_ptr, col_idx, values, diag);
    check_cuda(cudaGetLastError(), "diag launch");

    if (nnz > 0) {
        agg_fill_f32_kernel<<<grid_for(nnz), kBlock, 0, stream>>>(nnz, -1.0f, w);
        check_cuda(cudaGetLastError(), "fill weights launch");
        agg_edge_weight_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, row_ptr, col_idx, values,
                                                              diag, w);
        check_cuda(cudaGetLastError(), "edge weight launch");
    }

    agg_iota_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates);
    agg_fill_i32_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, -1, partner);
    agg_fill_i32_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, -1, strongest);
    check_cuda(cudaGetLastError(), "round 1 init launch");

    // Round 1: pairs. AMGx counts unassigned rows over a 3n-length partner array, which makes
    // its `== 0` and `< kMaxUnassignedFraction` exits unreachable here; counting over n and
    // dropping the fraction rule is equivalent, whereas applying it would leave up to 5% of
    // rows self-paired.
    int unassigned = n_rows;
    for (int iter = 0; iter < kMaxMatchingIterations + 1; ++iter) {
        agg_find_strongest_nomerge_kernel<<<blocks, kBlock, 0, stream>>>(
            n_rows, row_ptr, col_idx, w, partner, strongest);
        agg_match_edges_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, partner, aggregates,
                                                              strongest);
        check_cuda(cudaGetLastError(), "round 1 launch");
        const int previous = unassigned;
        unassigned = count_unset(n_rows, partner, counter, stream);
        if (unassigned == 0 || unassigned == previous) break;
    }

    agg_assign_unassigned_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, partner);
    check_cuda(cudaGetLastError(), "assign unassigned launch");

    agg_fill_f32_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, -1.0f, wsn);
    agg_fill_i32_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, -1, aggregated);
    check_cuda(cudaGetLastError(), "round 2 init launch");

    // Round 2: merge pairs into quads. Here all four AMGx exits apply, including the
    // kMaxUnassignedFraction one, because `aggregated` really is n-length upstream.
    unassigned = n_rows;
    for (int iter = 0; iter < kMaxMatchingIterations + 1; ++iter) {
        agg_find_strongest_store_kernel<<<blocks, kBlock, 0, stream>>>(
            n_rows, row_ptr, col_idx, w, aggregated, aggregates, partner, strongest, wsn);
        agg_agree_on_proposal_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregated,
                                                                    strongest, wsn, partner);
        agg_match_aggregates_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates,
                                                                   aggregated, strongest);
        check_cuda(cudaGetLastError(), "round 2 launch");
        const int previous = unassigned;
        unassigned = count_unset(n_rows, aggregated, counter, stream);
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
        agg_merge_existing_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, row_ptr, col_idx, w,
                                                                 aggregates, aggregated, cand);
        agg_join_existing_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, aggregated,
                                                                cand);
        check_cuda(cudaGetLastError(), "leftover merge launch");
        unassigned = count_unset(n_rows, aggregated, counter, stream);
    }

    // Order-preserving dense renumbering (agg_selector.cu:20-44): mark the labels in use,
    // exclusive-scan the flags, and gather. Surjective by construction.
    check_cuda(cudaMemsetAsync(used, 0, static_cast<size_t>(n_rows + 1) * sizeof(int), stream),
               "memset used");
    agg_mark_used_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, used);
    check_cuda(cudaGetLastError(), "mark used launch");

    DevBuf partial_buf(static_cast<size_t>(kScanChunks) * sizeof(int), "alloc scan partials");
    int* partial = partial_buf.as<int>();
    const int scan_blocks = grid_for(kScanChunks);
    agg_scan_chunk_sums_kernel<<<scan_blocks, kBlock, 0, stream>>>(n_rows, used, partial);
    agg_scan_partials_kernel<<<1, 1, 0, stream>>>(partial, excl + n_rows);
    agg_scan_write_kernel<<<scan_blocks, kBlock, 0, stream>>>(n_rows, used, partial, excl);
    check_cuda(cudaGetLastError(), "scan launch");

    agg_relabel_kernel<<<blocks, kBlock, 0, stream>>>(n_rows, aggregates, excl);
    check_cuda(cudaGetLastError(), "relabel launch");

    int n_coarse = 0;
    check_cuda(cudaMemcpyAsync(&n_coarse, excl + n_rows, sizeof(int), cudaMemcpyDeviceToHost,
                               stream),
               "copy n_coarse");
    check_cuda(cudaStreamSynchronize(stream), "sync n_coarse");
    return n_coarse;
}
