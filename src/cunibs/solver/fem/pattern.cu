#include "fem/fem.hpp"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cuda/std/bit>
#include <cuda/std/limits>

#include <bit>
#include <cstdint>

// Segment-wise CSR construction: for every segment, the sorted distinct values of a candidate list
// it generates for itself. Two operators are built this way -- the stiffness sparsity pattern and
// the recovery patches -- and they differ only in how the candidates are gathered, which is all a
// gather policy below says. Each builder runs twice, once to size its rows and once to fill them,
// since nothing knows how many distinct values a segment has until it has sorted it.
//
// A segment is one node's incident corners or one patch's neighbour rings. Segments are split by
// width into two classes rather than padded to one global width, so that a single pathological
// node -- a fan from a centre node, which is how an analytic sphere is built -- cannot set the sort
// width for the whole mesh. Each class gets the primitive suited to its regime, and those are
// opposite ones. Millions of segments of a few hundred candidates get a warp each and sort in
// shared memory, where cub::DeviceSegmentedSort measured 12x slower. A handful of segments of tens
// of thousands materialise their candidates and go through that segmented sort, where a warp on a
// bitonic network over global memory is three orders of magnitude slower, one segment being one
// warp for the whole device. Neither class caps how wide a segment may be.
//
// Sorting a multiset has one answer and duplicates are dropped by keeping the first of each equal
// run, so no segment's output depends on how the work was scheduled or on which class handled it.

namespace {

void check_cuda(cudaError_t err, const char* what) { ::check_cuda(err, "pattern", what); }

constexpr int kSortWarps = 4;
// Static shared memory a block may claim without opting in to the dynamic limit.
constexpr std::size_t kMaxSharedBytes = 48 * 1024;
// The widest padded candidate list kSortWarps warps can hold at once, and so the boundary between
// the two classes. Keeping the shared class at full occupancy matters more than stretching it to
// cover one more segment: the widest segment on a real head is a few hundred candidates, so nothing
// there ever leaves this class.
constexpr int kFastPad = 2048;
static_assert(static_cast<std::size_t>(kFastPad) * sizeof(int) * kSortWarps <= kMaxSharedBytes);
static_assert(std::has_single_bit(static_cast<unsigned>(kFastPad)),
              "the pad is rounded to a power of two, so the class boundary must be one");

constexpr unsigned kFullWarp = 0xffffffffu;

// Inclusive prefix sum across a warp; the total is the last lane's. Every lane must reach it.
__device__ __forceinline__ int warp_scan(int v, int lane) {
#pragma unroll
    for (int d = 1; d < kWarp; d <<= 1) {
        const int t = __shfl_up_sync(kFullWarp, v, d);
        if (lane >= d) v += t;
    }
    return v;
}

__device__ __forceinline__ int warp_total(int scan) {
    return __shfl_sync(kFullWarp, scan, kWarp - 1);
}

// The warp's index across the grid, for the kernels that put one segment on each. Off blockDim
// rather than kSortWarps so the mapping holds whatever width the block was launched at.
__device__ __forceinline__ int warp_in_grid() {
    return blockIdx.x * static_cast<int>(blockDim.x / kWarp) +
           static_cast<int>(threadIdx.x) / kWarp;
}

// Keep the first of each equal run in a sorted list, by one warp. Whether the list was sorted in
// shared memory or by CUB in global memory, the compaction is the same, and so is the answer: a
// multiset has one sorted form and its run structure follows from it. Every lane must reach this.
template <bool Fill>
__device__ __forceinline__ int warp_keep_first_of_each_run(const int* b, int n, int* dst,
                                                           int lane) {
    int kept = 0;
    for (int base = 0; base < n; base += kWarp) {
        const int i = base + lane;
        const int v = (i < n) ? b[i] : 0;
        const int keep = (i < n && (i == 0 || v != b[i - 1])) ? 1 : 0;
        const int scan = warp_scan(keep, lane);
        if (Fill && keep) dst[kept + scan - keep] = v;
        kept += warp_total(scan);
    }
    return kept;
}

// --- gather policies --------------------------------------------------------------------------
//
// A policy answers `size`, how many candidates a segment has, and `load`, which writes exactly
// that many into the buffer it is given and returns the count. The two must agree: the buffer is
// reserved from `size`, and nothing bounds `load` again.

// Row i's columns are the distinct nodes of the tetrahedra incident to node i, so its candidates
// are the four nodes of every corner in its node2corner segment: ~90 of them for ~14 distinct
// columns on a head mesh. Each corner writes a fixed slot, so the layout needs no scan.
struct IncidentGather {
    const int* __restrict__ tet_nodes;
    const int* __restrict__ ptr;
    const int* __restrict__ idx;

    __device__ int size(int s) const { return (ptr[s + 1] - ptr[s]) * 4; }

    __device__ int load(int s, int* __restrict__ out, int lane) const {
        const int begin = ptr[s], end = ptr[s + 1];
        for (int c = begin + lane; c < end; c += kWarp) {
            // A 16-byte load: tet_nodes is one contiguous 4 * n_tet block, so a tet's four ids
            // never straddle the boundary and the base is an allocation start, not a strided view.
            const int4 t = *reinterpret_cast<const int4*>(
                tet_nodes + static_cast<std::int64_t>(idx[c] >> 2) * 4);
            int* const slot = out + (c - begin) * 4;
            slot[0] = t.x;
            slot[1] = t.y;
            slot[2] = t.z;
            slot[3] = t.w;
        }
        return size(s);
    }
};

// A slot whose own tetrahedra reach at least min_nodes nodes keeps that first ring. A smaller one
// grows to the union of its neighbours' first rings, which is the same set as re-walking the
// same-tissue tetrahedra around it and needs no second pass over the mesh. The slot is its own
// neighbour, so the grown patch always contains the first ring. The union is where the duplication
// the sort exists to remove comes from: on a head mesh 2.8 candidates per distinct patch node.
struct PatchGather {
    const int* __restrict__ r1_ptr;
    const int* __restrict__ r1_idx;
    const int* __restrict__ neighbour;
    int min_nodes;

    __device__ int size(int s) const {
        const int begin = r1_ptr[s], end = r1_ptr[s + 1];
        if (end - begin >= min_nodes) return end - begin;
        int total = 0;
        for (int j = begin; j < end; ++j) {
            const int nb = neighbour[j];
            total += r1_ptr[nb + 1] - r1_ptr[nb];
        }
        return total;
    }

    __device__ int load(int s, int* __restrict__ out, int lane) const {
        const int begin = r1_ptr[s], end = r1_ptr[s + 1];
        const int ring = end - begin;
        if (ring >= min_nodes) {
            for (int i = lane; i < ring; i += kWarp) out[i] = r1_idx[begin + i];
            return ring;
        }
        // Runs of unequal length, so each lane needs where its neighbour's ring starts: a warp
        // scan of the lengths, chunk by chunk.
        int n = 0;
        for (int base = 0; base < ring; base += kWarp) {
            const int j = base + lane;
            int len = 0, src = 0;
            if (j < ring) {
                const int nb = neighbour[begin + j];
                src = r1_ptr[nb];
                len = r1_ptr[nb + 1] - src;
            }
            const int scan = warp_scan(len, lane);
            const int chunk = warp_total(scan);
            const int at = n + scan - len;
            for (int t = 0; t < len; ++t) out[at + t] = r1_idx[src + t];
            n += chunk;
        }
        return n;
    }
};

// --- the plan ---------------------------------------------------------------------------------

// The shared class's width, and how many segments fall outside it. Max and add are both
// order-independent, so the atomics cost nothing in reproducibility. Values across a warp are
// near-identical, which is the case the hardware's atomic coalescing handles well; a block
// reduction ahead of it measured slower than the writes it saved.
//
// kFastPad is a power of two, so `bit_ceil(n) > kFastPad` is just `n > kFastPad`. Both passes
// classify from the same `size`, so both reach the same split from the same input.
template <typename Gather>
__global__ void plan_kernel(Gather g, int* __restrict__ stats, int n_seg) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seg) return;
    const int n = g.size(s);
    if (n > kFastPad) {
        atomicAdd(stats + 1, 1);
    } else {
        atomicMax(stats + 0, n);
    }
}

// The wide segments' ids and candidate counts, in whatever order the atomic hands out. Position in
// these lists picks only which scratch slot a segment borrows; every segment still writes its own
// counts entry and its own slice of out_idx, so the order cannot reach the output.
template <typename Gather>
__global__ void compact_wide_kernel(Gather g, int* __restrict__ cursor, int* __restrict__ ids,
                                    int* __restrict__ sizes, int n_seg) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seg) return;
    const int n = g.size(s);
    if (n <= kFastPad) return;
    const int slot = atomicAdd(cursor, 1);
    ids[slot] = s;
    sizes[slot] = n;
}

// How the segments split: the shared class's padded width, and the wide segments with the candidate
// offsets a segmented sort needs. The width is a max over a class and the membership test is per
// segment, both order-independent, so the sizing and filling passes reach the same plan from the
// same input.
struct SegmentPlan {
    int fast_pad = 1;
    int n_wide = 0;
    int n_cand = 0;
    DeviceBuffer<int> wide_ids;
    DeviceBuffer<int> wide_off;
};

template <typename Gather>
SegmentPlan plan_segments(Gather g, int n_seg, cudaStream_t stream) {
    DeviceBuffer<int> stats = device_alloc<int>(2, "pattern", "segment plan");
    check_cuda(cudaMemsetAsync(stats.get(), 0, 2 * sizeof(int), stream), "memset(segment plan)");
    plan_kernel<<<grid_for(n_seg), kBlock, 0, stream>>>(g, stats.get(), n_seg);
    check_cuda(cudaGetLastError(), "segment plan launch");

    int host[2] = {0, 0};
    check_cuda(cudaMemcpyAsync(host, stats.get(), sizeof(host), cudaMemcpyDeviceToHost, stream),
               "copy(segment plan)");
    check_cuda(cudaStreamSynchronize(stream), "sync(segment plan)");

    SegmentPlan plan;
    plan.fast_pad = static_cast<int>(std::bit_ceil(static_cast<unsigned>(host[0])));
    plan.n_wide = host[1];
    // A mesh of ordinary valence stops here, and none of the wide class runs at all.
    if (!plan.n_wide) return plan;

    const std::size_t wide = static_cast<std::size_t>(plan.n_wide);
    plan.wide_ids = device_alloc<int>(wide, "pattern", "wide segment ids");
    plan.wide_off = device_alloc<int>(wide + 1, "pattern", "wide segment offsets");
    DeviceBuffer<int> sizes = device_alloc<int>(wide + 1, "pattern", "wide segment sizes");
    DeviceBuffer<int> cursor = device_alloc<int>(1, "pattern", "wide segment cursor");
    check_cuda(cudaMemsetAsync(cursor.get(), 0, sizeof(int), stream), "memset(wide cursor)");
    compact_wide_kernel<<<grid_for(n_seg), kBlock, 0, stream>>>(
        g, cursor.get(), plan.wide_ids.get(), sizes.get(), n_seg);
    check_cuda(cudaGetLastError(), "wide segment compaction launch");
    check_cuda(cudaMemsetAsync(sizes.get() + plan.n_wide, 0, sizeof(int), stream),
               "memset(wide total)");

    std::size_t bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, bytes, sizes.get(), plan.wide_off.get(),
                                             plan.n_wide + 1, stream),
               "size(wide exclusive sum)");
    DeviceBuffer<std::byte> temp = device_alloc<std::byte>(bytes, "pattern", "wide scan scratch");
    check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), bytes, sizes.get(), plan.wide_off.get(),
                                             plan.n_wide + 1, stream),
               "wide exclusive sum");

    check_cuda(cudaMemcpyAsync(&plan.n_cand, plan.wide_off.get() + plan.n_wide, sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "copy(wide candidate total)");
    // The scratch above dies with this scope, so the kernels have to have finished reading it.
    check_cuda(cudaStreamSynchronize(stream), "sync(wide candidate total)");
    return plan;
}

// --- the shared class -------------------------------------------------------------------------

// One segment, by one warp, into the buffer `b` the caller chose. Everything that decides the
// row is here; the two launch classes differ only in where `b` points and how they enumerate
// segments, so neither can change what a row contains.
template <typename Gather, bool Fill>
__device__ __forceinline__ void segment_unique_one(Gather g, const int* __restrict__ out_ptr,
                                                   int* __restrict__ out_idx,
                                                   int* __restrict__ counts, int s, int* b,
                                                   int lane) {
    const int n = g.load(s, b, lane);
    if (n == 0) {
        if (lane == 0 && !Fill) counts[s] = 0;
        return;
    }

    // A bitonic network over the next power of two at or above n, not over pad: most segments stop
    // well short of the widest one, and the padding is what the network would otherwise sort.
    const int m = static_cast<int>(cuda::std::bit_ceil(static_cast<unsigned>(n)));
    // Node ids are non-negative, so the maximum pads to the tail and never displaces a candidate.
    for (int i = n + lane; i < m; i += kWarp) {
        b[i] = cuda::std::numeric_limits<int>::max();
    }
    __syncwarp();

    for (int k = 2; k <= m; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = lane; i < m; i += kWarp) {
                const int ixj = i ^ j;
                // Only the lower index of a pair acts, so exactly one lane writes each of the two
                // slots and the stores within a stage never collide. Both values are read into
                // registers first: this is the innermost loop of the sort, and re-reading shared
                // memory inside the comparison costs measurably.
                if (ixj > i) {
                    const int x = b[i], y = b[ixj];
                    if ((x > y) == ((i & k) == 0)) {
                        b[i] = y;
                        b[ixj] = x;
                    }
                }
            }
            __syncwarp();
        }
    }

    const int kept =
        warp_keep_first_of_each_run<Fill>(b, n, Fill ? out_idx + out_ptr[s] : nullptr, lane);
    if (lane == 0 && !Fill) counts[s] = kept;
}

// One warp per segment over the whole segment range, skipping what the wide class owns. With Fill
// the distinct values go to out_idx at the row start the sizing pass scanned; without it only the
// row's length is recorded.
template <typename Gather, bool Fill>
__global__ __launch_bounds__(kWarp * kSortWarps) void segment_unique_kernel(
    Gather g, const int* __restrict__ out_ptr, int* __restrict__ out_idx,
    int* __restrict__ counts, int n_seg, int pad) {
    extern __shared__ int scratch[];

    const int lane = threadIdx.x % kWarp;
    const int s = warp_in_grid();
    if (s >= n_seg || g.size(s) > kFastPad) return;
    int* const b = scratch + (threadIdx.x / kWarp) * pad;
    segment_unique_one<Gather, Fill>(g, out_ptr, out_idx, counts, s, b, lane);
}

// kSortWarps warps a block, one segment each, over the whole segment range. kFastPad is chosen so
// this allocation always fits, so occupancy here never depends on what the widest segment in the
// mesh happens to be.
template <bool Fill, typename Gather>
void launch_fast(Gather g, const int* out_ptr, int* out_idx, int* counts, int n_seg, int pad,
                 cudaStream_t stream) {
    const std::size_t shared = static_cast<std::size_t>(pad) * sizeof(int) * kSortWarps;
    const unsigned grid = grid_for(n_seg, kSortWarps);
    if (!grid) return;
    segment_unique_kernel<Gather, Fill><<<grid, kWarp * kSortWarps, shared, stream>>>(
        g, out_ptr, out_idx, counts, n_seg, pad);
    check_cuda(cudaGetLastError(), "segment unique launch");
}

// --- the wide class ---------------------------------------------------------------------------
//
// Segments too wide for shared memory: their candidates are materialised in global memory, sorted
// by CUB as one segmented problem, and compacted a warp per segment.

template <typename Gather>
__global__ __launch_bounds__(kWarp * kSortWarps) void gather_wide_kernel(
    Gather g, const int* __restrict__ seg_ids, const int* __restrict__ off,
    int* __restrict__ cand, int n_wide) {
    const int lane = threadIdx.x % kWarp;
    const int w = warp_in_grid();
    if (w >= n_wide) return;
    g.load(seg_ids[w], cand + off[w], lane);
}

template <bool Fill>
__global__ __launch_bounds__(kWarp * kSortWarps) void dedupe_wide_kernel(
    const int* __restrict__ sorted, const int* __restrict__ off,
    const int* __restrict__ seg_ids, const int* __restrict__ out_ptr, int* __restrict__ out_idx,
    int* __restrict__ counts, int n_wide) {
    const int lane = threadIdx.x % kWarp;
    const int w = warp_in_grid();
    if (w >= n_wide) return;
    const int s = seg_ids[w];
    const int kept = warp_keep_first_of_each_run<Fill>(
        sorted + off[w], off[w + 1] - off[w], Fill ? out_idx + out_ptr[s] : nullptr, lane);
    if (lane == 0 && !Fill) counts[s] = kept;
}

// The buffers here are two ints per candidate of the wide segments only, so they are bounded by how
// much of the mesh is pathological rather than by the mesh.
template <bool Fill, typename Gather>
void launch_wide(Gather g, const int* out_ptr, int* out_idx, int* counts, const SegmentPlan& plan,
                 cudaStream_t stream) {
    if (!plan.n_wide) return;
    const int n_wide = plan.n_wide;
    const int* seg_ids = plan.wide_ids.get();
    const int* off = plan.wide_off.get();
    const unsigned grid = grid_for(n_wide, kSortWarps);
    const std::size_t n_cand = static_cast<std::size_t>(plan.n_cand);

    DeviceBuffer<int> cand = device_alloc<int>(n_cand, "pattern", "wide candidates");
    DeviceBuffer<int> sorted = device_alloc<int>(n_cand, "pattern", "wide sorted candidates");
    gather_wide_kernel<<<grid, kWarp * kSortWarps, 0, stream>>>(g, seg_ids, off, cand.get(),
                                                                n_wide);
    check_cuda(cudaGetLastError(), "wide segment gather launch");

    std::size_t bytes = 0;
    check_cuda(cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, cand.get(), sorted.get(),
                                                  plan.n_cand, n_wide, off, off + 1, stream),
               "size(wide segmented sort)");
    DeviceBuffer<std::byte> temp = device_alloc<std::byte>(bytes, "pattern", "wide sort scratch");
    check_cuda(cub::DeviceSegmentedSort::SortKeys(temp.get(), bytes, cand.get(), sorted.get(),
                                                  plan.n_cand, n_wide, off, off + 1, stream),
               "wide segmented sort");

    dedupe_wide_kernel<Fill><<<grid, kWarp * kSortWarps, 0, stream>>>(
        sorted.get(), off, seg_ids, out_ptr, out_idx, counts, n_wide);
    check_cuda(cudaGetLastError(), "wide segment dedupe launch");
    // Every buffer above dies with this call, so the kernels have to have finished reading them.
    check_cuda(cudaStreamSynchronize(stream), "sync(wide segment scratch)");
}

// Distinct counts per segment, exclusive-scanned into out_ptr. Returns the total, which is the
// length the caller allocates the column index at.
template <typename Gather>
int count_rows(Gather g, int* out_ptr, int n_seg, cudaStream_t stream) {
    if (n_seg <= 0) return 0;
    const SegmentPlan plan = plan_segments(g, n_seg, stream);
    DeviceBuffer<int> counts =
        device_alloc<int>(static_cast<std::size_t>(n_seg) + 1, "pattern", "row counts");
    launch_fast<false>(g, nullptr, nullptr, counts.get(), n_seg, plan.fast_pad, stream);
    launch_wide<false>(g, nullptr, nullptr, counts.get(), plan, stream);
    check_cuda(cudaMemsetAsync(counts.get() + n_seg, 0, sizeof(int), stream), "memset(total)");

    std::size_t bytes = 0;
    check_cuda(
        cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts.get(), out_ptr, n_seg + 1, stream),
        "size(exclusive sum)");
    DeviceBuffer<std::byte> temp = device_alloc<std::byte>(bytes, "pattern", "scan scratch");
    check_cuda(
        cub::DeviceScan::ExclusiveSum(temp.get(), bytes, counts.get(), out_ptr, n_seg + 1, stream),
        "exclusive sum");

    int nnz = 0;
    check_cuda(
        cudaMemcpyAsync(&nnz, out_ptr + n_seg, sizeof(int), cudaMemcpyDeviceToHost, stream),
        "copy(nnz)");
    check_cuda(cudaStreamSynchronize(stream), "sync(nnz)");
    return nnz;
}

template <typename Gather>
void fill_rows(Gather g, const int* out_ptr, int* out_idx, int n_seg, cudaStream_t stream) {
    if (n_seg <= 0) return;
    const SegmentPlan plan = plan_segments(g, n_seg, stream);
    launch_fast<true>(g, out_ptr, out_idx, nullptr, n_seg, plan.fast_pad, stream);
    launch_wide<true>(g, out_ptr, out_idx, nullptr, plan, stream);
}

}  // namespace

int count_incident_node_csr(const int* tet_nodes, const int* ptr, const int* idx, int* out_ptr,
                            int n_seg, cudaStream_t stream) {
    return count_rows(IncidentGather{tet_nodes, ptr, idx}, out_ptr, n_seg, stream);
}

void fill_incident_node_csr(const int* tet_nodes, const int* ptr, const int* idx,
                            const int* out_ptr, int* out_idx, int n_seg, cudaStream_t stream) {
    fill_rows(IncidentGather{tet_nodes, ptr, idx}, out_ptr, out_idx, n_seg, stream);
}

int count_patch_csr(const int* r1_ptr, const int* r1_idx, const int* neighbour, int min_nodes,
                    int* out_ptr, int n_slots, cudaStream_t stream) {
    return count_rows(PatchGather{r1_ptr, r1_idx, neighbour, min_nodes}, out_ptr, n_slots, stream);
}

void fill_patch_csr(const int* r1_ptr, const int* r1_idx, const int* neighbour, int min_nodes,
                    const int* out_ptr, int* out_idx, int n_slots, cudaStream_t stream) {
    fill_rows(PatchGather{r1_ptr, r1_idx, neighbour, min_nodes}, out_ptr, out_idx, n_slots,
              stream);
}
