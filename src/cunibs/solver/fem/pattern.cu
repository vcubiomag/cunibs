#include "fem/fem.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda/std/bit>
#include <cuda/std/limits>

#include <bit>
#include <cstdint>
#include <stdexcept>

// Segment-wise CSR construction: for every segment, the sorted distinct values of a candidate list
// it generates for itself. Two operators are built this way -- the stiffness sparsity pattern and
// the recovery patches -- and they differ only in how the candidates are gathered, which is all a
// gather policy below says.
//
// The candidates never reach global memory. A segment is one node's incident corners or one
// patch's neighbour rings: a few hundred entries against a segment count in the millions, so it
// fits in shared memory and one warp sorts it there in a single pass. The price is that each
// builder runs twice, once to size its rows and once to fill them, since nothing knows how many
// distinct values a segment has until it has sorted it.
//
// Sorting a multiset has one answer and duplicates are dropped by keeping the first of each equal
// run, so neither pass depends on how the work was scheduled.

namespace {

void check_cuda(cudaError_t err, const char* what) { ::check_cuda(err, "pattern", what); }

constexpr int kSortWarps = 4;
// Static shared memory a block may claim without opting in to the dynamic limit.
constexpr std::size_t kMaxSharedBytes = 48 * 1024;
// A segment's padded candidate list has to fit in shared memory, and the padding is to a power of
// two, so this is the largest such width one warp per block can hold. It corresponds to a node of
// two thousand incident elements; the widest segment on a real head is a few hundred candidates.
constexpr int kMaxPad = 8192;
static_assert(static_cast<std::size_t>(kMaxPad) * sizeof(int) <= kMaxSharedBytes);
static_assert(std::has_single_bit(static_cast<unsigned>(kMaxPad)),
              "the pad is rounded to a power of two, so its cap must be one");

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

// --- gather policies --------------------------------------------------------------------------
//
// A policy answers `size`, how many candidates a segment has, and `load`, which writes exactly
// that many into a warp's shared buffer and returns the count. The two must agree: the buffer is
// reserved from the largest `size` over all segments, and nothing bounds `load` again.

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

// --- the shared pass --------------------------------------------------------------------------

// Max is order-independent, so the atomic costs nothing in reproducibility. Values across a warp
// are near-identical, which is the case the hardware's atomic coalescing handles well; a block
// reduction ahead of it measured slower than the writes it saved.
template <typename Gather>
__global__ void widest_segment_kernel(Gather g, int* __restrict__ widest, int n_seg) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seg) return;
    atomicMax(widest, g.size(s));
}

// One warp per segment: gather into shared memory, sort it there, and keep the first of each equal
// run. With Fill the distinct values go to out_idx at the row start the sizing pass scanned;
// without it only the row's length is recorded.
template <typename Gather, bool Fill>
__global__ __launch_bounds__(kWarp * kSortWarps) void segment_unique_kernel(
    Gather g, const int* __restrict__ out_ptr, int* __restrict__ out_idx,
    int* __restrict__ counts, int n_seg, int pad) {
    extern __shared__ int scratch[];

    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    // Off blockDim, not kSortWarps: a segment wide enough to crowd shared memory is launched one
    // warp per block, and the mapping has to follow.
    const int s = blockIdx.x * static_cast<int>(blockDim.x / kWarp) + warp;
    if (s >= n_seg) return;
    int* const b = scratch + warp * pad;

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

    int* const dst = Fill ? out_idx + out_ptr[s] : nullptr;
    int kept = 0;
    for (int base = 0; base < n; base += kWarp) {
        const int i = base + lane;
        const int v = (i < n) ? b[i] : 0;
        const int keep = (i < n && (i == 0 || v != b[i - 1])) ? 1 : 0;
        const int scan = warp_scan(keep, lane);
        if (Fill && keep) dst[kept + scan - keep] = v;
        kept += warp_total(scan);
    }
    if (lane == 0 && !Fill) counts[s] = kept;
}

// The padded width every warp reserves, rounded up from the widest segment there is. A max over
// the segments is order-independent, so both passes reach the same width from the same input.
template <typename Gather>
int shared_pad(Gather g, int n_seg, cudaStream_t stream) {
    DeviceBuffer<int> widest = device_alloc<int>(1, "pattern", "widest segment");
    check_cuda(cudaMemsetAsync(widest.get(), 0, sizeof(int), stream), "memset(widest segment)");
    widest_segment_kernel<<<grid_for(n_seg), kBlock, 0, stream>>>(g, widest.get(), n_seg);
    check_cuda(cudaGetLastError(), "widest segment launch");

    int host = 0;
    check_cuda(cudaMemcpyAsync(&host, widest.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
               "copy(widest segment)");
    check_cuda(cudaStreamSynchronize(stream), "sync(widest segment)");

    const int pad = static_cast<int>(std::bit_ceil(static_cast<unsigned>(host)));
    if (pad > kMaxPad) {
        throw std::invalid_argument(
            "pattern: a segment of " + std::to_string(host) +
            " candidates does not fit in shared memory; the mesh has a node of implausible "
            "valence");
    }
    return pad;
}

template <bool Fill, typename Gather>
void launch_pass(Gather g, const int* out_ptr, int* out_idx, int* counts, int n_seg, int pad,
                 cudaStream_t stream) {
    // A wide segment gets a block to itself rather than a bigger shared allocation than the
    // hardware will give a block.
    const std::size_t per_warp = static_cast<std::size_t>(pad) * sizeof(int);
    const int warps = (per_warp * kSortWarps <= kMaxSharedBytes) ? kSortWarps : 1;
    const unsigned grid = grid_for(n_seg, warps);
    if (!grid) return;
    segment_unique_kernel<Gather, Fill><<<grid, kWarp * warps, per_warp * warps, stream>>>(
        g, out_ptr, out_idx, counts, n_seg, pad);
    check_cuda(cudaGetLastError(), "segment unique launch");
}

// Distinct counts per segment, exclusive-scanned into out_ptr. Returns the total, which is the
// length the caller allocates the column index at.
template <typename Gather>
int count_rows(Gather g, int* out_ptr, int n_seg, cudaStream_t stream) {
    if (n_seg <= 0) return 0;
    const int pad = shared_pad(g, n_seg, stream);
    DeviceBuffer<int> counts =
        device_alloc<int>(static_cast<std::size_t>(n_seg) + 1, "pattern", "row counts");
    launch_pass<false>(g, nullptr, nullptr, counts.get(), n_seg, pad, stream);
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
    const int pad = shared_pad(g, n_seg, stream);
    launch_pass<true>(g, out_ptr, out_idx, nullptr, n_seg, pad, stream);
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
