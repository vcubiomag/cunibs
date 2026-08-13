#include "fem/fem.hpp"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

// Segment-wise CSR construction: gather a candidate list per segment, sort the segments, and keep
// the first of every equal run. Two operators are built this way -- the stiffness sparsity pattern
// and the recovery patches -- and they differ only in how the candidates are gathered.
//
// Sorting a multiset has one answer, so neither result depends on how CUB schedules the work.

namespace {

void check_cuda(cudaError_t err, const char* what) { ::check_cuda(err, "pattern", what); }

__global__ void scale4_kernel(const int* __restrict__ ptr, int* __restrict__ seg, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) seg[i] = ptr[i] * 4;
}

// One thread per corner, copying that corner's tetrahedron as a single 16-byte load: tet_nodes is
// one contiguous 4 * n_tet block, so a tet's four ids never straddle the boundary.
__global__ void gather_incident_kernel(const int* __restrict__ tet_nodes,
                                       const int* __restrict__ idx, int* __restrict__ cand,
                                       int n_corner) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_corner) return;
    reinterpret_cast<int4*>(cand)[p] =
        *reinterpret_cast<const int4*>(tet_nodes + static_cast<std::int64_t>(idx[p] >> 2) * 4);
}

__global__ void patch_counts_kernel(const int* __restrict__ r1_ptr,
                                    const int* __restrict__ neighbour, int min_nodes,
                                    int* __restrict__ counts, int n_slots) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    const int begin = r1_ptr[s], end = r1_ptr[s + 1];
    if (end - begin >= min_nodes) {
        counts[s] = end - begin;
        return;
    }
    int total = 0;
    for (int j = begin; j < end; ++j) {
        const int nb = neighbour[j];
        total += r1_ptr[nb + 1] - r1_ptr[nb];
    }
    counts[s] = total;
}

__global__ void patch_gather_kernel(const int* __restrict__ r1_ptr, const int* __restrict__ r1_idx,
                                    const int* __restrict__ neighbour, int min_nodes,
                                    const int* __restrict__ seg, int* __restrict__ cand,
                                    int n_slots) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    const int begin = r1_ptr[s], end = r1_ptr[s + 1];
    int w = seg[s];
    if (end - begin >= min_nodes) {
        for (int j = begin; j < end; ++j) cand[w++] = r1_idx[j];
        return;
    }
    for (int j = begin; j < end; ++j) {
        const int nb = neighbour[j];
        const int row_e = r1_ptr[nb + 1];
        for (int p = r1_ptr[nb]; p < row_e; ++p) cand[w++] = r1_idx[p];
    }
}

__global__ void count_distinct_kernel(const int* __restrict__ sorted, const int* __restrict__ seg,
                                      int* __restrict__ counts, int n_seg) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_seg) return;
    const int begin = seg[row], end = seg[row + 1];
    int n = 0;
    for (int p = begin; p < end; ++p) {
        if (p == begin || sorted[p] != sorted[p - 1]) ++n;
    }
    counts[row] = n;
}

__global__ void write_distinct_kernel(const int* __restrict__ sorted, const int* __restrict__ seg,
                                      const int* __restrict__ out_ptr, int* __restrict__ out_idx,
                                      int n_seg) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_seg) return;
    const int begin = seg[row], end = seg[row + 1];
    int w = out_ptr[row];
    for (int p = begin; p < end; ++p) {
        if (p == begin || sorted[p] != sorted[p - 1]) out_idx[w++] = sorted[p];
    }
}

// Shared tail: sort each segment of `cand`, then compact the distinct values back into it and
// leave the row offsets in `out_ptr`. `counts` is scratch of n_seg + 1 entries, whose last slot
// carries the zero the exclusive scan needs to deposit the total.
int sort_and_compact(int* cand, int* sorted, const int* seg, int* counts, int* out_ptr, int n_seg,
                     int n_cand, cudaStream_t stream) {
    std::size_t sort_bytes = 0;
    std::size_t scan_bytes = 0;
    check_cuda(cub::DeviceSegmentedSort::SortKeys(nullptr, sort_bytes, cand, sorted, n_cand,
                                                  n_seg, seg, seg + 1, stream),
               "size(segmented sort)");
    check_cuda(
        cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, counts, out_ptr, n_seg + 1, stream),
        "size(exclusive sum)");
    DeviceBuffer<std::byte> temp =
        device_alloc<std::byte>(std::max(sort_bytes, scan_bytes), "pattern", "cub scratch");

    check_cuda(cub::DeviceSegmentedSort::SortKeys(temp.get(), sort_bytes, cand, sorted, n_cand,
                                                  n_seg, seg, seg + 1, stream),
               "segmented sort");

    const unsigned blocks = grid_for(n_seg);
    count_distinct_kernel<<<blocks, kBlock, 0, stream>>>(sorted, seg, counts, n_seg);
    check_cuda(cudaMemsetAsync(counts + n_seg, 0, sizeof(int), stream), "memset(total)");
    check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), scan_bytes, counts, out_ptr, n_seg + 1,
                                             stream),
               "exclusive sum");
    // The compaction lands in the candidate buffer, which the sort has already consumed.
    write_distinct_kernel<<<blocks, kBlock, 0, stream>>>(sorted, seg, out_ptr, cand, n_seg);
    check_cuda(cudaGetLastError(), "compaction launch");

    int nnz = 0;
    check_cuda(
        cudaMemcpyAsync(&nnz, out_ptr + n_seg, sizeof(int), cudaMemcpyDeviceToHost, stream),
        "copy(nnz)");
    check_cuda(cudaStreamSynchronize(stream), "sync(nnz)");
    return nnz;
}

}  // namespace

// Row i's columns are the distinct nodes of the tetrahedra incident to node i, so its candidates
// are the four nodes of every corner in its node2corner segment: ~96 of them for ~14 distinct
// columns on a head mesh. 4 * ptr is already the exclusive prefix of those counts, so the
// candidate list needs no scan of its own.
int build_incident_node_csr(const int* tet_nodes, const int* ptr, const int* idx, int* cand,
                            int* sorted, int* out_ptr, int n_seg, int n_corner,
                            cudaStream_t stream) {
    if (n_seg <= 0) return 0;
    const std::int64_t n_cand = static_cast<std::int64_t>(n_corner) * 4;
    if (n_cand > static_cast<std::int64_t>(INT32_MAX)) {
        throw std::invalid_argument("build_incident_node_csr: candidate list exceeds int range");
    }

    DeviceBuffer<int> seg =
        device_alloc<int>(static_cast<std::size_t>(n_seg) + 1, "pattern", "segment offsets");
    DeviceBuffer<int> counts =
        device_alloc<int>(static_cast<std::size_t>(n_seg) + 1, "pattern", "row counts");
    scale4_kernel<<<grid_for(n_seg + 1), kBlock, 0, stream>>>(ptr, seg.get(), n_seg + 1);
    if (const unsigned blocks = grid_for(n_corner)) {
        gather_incident_kernel<<<blocks, kBlock, 0, stream>>>(tet_nodes, idx, cand, n_corner);
    }
    check_cuda(cudaGetLastError(), "candidate launch");

    return sort_and_compact(cand, sorted, seg.get(), counts.get(), out_ptr, n_seg,
                            static_cast<int>(n_cand), stream);
}

// A slot whose own tetrahedra reach at least min_nodes nodes keeps that first ring. A smaller one
// grows to the union of its neighbours' first rings, which is the same set as re-walking the
// same-tissue tetrahedra around it and needs no second pass over the mesh. The slot is its own
// neighbour, so the grown patch always contains the first ring.
int build_patch_csr(const int* r1_ptr, const int* r1_idx, const int* neighbour, int min_nodes,
                    int* cand, int* sorted, int* out_ptr, int n_slots, int n_cand,
                    cudaStream_t stream) {
    if (n_slots <= 0) return 0;
    DeviceBuffer<int> seg =
        device_alloc<int>(static_cast<std::size_t>(n_slots) + 1, "pattern", "segment offsets");
    DeviceBuffer<int> counts =
        device_alloc<int>(static_cast<std::size_t>(n_slots) + 1, "pattern", "row counts");

    const unsigned blocks = grid_for(n_slots);
    patch_counts_kernel<<<blocks, kBlock, 0, stream>>>(r1_ptr, neighbour, min_nodes, counts.get(),
                                                       n_slots);
    check_cuda(cudaMemsetAsync(counts.get() + n_slots, 0, sizeof(int), stream), "memset(total)");
    std::size_t scan_bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, counts.get(), seg.get(),
                                             n_slots + 1, stream),
               "size(candidate scan)");
    {
        DeviceBuffer<std::byte> temp =
            device_alloc<std::byte>(scan_bytes, "pattern", "candidate scan scratch");
        check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), scan_bytes, counts.get(), seg.get(),
                                                 n_slots + 1, stream),
                   "candidate scan");
    }

    int total = 0;
    check_cuda(cudaMemcpyAsync(&total, seg.get() + n_slots, sizeof(int), cudaMemcpyDeviceToHost,
                               stream),
               "copy(candidate total)");
    check_cuda(cudaStreamSynchronize(stream), "sync(candidate total)");
    if (total > n_cand) {
        throw std::invalid_argument("build_patch_csr: work buffers too small for the candidates");
    }

    patch_gather_kernel<<<blocks, kBlock, 0, stream>>>(r1_ptr, r1_idx, neighbour, min_nodes,
                                                       seg.get(), cand, n_slots);
    check_cuda(cudaGetLastError(), "patch gather launch");

    return sort_and_compact(cand, sorted, seg.get(), counts.get(), out_ptr, n_slots, total,
                            stream);
}
