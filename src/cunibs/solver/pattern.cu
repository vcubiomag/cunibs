#include "kernels.hpp"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

// CSR sparsity pattern of the P1 stiffness matrix, built from the node2corner incidence map.
//
// Row i's columns are the distinct nodes of the tetrahedra incident to node i, so its candidates
// are the four nodes of every corner in its node2corner segment: ~96 of them for ~14 distinct
// columns on a head mesh. 4 * ptr is already the exclusive prefix of those counts, so the
// candidate list needs no scan of its own.
//
// Each row is then a CUB sort segment, and keeping the first of every equal run leaves the
// columns sorted and distinct. Sorting a multiset has one answer, so the pattern does not depend
// on how CUB schedules the work.

namespace {

void check_cuda(cudaError_t err, const char* what) { ::check_cuda(err, "pattern", what); }

__global__ void segment_offsets_kernel(const int* __restrict__ ptr, int* __restrict__ seg,
                                       int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) seg[i] = ptr[i] * 4;
}

// One thread per corner, copying that corner's tetrahedron as a single 16-byte load: tet_nodes is
// one contiguous 4 * n_tet block, so a tet's four ids never straddle the boundary.
__global__ void gather_candidates_kernel(const int* __restrict__ tet_nodes,
                                         const int* __restrict__ idx, int* __restrict__ cand,
                                         int n_corner) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_corner) return;
    reinterpret_cast<int4*>(cand)[p] =
        *reinterpret_cast<const int4*>(tet_nodes + static_cast<std::int64_t>(idx[p] >> 2) * 4);
}

__global__ void count_distinct_kernel(const int* __restrict__ sorted, const int* __restrict__ seg,
                                      int* __restrict__ counts, int n_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const int begin = seg[row], end = seg[row + 1];
    int n = 0;
    for (int p = begin; p < end; ++p) {
        if (p == begin || sorted[p] != sorted[p - 1]) ++n;
    }
    counts[row] = n;
}

__global__ void write_distinct_kernel(const int* __restrict__ sorted, const int* __restrict__ seg,
                                      const int* __restrict__ indptr, int* __restrict__ indices,
                                      int n_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const int begin = seg[row], end = seg[row + 1];
    int w = indptr[row];
    for (int p = begin; p < end; ++p) {
        if (p == begin || sorted[p] != sorted[p - 1]) indices[w++] = sorted[p];
    }
}

}  // namespace

int build_stiffness_pattern(const int* tet_nodes, const int* ptr, const int* idx, int* cand,
                            int* sorted, int* indptr, int n_rows, int n_tet,
                            cudaStream_t stream) {
    if (n_rows <= 0) return 0;
    const std::int64_t n_corner = static_cast<std::int64_t>(n_tet) * 4;
    if (n_corner * 4 > static_cast<std::int64_t>(INT32_MAX)) {
        throw std::invalid_argument("build_stiffness_pattern: candidate list exceeds int range");
    }

    // seg[i] = 4 * ptr[i], the candidate segment bounds, and one extra slot past the rows so the
    // exclusive scan below can deposit the total.
    DeviceBuffer<int> seg = device_alloc<int>(static_cast<std::size_t>(n_rows) + 1, "pattern",
                                             "segment offsets");
    DeviceBuffer<int> counts =
        device_alloc<int>(static_cast<std::size_t>(n_rows) + 1, "pattern", "row counts");
    segment_offsets_kernel<<<grid_for(n_rows + 1), kBlock, 0, stream>>>(ptr, seg.get(),
                                                                        n_rows + 1);
    if (const unsigned blocks = grid_for(n_corner)) {
        gather_candidates_kernel<<<blocks, kBlock, 0, stream>>>(tet_nodes, idx, cand,
                                                                static_cast<int>(n_corner));
    }
    check_cuda(cudaGetLastError(), "candidate launch");

    std::size_t sort_bytes = 0;
    std::size_t scan_bytes = 0;
    check_cuda(cub::DeviceSegmentedSort::SortKeys(nullptr, sort_bytes, cand, sorted,
                                                  static_cast<int>(n_corner * 4), n_rows,
                                                  seg.get(), seg.get() + 1, stream),
               "size(segmented sort)");
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, counts.get(), indptr,
                                             n_rows + 1, stream),
               "size(exclusive sum)");
    DeviceBuffer<std::byte> temp =
        device_alloc<std::byte>(std::max(sort_bytes, scan_bytes), "pattern", "cub scratch");

    check_cuda(cub::DeviceSegmentedSort::SortKeys(temp.get(), sort_bytes, cand, sorted,
                                                  static_cast<int>(n_corner * 4), n_rows,
                                                  seg.get(), seg.get() + 1, stream),
               "segmented sort");

    const unsigned rows = grid_for(n_rows);
    count_distinct_kernel<<<rows, kBlock, 0, stream>>>(sorted, seg.get(), counts.get(), n_rows);
    check_cuda(cudaMemsetAsync(counts.get() + n_rows, 0, sizeof(int), stream), "memset(total)");
    check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), scan_bytes, counts.get(), indptr,
                                             n_rows + 1, stream),
               "exclusive sum");
    // The compaction lands in the candidate buffer, which the sort has already consumed.
    write_distinct_kernel<<<rows, kBlock, 0, stream>>>(sorted, seg.get(), indptr, cand, n_rows);
    check_cuda(cudaGetLastError(), "compaction launch");

    int nnz = 0;
    check_cuda(cudaMemcpyAsync(&nnz, indptr + n_rows, sizeof(int), cudaMemcpyDeviceToHost, stream),
               "copy(nnz)");
    check_cuda(cudaStreamSynchronize(stream), "sync(nnz)");
    return nnz;
}
