#include "kernels.hpp"

#include <cstdint>

// One thread per row, the same shape rhs.cu uses for the node gather and for the same reason:
// it removes the atomics and fixes the summation order.
//   A[i,j] = Σ_{(e,i_loc): tet_nodes[e,i_loc]=i}  vols[e] · cond[e] · (g[e,i_loc] · g[e,j_loc])
// summed over the four j_loc of each incident element.
//
// node2corner fixes the order: its stable sort pins which element contributes when, then j_loc
// ascending within an element. A sparse merge would instead sum each entry in whatever order the
// work happened to be scheduled in, which is not reproducible between two assemblies of the same
// mesh.

namespace {

// The column list of a row is sorted (the caller canonicalises the pattern), so a contribution
// finds its slot by bisection rather than by scanning.
__device__ __forceinline__ int find_col(const int* __restrict__ indices, int lo, int hi, int col) {
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (indices[mid] < col) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void stiffness_rows_kernel(const double* __restrict__ g,
                                      const double* __restrict__ scale,
                                      const int* __restrict__ tet_nodes,
                                      const int* __restrict__ ptr, const int* __restrict__ idx,
                                      const int* __restrict__ indptr,
                                      const int* __restrict__ indices, double* __restrict__ data,
                                      int n_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int row_begin = indptr[row];
    const int row_end = indptr[row + 1];
    for (int p = row_begin; p < row_end; ++p) data[p] = 0.0;

    const int begin = ptr[row];
    const int end = ptr[row + 1];
    for (int p = begin; p < end; ++p) {
        const int corner = idx[p];
        const int e = corner >> 2;
        const double s = scale[e];
        const double gi0 = g[static_cast<std::int64_t>(corner) * 3 + 0];
        const double gi1 = g[static_cast<std::int64_t>(corner) * 3 + 1];
        const double gi2 = g[static_cast<std::int64_t>(corner) * 3 + 2];
        const std::int64_t ebase = static_cast<std::int64_t>(e) * 4;
        for (int j = 0; j < 4; ++j) {
            const std::int64_t cj = ebase + j;
            const double dot = gi0 * g[cj * 3 + 0] + gi1 * g[cj * 3 + 1] + gi2 * g[cj * 3 + 2];
            const int col = tet_nodes[cj];
            data[find_col(indices, row_begin, row_end, col)] += s * dot;
        }
    }
}

}  // namespace

void launch_stiffness_rows(const double* g, const double* scale, const int* tet_nodes,
                           const int* ptr, const int* idx, const int* indptr, const int* indices,
                           double* data, int n_rows, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_rows)) {
        stiffness_rows_kernel<<<blocks, kBlock, 0, stream>>>(g, scale, tet_nodes, ptr, idx, indptr,
                                                             indices, data, n_rows);
    }
}
