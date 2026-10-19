#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cuda/std/algorithm>

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

    const auto gv = unsized<Vec3View<const double, std::int64_t>>(g);
    const auto tn = unsized<Tet4View<const int, std::int64_t>>(tet_nodes);

    const int begin = ptr[row];
    const int end = ptr[row + 1];
    for (int p = begin; p < end; ++p) {
        const int corner = idx[p];
        const int e = corner >> 2;
        const double s = scale[e];
        const double gi0 = gv(corner, 0);
        const double gi1 = gv(corner, 1);
        const double gi2 = gv(corner, 2);
        for (int j = 0; j < 4; ++j) {
            const std::int64_t cj = static_cast<std::int64_t>(e) * 4 + j;
            const double dot = gi0 * gv(cj, 0) + gi1 * gv(cj, 1) + gi2 * gv(cj, 2);
            const int col = tn(e, j);
            // The row's columns are sorted, since the caller canonicalises the pattern.
            data[cuda::std::lower_bound(indices + row_begin, indices + row_end, col) - indices] +=
                s * dot;
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
