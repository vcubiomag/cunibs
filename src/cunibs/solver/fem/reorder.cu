#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cuda/std/algorithm>

// Support for laying a mesh out along a Morton curve. The node permutation itself is a sort, so it
// stays on the Python side; what needs a kernel is the key the tetrahedra are then sorted by.

namespace {

// One thread per tetrahedron, reading the key through the permutation so the connectivity is
// never renumbered just to be reduced; the permutation gathered from here is a few megabytes and
// stays in L2 across the sweep.
__global__ void tet_lowest_node_kernel(const int* __restrict__ inverse,
                                       Tet4View<const int> tet_nodes, int* __restrict__ lowest,
                                       int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;
    // One 16-byte load of the row: the view is over one contiguous 4 * n_tet block, so a tet's
    // four ids never straddle the boundary.
    const int4 t = *reinterpret_cast<const int4*>(&tet_nodes(e, 0));
    lowest[e] = cuda::std::min(cuda::std::min(inverse[t.x], inverse[t.y]),
                               cuda::std::min(inverse[t.z], inverse[t.w]));
}

}  // namespace

void launch_tet_lowest_node(const int* inverse, const int* tet_nodes, int* lowest, int n_tet,
                            cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_tet)) {
        tet_lowest_node_kernel<<<blocks, kBlock, 0, stream>>>(
            inverse, Tet4View<const int>(tet_nodes, n_tet), lowest, n_tet);
        check_cuda(cudaGetLastError(), "reorder", "tet lowest node launch");
    }
}
