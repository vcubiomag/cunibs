#include "coil/coil.hpp"
#include "core/device_math.cuh"

namespace {

__global__ void dadt_element_average_kernel(const float* __restrict__ dadt_nodes,
                                            const int* __restrict__ tet_nodes,
                                            float* __restrict__ dadt_elm, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    const Vec3View<const float> nodal(dadt_nodes, kUnsizedRows);
    const Tet4View<const int> tn(tet_nodes, n_tet);
    const Vec3View<float> elm(dadt_elm, n_tet);

    const int n0 = tn(e, 0), n1 = tn(e, 1), n2 = tn(e, 2), n3 = tn(e, 3);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        elm(e, c) = 0.25f * (nodal(n0, c) + nodal(n1, c) + nodal(n2, c) + nodal(n3, c));
    }
}

}  // namespace

void launch_dadt_element_average(const float* dadt_nodes, const int* tet_nodes, float* dadt_elm,
                                 int n_tet, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_tet)) {
        dadt_element_average_kernel<<<blocks, kBlock, 0, stream>>>(dadt_nodes, tet_nodes,
                                                                   dadt_elm, n_tet);
    }
}
