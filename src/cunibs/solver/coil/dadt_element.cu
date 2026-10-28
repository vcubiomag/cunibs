#include "coil/coil.hpp"
#include "core/device_math.cuh"

namespace {

// The average is corner_mean, which reconstruct.cu's nodal kernels also take, so an element field
// built here is the same bits as one those kernels form in place.
__global__ void dadt_element_average_kernel(const float* __restrict__ dadt_nodes,
                                            const int* __restrict__ tet_nodes,
                                            float* __restrict__ dadt_elm, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;
    const cuda::std::array<int, 4> tn = load_tet_nodes(Tet4View<const int>(tet_nodes, n_tet), e);
    const auto v = unsized<Vec3View<const float>>(dadt_nodes);
    const Vec3View<float> elm(dadt_elm, n_tet);
#pragma unroll
    for (int d = 0; d < 3; ++d) {
        elm(e, d) = corner_mean(v(tn[0], d), v(tn[1], d), v(tn[2], d), v(tn[3], d));
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
