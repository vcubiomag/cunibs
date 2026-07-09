#include "kernels.hpp"

#include <cuda_runtime.h>

namespace {

constexpr int kBlock = 256;

__global__ void dadt_element_average_kernel(const float* __restrict__ dadt_nodes,
                                            const int* __restrict__ tet_nodes,
                                            float* __restrict__ dadt_elm, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    const int n0 = tet_nodes[e * 4 + 0] * 3;
    const int n1 = tet_nodes[e * 4 + 1] * 3;
    const int n2 = tet_nodes[e * 4 + 2] * 3;
    const int n3 = tet_nodes[e * 4 + 3] * 3;
    const int out = e * 3;

    dadt_elm[out + 0] =
        0.25f * (dadt_nodes[n0 + 0] + dadt_nodes[n1 + 0] + dadt_nodes[n2 + 0] + dadt_nodes[n3 + 0]);
    dadt_elm[out + 1] =
        0.25f * (dadt_nodes[n0 + 1] + dadt_nodes[n1 + 1] + dadt_nodes[n2 + 1] + dadt_nodes[n3 + 1]);
    dadt_elm[out + 2] =
        0.25f * (dadt_nodes[n0 + 2] + dadt_nodes[n1 + 2] + dadt_nodes[n2 + 2] + dadt_nodes[n3 + 2]);
}

}

void launch_dadt_element_average(const float* dadt_nodes, const int* tet_nodes, float* dadt_elm,
                                 int n_tet, cudaStream_t stream) {
    const int blocks = (n_tet + kBlock - 1) / kBlock;
    dadt_element_average_kernel<<<blocks, kBlock, 0, stream>>>(dadt_nodes, tet_nodes, dadt_elm,
                                                               n_tet);
}
