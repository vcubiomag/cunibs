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

// The k-wide form over the interleaved (n_nodes, K, 3) field the forward path stages, so that
// tet_nodes is read once per block and a corner's K placements are one contiguous gather. Each
// column's expression is the single kernel's, so its result is the same bit for bit.
template <int K>
__global__ void dadt_element_average_block_kernel(const float* __restrict__ dadt_nodes,
                                                  const int* __restrict__ tet_nodes,
                                                  PtrPack dadt_elm, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;
    const Tet4View<const int> tn(tet_nodes, n_tet);
    cuda::std::array<float, 3 * K> v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        load_row<3 * K>(dadt_nodes + static_cast<std::int64_t>(tn(e, i)) * (3 * K), v[i]);
    }
#pragma unroll
    for (int c = 0; c < K; ++c) {
        const Vec3View<float> elm(dadt_elm[c], n_tet);
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            const int q = 3 * c + d;
            elm(e, d) = 0.25f * (v[0][q] + v[1][q] + v[2][q] + v[3][q]);
        }
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

void launch_dadt_element_average_block(const float* dadt_nodes, const int* tet_nodes,
                                       float* const* dadt_elm, int n_tet, int k,
                                       cudaStream_t stream) {
    const PtrPack out = ptr_pack(dadt_elm, k);
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(k, "element average supports k in [1, 8]", [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(n_tet)) {
            dadt_element_average_block_kernel<K>
                <<<blocks, kBlock, 0, stream>>>(dadt_nodes, tet_nodes, out, n_tet);
        }
    });
}
