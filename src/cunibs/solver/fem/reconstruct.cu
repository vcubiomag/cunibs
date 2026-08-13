#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cstdint>

// Assign one thread per tetrahedron to avoid scatter writes.
//   grad_v[k] = Σ_i v[tet_nodes[e,i]] · g[e,i,k]
//   E[e]      = −grad_v − dadt_elm[e]
//   magnE[e]  = ‖E[e]‖
// Accumulate grad_v in float64 because subtracting dA/dt causes cancellation near the coil.

namespace {

__global__ void reconstruct_kernel(const double* __restrict__ v, const int* __restrict__ tet_nodes,
                                   const float* __restrict__ g, const float* __restrict__ dadt_elm,
                                   float* __restrict__ e_out, float* __restrict__ magn_out,
                                   int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    const Vec3d grad = tet_grad4(v, tet_nodes, g, e);
    const double ex = -grad.x - static_cast<double>(dadt_elm[e * 3 + 0]);
    const double ey = -grad.y - static_cast<double>(dadt_elm[e * 3 + 1]);
    const double ez = -grad.z - static_cast<double>(dadt_elm[e * 3 + 2]);
    e_out[e * 3 + 0] = static_cast<float>(ex);
    e_out[e * 3 + 1] = static_cast<float>(ey);
    e_out[e * 3 + 2] = static_cast<float>(ez);
    magn_out[e] = static_cast<float>(sqrt(ex * ex + ey * ey + ez * ez));
}

// Block variant: tet_nodes and g are read once for all k placements. v_block is row-major
// (n_nodes, k) float64; the fp64 gradient accumulation per column matches the single-RHS kernel
// exactly.
__global__ void reconstruct_block_kernel(const double* __restrict__ v_block,
                                         const int* __restrict__ tet_nodes,
                                         const float* __restrict__ g, ConstPtrPack dadt_elm,
                                         PtrPack e_out, PtrPack magn_out, int n_tet, int k) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    int nodes[4];
    float gm[4][3];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        nodes[i] = tet_nodes[e * 4 + i];
        const int base = (e * 4 + i) * 3;
        gm[i][0] = g[base + 0];
        gm[i][1] = g[base + 1];
        gm[i][2] = g[base + 2];
    }

    for (int c = 0; c < k; ++c) {
        double gx = 0.0, gy = 0.0, gz = 0.0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const double vi = v_block[static_cast<std::int64_t>(nodes[i]) * k + c];
            gx += vi * static_cast<double>(gm[i][0]);
            gy += vi * static_cast<double>(gm[i][1]);
            gz += vi * static_cast<double>(gm[i][2]);
        }
        const float* de = dadt_elm.p[c] + e * 3;
        const double ex = -gx - static_cast<double>(de[0]);
        const double ey = -gy - static_cast<double>(de[1]);
        const double ez = -gz - static_cast<double>(de[2]);
        float* eo = e_out.p[c] + e * 3;
        eo[0] = static_cast<float>(ex);
        eo[1] = static_cast<float>(ey);
        eo[2] = static_cast<float>(ez);
        magn_out.p[c][e] = static_cast<float>(sqrt(ex * ex + ey * ey + ez * ez));
    }
}

}  // namespace

void launch_reconstruct(const double* v, const int* tet_nodes, const float* g,
                        const float* dadt_elm, float* e_out, float* magn_out, int n_tet,
                        cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_tet)) {
        reconstruct_kernel<<<blocks, kBlock, 0, stream>>>(v, tet_nodes, g, dadt_elm, e_out,
                                                          magn_out, n_tet);
    }
}

void launch_reconstruct_block(const double* v_block, const int* tet_nodes, const float* g,
                              const float* const* dadt_elm, float* const* e_out,
                              float* const* magn_out, int n_tet, int k, cudaStream_t stream) {
    ConstPtrPack in{};
    PtrPack eo{};
    PtrPack mo{};
    for (int c = 0; c < k; ++c) {
        in.p[c] = dadt_elm[c];
        eo.p[c] = e_out[c];
        mo.p[c] = magn_out[c];
    }
    if (const unsigned blocks = grid_for(n_tet)) {
        reconstruct_block_kernel<<<blocks, kBlock, 0, stream>>>(v_block, tet_nodes, g, in, eo, mo,
                                                                n_tet, k);
    }
}
