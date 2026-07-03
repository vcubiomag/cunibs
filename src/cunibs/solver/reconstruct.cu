#include "kernels.hpp"

#include <cuda_runtime.h>

// Assign one thread per tetrahedron to avoid scatter writes.
//   grad_v[k] = Σ_i v[tet_nodes[e,i]] · g[e,i,k]
//   E[e]      = −grad_v − dadt_elm[e]
//   magnE[e]  = ‖E[e]‖
// Accumulate grad_v in float64 because subtracting dA/dt causes cancellation near the coil.
// Float32 accumulation increases relative L2 error in magnE by about 1e-5.

namespace {

constexpr int kBlock = 256;

__global__ void reconstruct_kernel(const double* __restrict__ v, const int* __restrict__ tet_nodes,
                                   const float* __restrict__ g, const float* __restrict__ dadt_elm,
                                   float* __restrict__ e_out, float* __restrict__ magn_out,
                                   int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    double gx = 0.0, gy = 0.0, gz = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const double vi = v[tet_nodes[e * 4 + i]];
        const int base = (e * 4 + i) * 3;
        gx += vi * static_cast<double>(g[base + 0]);
        gy += vi * static_cast<double>(g[base + 1]);
        gz += vi * static_cast<double>(g[base + 2]);
    }
    const double ex = -gx - static_cast<double>(dadt_elm[e * 3 + 0]);
    const double ey = -gy - static_cast<double>(dadt_elm[e * 3 + 1]);
    const double ez = -gz - static_cast<double>(dadt_elm[e * 3 + 2]);
    e_out[e * 3 + 0] = static_cast<float>(ex);
    e_out[e * 3 + 1] = static_cast<float>(ey);
    e_out[e * 3 + 2] = static_cast<float>(ez);
    magn_out[e] = static_cast<float>(sqrt(ex * ex + ey * ey + ez * ez));
}

}  // namespace

void launch_reconstruct(const double* v, const int* tet_nodes, const float* g,
                        const float* dadt_elm, float* e_out, float* magn_out, int n_tet,
                        cudaStream_t stream) {
    const int blocks = (n_tet + kBlock - 1) / kBlock;
    reconstruct_kernel<<<blocks, kBlock, 0, stream>>>(v, tet_nodes, g, dadt_elm, e_out, magn_out,
                                                      n_tet);
}
