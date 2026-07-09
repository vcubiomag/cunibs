#include "kernels.hpp"

#include <cuda_runtime.h>

// Assign one thread per node to avoid atomic updates and fix the sum order.
//   b[node] = Σ_{(e,i): tet_nodes[e,i]=node}  neg_vc[e] · dot(dadt_elm[e], g[e,i])
// ``neg_vc[e] = -vols[e] * cond[e]`` is precomputed because it does not change by placement.

namespace {

constexpr int kBlock = 256;

__global__ void rhs_kernel(const float* __restrict__ dadt_elm, const float* __restrict__ g,
                           const float* __restrict__ neg_vc, const int* __restrict__ ptr,
                           const int* __restrict__ idx, float* __restrict__ b, int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const int begin = ptr[node];
    const int end = ptr[node + 1];
    float acc = 0.f;
    for (int p = begin; p < end; ++p) {
        const int c = idx[p];
        const int e = c >> 2;
        const float dot = dadt_elm[e * 3 + 0] * g[c * 3 + 0] +
                          dadt_elm[e * 3 + 1] * g[c * 3 + 1] +
                          dadt_elm[e * 3 + 2] * g[c * 3 + 2];
        acc += neg_vc[e] * dot;
    }
    b[node] = acc;
}

__global__ void rhs_weighted_kernel(const float* __restrict__ dadt_elm,
                                    const float* __restrict__ wg, const int* __restrict__ ptr,
                                    const int* __restrict__ idx, float* __restrict__ b,
                                    int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const int begin = ptr[node];
    const int end = ptr[node + 1];
    float acc = 0.f;
    for (int p = begin; p < end; ++p) {
        const int c = idx[p];
        const int e = c >> 2;
        acc += dadt_elm[e * 3 + 0] * wg[c * 3 + 0] +
               dadt_elm[e * 3 + 1] * wg[c * 3 + 1] +
               dadt_elm[e * 3 + 2] * wg[c * 3 + 2];
    }
    b[node] = acc;
}

__global__ void weighted_gradient_kernel(const float* __restrict__ g,
                                         const float* __restrict__ neg_vc,
                                         float* __restrict__ wg, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    wg[i] = g[i] * neg_vc[i / 12];
}

}  // namespace

void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream) {
    const int blocks = (n_nodes + kBlock - 1) / kBlock;
    rhs_kernel<<<blocks, kBlock, 0, stream>>>(dadt_elm, g, neg_vc, ptr, idx, b, n_nodes);
}

void launch_rhs_weighted(const float* dadt_elm, const float* wg, const int* ptr, const int* idx,
                         float* b, int n_nodes, cudaStream_t stream) {
    const int blocks = (n_nodes + kBlock - 1) / kBlock;
    rhs_weighted_kernel<<<blocks, kBlock, 0, stream>>>(dadt_elm, wg, ptr, idx, b, n_nodes);
}

void launch_weighted_gradient(const float* g, const float* neg_vc, float* wg, int n_tet,
                              cudaStream_t stream) {
    const int n = n_tet * 12;
    const int blocks = (n + kBlock - 1) / kBlock;
    weighted_gradient_kernel<<<blocks, kBlock, 0, stream>>>(g, neg_vc, wg, n);
}
