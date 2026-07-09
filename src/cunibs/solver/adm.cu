#include "kernels.hpp"

#include <cuda_runtime.h>

namespace {

constexpr int kBlock = 256;

//   w_e[e,k] = (−neg_vc[e]) · Σ_i values[tet_nodes[e,i]] · g[e,i,k]   (= vol_e·σ_e·(G_e λ))
// Accumulate in float64: the weight feeds a difference against the direct ROI term.
__global__ void element_weight_kernel(const double* __restrict__ values,
                                      const int* __restrict__ tet_nodes,
                                      const float* __restrict__ g,
                                      const float* __restrict__ neg_vc,
                                      double* __restrict__ w_e, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    double gx = 0.0, gy = 0.0, gz = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const double vi = values[tet_nodes[e * 4 + i]];
        const int base = (e * 4 + i) * 3;
        gx += vi * static_cast<double>(g[base + 0]);
        gy += vi * static_cast<double>(g[base + 1]);
        gz += vi * static_cast<double>(g[base + 2]);
    }
    const double s = -static_cast<double>(neg_vc[e]);
    w_e[e * 3 + 0] = s * gx;
    w_e[e * 3 + 1] = s * gy;
    w_e[e * 3 + 2] = s * gz;
}

//   node_w[n,k] = ¼ Σ_{c ∋ n} w_e[c>>2, k]   (node2corner stores corner ids c = 4e + i)
__global__ void node_scatter3_kernel(const double* __restrict__ w_e, const int* __restrict__ ptr,
                                     const int* __restrict__ idx, double* __restrict__ node_w,
                                     int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const int begin = ptr[node];
    const int end = ptr[node + 1];
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int p = begin; p < end; ++p) {
        const int e = idx[p] >> 2;
        ax += w_e[e * 3 + 0];
        ay += w_e[e * 3 + 1];
        az += w_e[e * 3 + 2];
    }
    node_w[node * 3 + 0] = 0.25 * ax;
    node_w[node * 3 + 1] = 0.25 * ay;
    node_w[node * 3 + 2] = 0.25 * az;
}

}  // namespace

void launch_element_weight(const double* values, const int* tet_nodes, const float* g,
                           const float* neg_vc, double* w_e, int n_tet, cudaStream_t stream) {
    const int blocks = (n_tet + kBlock - 1) / kBlock;
    element_weight_kernel<<<blocks, kBlock, 0, stream>>>(values, tet_nodes, g, neg_vc, w_e, n_tet);
}

void launch_node_scatter3(const double* w_e, const int* ptr, const int* idx, double* node_w,
                          int n_nodes, cudaStream_t stream) {
    const int blocks = (n_nodes + kBlock - 1) / kBlock;
    node_scatter3_kernel<<<blocks, kBlock, 0, stream>>>(w_e, ptr, idx, node_w, n_nodes);
}
