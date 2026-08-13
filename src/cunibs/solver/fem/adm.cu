#include "core/device_math.cuh"
#include "fem/fem.hpp"

namespace {

//   w_e[e,k] = (−neg_vc[e]) · Σ_i values[tet_nodes[e,i]] · g[e,i,k]   (= vol_e·σ_e·(G_e λ))
// Accumulate in float64: the weight feeds a difference against the direct ROI term.
__global__ void element_weight_kernel(const double* __restrict__ values,
                                      const int* __restrict__ tet_nodes,
                                      const float* __restrict__ g,
                                      const float* __restrict__ neg_vc,
                                      double* __restrict__ w_e, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    const Vec3d grad = tet_grad4(values, tet_nodes, g, e);
    const double s = -static_cast<double>(neg_vc[e]);
    w_e[e * 3 + 0] = s * grad.x;
    w_e[e * 3 + 1] = s * grad.y;
    w_e[e * 3 + 2] = s * grad.z;
}

//   out[n] = Σ_{c ∋ n} corner[c]   (node2corner stores corner ids c = 4e + i)
// One thread per node over the incidence CSR, so the per-node order is the one every other
// assembly uses and the sum is reproducible; a scatter would need atomics and sum each node's
// corners in whatever order the blocks retired.
__global__ void node_gather_kernel(const double* __restrict__ corner, const int* __restrict__ ptr,
                                   const int* __restrict__ idx, double* __restrict__ out,
                                   int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const int end = ptr[node + 1];
    double acc = 0.0;
    for (int p = ptr[node]; p < end; ++p) acc += corner[idx[p]];
    out[node] = acc;
}

//   node_w[n,k] = ¼ Σ_{c ∋ n} w_e[c>>2, k]
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
    if (const unsigned blocks = grid_for(n_tet)) {
        element_weight_kernel<<<blocks, kBlock, 0, stream>>>(values, tet_nodes, g, neg_vc, w_e,
                                                             n_tet);
    }
}

void launch_node_gather(const double* corner, const int* ptr, const int* idx, double* out,
                        int n_nodes, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_nodes)) {
        node_gather_kernel<<<blocks, kBlock, 0, stream>>>(corner, ptr, idx, out, n_nodes);
    }
}

void launch_node_scatter3(const double* w_e, const int* ptr, const int* idx, double* node_w,
                          int n_nodes, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_nodes)) {
        node_scatter3_kernel<<<blocks, kBlock, 0, stream>>>(w_e, ptr, idx, node_w, n_nodes);
    }
}
