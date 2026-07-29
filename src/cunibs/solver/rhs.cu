#include "kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>

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

// One thread per corner c: q[c] = dadt_elm[c>>2] · wg[c]. Kept separate from the node-centric
// gather so both reads coalesce (corners c=4e..4e+3 share e → broadcast; wg[c] is
// corner-contiguous); a gather that indexed dadt_elm directly would read it uncoalesced once
// per incident node.
__global__ void rhs_corner_kernel(const float* __restrict__ dadt_elm, const float* __restrict__ wg,
                                  float* __restrict__ q, int n_corner) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_corner) return;
    const int e = c >> 2;
    q[c] = dadt_elm[e * 3 + 0] * wg[c * 3 + 0] + dadt_elm[e * 3 + 1] * wg[c * 3 + 1] +
           dadt_elm[e * 3 + 2] * wg[c * 3 + 2];
}

// Segmented reduction b[node] = Σ_{p∈[ptr[node],ptr[node+1])} q[idx[p]]. The per-node
// accumulation order is fixed by ptr/idx, so b is bit-reproducible run to run.
__global__ void rhs_gather_kernel(const float* __restrict__ q, const int* __restrict__ ptr,
                                  const int* __restrict__ idx, float* __restrict__ b, int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;
    const int begin = ptr[node];
    const int end = ptr[node + 1];
    float acc = 0.f;
    for (int p = begin; p < end; ++p) acc += q[idx[p]];
    b[node] = acc;
}

__global__ void weighted_gradient_kernel(const float* __restrict__ g,
                                         const float* __restrict__ neg_vc,
                                         float* __restrict__ wg, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    wg[i] = g[i] * neg_vc[i / 12];
}

struct RhsInPack {
    const float* p[kMaxStageBlock];
};

// Block corner pass: wg[c] (the 189 MB shared read) is loaded once for all k
// placements; q_block is row-major (n_corner, k).
__global__ void rhs_corner_block_kernel(RhsInPack dadt_elm, const float* __restrict__ wg,
                                        float* __restrict__ q_block, int n_corner, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_corner) return;
    const int e = c >> 2;
    const float w0 = wg[c * 3 + 0];
    const float w1 = wg[c * 3 + 1];
    const float w2 = wg[c * 3 + 2];
    const std::int64_t out = static_cast<std::int64_t>(c) * k;
    for (int i = 0; i < k; ++i) {
        const float* de = dadt_elm.p[i] + e * 3;
        q_block[out + i] = de[0] * w0 + de[1] * w1 + de[2] * w2;
    }
}

// Block gather: node2corner ptr/idx (66 MB shared) read once; per-node accumulation
// order per column matches the single-RHS kernel, so each column is bit-identical to
// its serial counterpart. Writes b_block row-major (n_nodes, k) — the solver layout.
__global__ void rhs_gather_block_kernel(const float* __restrict__ q_block,
                                        const int* __restrict__ ptr,
                                        const int* __restrict__ idx,
                                        float* __restrict__ b_block, int n_nodes, int k) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;
    const int begin = ptr[node];
    const int end = ptr[node + 1];
    float acc[kMaxStageBlock];
    for (int i = 0; i < k; ++i) acc[i] = 0.f;
    for (int p = begin; p < end; ++p) {
        const std::int64_t base = static_cast<std::int64_t>(idx[p]) * k;
        for (int i = 0; i < k; ++i) acc[i] += q_block[base + i];
    }
    const std::int64_t out = static_cast<std::int64_t>(node) * k;
    for (int i = 0; i < k; ++i) b_block[out + i] = acc[i];
}

}  // namespace

void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream) {
    const int blocks = (n_nodes + kBlock - 1) / kBlock;
    rhs_kernel<<<blocks, kBlock, 0, stream>>>(dadt_elm, g, neg_vc, ptr, idx, b, n_nodes);
}

void launch_rhs_weighted(const float* dadt_elm, const float* wg, const int* ptr, const int* idx,
                         float* b, int n_nodes, int n_tet, cudaStream_t stream) {
    const int n_corner = 4 * n_tet;
    float* q = nullptr;
    // Safe here because the RHS build runs outside the solver's CUDA-graph capture; cudaMallocAsync
    // would be illegal inside a captured region.
    cudaMallocAsync(&q, static_cast<size_t>(n_corner) * sizeof(float), stream);
    rhs_corner_kernel<<<(n_corner + kBlock - 1) / kBlock, kBlock, 0, stream>>>(dadt_elm, wg, q,
                                                                               n_corner);
    rhs_gather_kernel<<<(n_nodes + kBlock - 1) / kBlock, kBlock, 0, stream>>>(q, ptr, idx, b,
                                                                              n_nodes);
    cudaFreeAsync(q, stream);
}

void launch_weighted_gradient(const float* g, const float* neg_vc, float* wg, int n_tet,
                              cudaStream_t stream) {
    const int n = n_tet * 12;
    const int blocks = (n + kBlock - 1) / kBlock;
    weighted_gradient_kernel<<<blocks, kBlock, 0, stream>>>(g, neg_vc, wg, n);
}

void launch_rhs_weighted_block(const float* const* dadt_elm, const float* wg, const int* ptr,
                               const int* idx, float* b_block, int n_nodes, int n_tet, int k,
                               cudaStream_t stream) {
    RhsInPack in{};
    for (int i = 0; i < k; ++i) in.p[i] = dadt_elm[i];
    const int n_corner = 4 * n_tet;
    float* q_block = nullptr;
    // Outside any CUDA-graph capture (same constraint as launch_rhs_weighted).
    cudaMallocAsync(&q_block, static_cast<size_t>(n_corner) * k * sizeof(float), stream);
    rhs_corner_block_kernel<<<(n_corner + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        in, wg, q_block, n_corner, k);
    rhs_gather_block_kernel<<<(n_nodes + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        q_block, ptr, idx, b_block, n_nodes, k);
    cudaFreeAsync(q_block, stream);
}
