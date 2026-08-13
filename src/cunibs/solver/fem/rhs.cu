#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cstdint>

// Assign one thread per node to avoid atomic updates and fix the sum order.
//   b[node] = Σ_{(e,i): tet_nodes[e,i]=node}  neg_vc[e] · dot(dadt_elm[e], g[e,i])
// ``neg_vc[e] = -vols[e] * cond[e]``.
//
// Two forms. The fused one does the whole thing in the node gather, which reads g uncoalesced
// once per incident node; the staged one pays a corner pass first so that both halves coalesce.

namespace {

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

// One thread per corner c: q[c] = neg_vc[c>>2] · dadt_elm[c>>2] · g[c]. Kept separate from the
// node-centric gather so both reads coalesce (corners c=4e..4e+3 share e → broadcast; g[c] is
// corner-contiguous).
__global__ void rhs_corner_kernel(const float* __restrict__ dadt_elm, const float* __restrict__ g,
                                  const float* __restrict__ neg_vc, float* __restrict__ q,
                                  int n_corner) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_corner) return;
    const int e = c >> 2;
    const float w = neg_vc[e];
    q[c] = dadt_elm[e * 3 + 0] * (g[c * 3 + 0] * w) + dadt_elm[e * 3 + 1] * (g[c * 3 + 1] * w) +
           dadt_elm[e * 3 + 2] * (g[c * 3 + 2] * w);
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

// Block corner pass: g and neg_vc are loaded once for all k placements; q_block is row-major
// (n_corner, k).
__global__ void rhs_corner_block_kernel(ConstPtrPack dadt_elm, const float* __restrict__ g,
                                        const float* __restrict__ neg_vc,
                                        float* __restrict__ q_block, int n_corner, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_corner) return;
    const int e = c >> 2;
    const float w = neg_vc[e];
    const float w0 = g[c * 3 + 0] * w;
    const float w1 = g[c * 3 + 1] * w;
    const float w2 = g[c * 3 + 2] * w;
    const std::int64_t out = static_cast<std::int64_t>(c) * k;
    for (int i = 0; i < k; ++i) {
        const float* de = dadt_elm[i] + e * 3;
        q_block[out + i] = de[0] * w0 + de[1] * w1 + de[2] * w2;
    }
}

// Block gather: node2corner ptr/idx read once. Per-node accumulation order per column matches the
// single-RHS kernel, so each column is bit-identical to its serial counterpart. Writes b_block
// row-major (n_nodes, k) — the solver layout.
__global__ void rhs_gather_block_kernel(const float* __restrict__ q_block,
                                        const int* __restrict__ ptr,
                                        const int* __restrict__ idx,
                                        float* __restrict__ b_block, int n_nodes, int k) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;
    const int begin = ptr[node];
    const int end = ptr[node + 1];
    cuda::std::array<float, kMaxStageBlock> acc;
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
    if (const unsigned blocks = grid_for(n_nodes)) {
        rhs_kernel<<<blocks, kBlock, 0, stream>>>(dadt_elm, g, neg_vc, ptr, idx, b, n_nodes);
    }
}

void launch_rhs_staged(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                       const int* idx, float* b, int n_nodes, int n_tet, cudaStream_t stream) {
    const int n_corner = 4 * n_tet;
    float* q = nullptr;
    // Safe here because the RHS build runs outside the solver's CUDA-graph capture; cudaMallocAsync
    // would be illegal inside a captured region.
    check_cuda(cudaMallocAsync(&q, static_cast<size_t>(n_corner) * sizeof(float), stream), "rhs",
               "mallocAsync(q)");
    if (const unsigned blocks = grid_for(n_corner)) {
        rhs_corner_kernel<<<blocks, kBlock, 0, stream>>>(dadt_elm, g, neg_vc, q, n_corner);
    }
    if (const unsigned blocks = grid_for(n_nodes)) {
        rhs_gather_kernel<<<blocks, kBlock, 0, stream>>>(q, ptr, idx, b, n_nodes);
    }
    check_cuda(cudaFreeAsync(q, stream), "rhs", "freeAsync(q)");
}

void launch_rhs_staged_block(const float* const* dadt_elm, const float* g, const float* neg_vc,
                             const int* ptr, const int* idx, float* b_block, int n_nodes,
                             int n_tet, int k, cudaStream_t stream) {
    ConstPtrPack in{};
    for (int i = 0; i < k; ++i) in[i] = dadt_elm[i];
    const int n_corner = 4 * n_tet;
    float* q_block = nullptr;
    // Outside any CUDA-graph capture (same constraint as launch_rhs_staged).
    check_cuda(cudaMallocAsync(&q_block, static_cast<size_t>(n_corner) * k * sizeof(float),
                               stream),
               "rhs", "mallocAsync(q_block)");
    if (const unsigned blocks = grid_for(n_corner)) {
        rhs_corner_block_kernel<<<blocks, kBlock, 0, stream>>>(in, g, neg_vc, q_block, n_corner,
                                                               k);
    }
    if (const unsigned blocks = grid_for(n_nodes)) {
        rhs_gather_block_kernel<<<blocks, kBlock, 0, stream>>>(q_block, ptr, idx, b_block,
                                                               n_nodes, k);
    }
    check_cuda(cudaFreeAsync(q_block, stream), "rhs", "freeAsync(q_block)");
}
