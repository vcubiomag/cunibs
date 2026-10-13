#include "core/device_math.cuh"
#include "fem/fem.hpp"

// Assign one thread per tetrahedron to avoid scatter writes.
//   grad_v[k] = Σ_i v[tet_nodes[e,i]] · g[e,i,k]
//   E[e]      = −grad_v − dadt_elm[e]
//   magnE[e]  = ‖E[e]‖
//
// The P1 gradients of a tetrahedron sum to zero, so grad_v = Σ_{i≥1} (v_i − v_0) g_i. The
// differences are the only fp64 arithmetic: they remove the potential's common level exactly,
// and that level is what made the direct sum lose digits. A grounded potential is large against
// its change across one element, and the fp32 g do not sum to zero to the last bit, so the direct
// form leaked |v| |g| ε into the gradient even when accumulated in fp64. After the differences
// every term is on the scale of the gradient itself, and the rest runs in fp32: this kernel was
// fp64-bound on a 1:64 part, at roughly forty fp64 instructions per element and placement.

namespace {

struct TetGeometry {
    int nodes[4];
    float g[3][3];  // g[i - 1] for corners 1..3; corner 0 is implied by the zero sum
};

__device__ __forceinline__ TetGeometry load_tet(Tet4View<const int> tn, TetGradView<const float> gv,
                                                int e) {
    TetGeometry t;
#pragma unroll
    for (int i = 0; i < 4; ++i) t.nodes[i] = tn(e, i);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int d = 0; d < 3; ++d) t.g[i][d] = gv(e, i + 1, d);
    }
    return t;
}

// One placement's field on one element from its four nodal potentials. Shared by the single and
// block kernels so that a column's arithmetic is the same whichever ran.
__device__ __forceinline__ void tet_field(const TetGeometry& t, const double (&v)[4],
                                          const float* __restrict__ dadt, float* __restrict__ out,
                                          float& magn) {
    const float d1 = static_cast<float>(v[1] - v[0]);
    const float d2 = static_cast<float>(v[2] - v[0]);
    const float d3 = static_cast<float>(v[3] - v[0]);
    float sq = 0.f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        // Explicit fmaf, so the contraction cannot differ between the single and block kernels.
        const float grad = fmaf(d3, t.g[2][k], fmaf(d2, t.g[1][k], __fmul_rn(d1, t.g[0][k])));
        const float ek = __fsub_rn(-grad, dadt[k]);
        out[k] = ek;
        sq = fmaf(ek, ek, sq);
    }
    magn = sqrtf(sq);
}

__global__ void reconstruct_kernel(const double* __restrict__ v, const int* __restrict__ tet_nodes,
                                   const float* __restrict__ g, const float* __restrict__ dadt_elm,
                                   float* __restrict__ e_out, float* __restrict__ magn_out,
                                   int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;
    const TetGeometry t =
        load_tet(Tet4View<const int>(tet_nodes, n_tet), TetGradView<const float>(g, n_tet), e);
    double vn[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) vn[i] = v[t.nodes[i]];
    const Vec3View<const float> dadt(dadt_elm, n_tet);
    const Vec3View<float> out(e_out, n_tet);
    float m;
    tet_field(t, vn, &dadt(e, 0), &out(e, 0), m);
    magn_out[e] = m;
}

// Block variant: tet_nodes and g are read once for all K placements. v_block is row-major
// (n_nodes, K) float64, so a node's K potentials are one vector load.
template <int K>
__global__ void reconstruct_block_kernel(const double* __restrict__ v_block,
                                         const int* __restrict__ tet_nodes,
                                         const float* __restrict__ g, ConstPtrPack dadt_elm,
                                         PtrPack e_out, PtrPack magn_out, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;
    const TetGeometry t =
        load_tet(Tet4View<const int>(tet_nodes, n_tet), TetGradView<const float>(g, n_tet), e);
    const WidthView<const double, K> v(v_block, kUnsizedRows);
    cuda::std::array<double, K> vn[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) load_row<K>(&v(t.nodes[i], 0), vn[i]);
#pragma unroll
    for (int c = 0; c < K; ++c) {
        const double vc[4] = {vn[0][c], vn[1][c], vn[2][c], vn[3][c]};
        const Vec3View<const float> dadt(dadt_elm[c], n_tet);
        const Vec3View<float> out(e_out[c], n_tet);
        float m;
        tet_field(t, vc, &dadt(e, 0), &out(e, 0), m);
        magn_out[c][e] = m;
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
    const ConstPtrPack in = const_ptr_pack(dadt_elm, k);
    const PtrPack eo = ptr_pack(e_out, k);
    const PtrPack mo = ptr_pack(magn_out, k);
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(k, "reconstruct supports k in [1, 8]", [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(n_tet)) {
            reconstruct_block_kernel<K>
                <<<blocks, kBlock, 0, stream>>>(v_block, tet_nodes, g, in, eo, mo, n_tet);
        }
    });
}
