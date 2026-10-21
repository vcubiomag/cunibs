#include "core/device_math.cuh"
#include "fem/fem.hpp"

// Assign one thread per tetrahedron to avoid scatter writes.
//   grad_v[k] = Σ_i v[tet_nodes[e,i]] · g[e,i,k]
//   E[e]      = −grad_v − dadt_elm[e]
//   magnE[e]  = ‖E[e]‖
//
// The P1 gradients of a tetrahedron sum to zero, so grad_v = Σ_{i≥1} (v_i − v_0) g_i. The
// differences are the only fp64 arithmetic: they remove the potential's common level exactly. A
// grounded potential is large against its change across one element, and the fp32 g do not sum to
// zero to the last bit, so the direct sum would leak |v| |g| ε into the gradient even accumulated
// in fp64. After the differences every term is on the scale of the gradient itself, so the rest
// runs in fp32, which keeps the kernel off the 1:64 fp64 pipe.

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

// One placement's field on one element from its four nodal potentials; returns |E|.
__device__ __forceinline__ float tet_field(const TetGeometry& t, const double (&v)[4],
                                          const float (&dadt)[3], float (&e)[3]) {
    const float d1 = static_cast<float>(v[1] - v[0]);
    const float d2 = static_cast<float>(v[2] - v[0]);
    const float d3 = static_cast<float>(v[3] - v[0]);
    float sq = 0.f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        // Explicit fmaf, so the contraction cannot differ between the K instantiations.
        const float grad = fmaf(d3, t.g[2][k], fmaf(d2, t.g[1][k], __fmul_rn(d1, t.g[0][k])));
        e[k] = __fsub_rn(-grad, dadt[k]);
        sq = fmaf(e[k], e[k], sq);
    }
    return sqrtf(sq);
}

// dA/dt already averaged onto the elements, one (n_tet, 3) array per placement.
struct ElementDadt {
    ConstPtrPack elm;
    int n_tet;

    __device__ __forceinline__ void load(int e, const int (&)[4], int c, float (&out)[3]) const {
        const Vec3View<const float> v(elm[c], n_tet);
#pragma unroll
        for (int d = 0; d < 3; ++d) out[d] = v(e, d);
    }
};

// Nodal dA/dt, averaged onto the element here: placement c of node n is at
// nodes[n * row + c * column]. The mean is dadt_element.cu's expression, so it is the same bits
// as reading that kernel's output.
struct NodalDadt {
    const float* __restrict__ nodes;
    std::int64_t row;
    int column;

    __device__ __forceinline__ void load(int, const int (&tn)[4], int c, float (&out)[3]) const {
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            const std::int64_t off = static_cast<std::int64_t>(c) * column + d;
            out[d] = 0.25f * (nodes[tn[0] * row + off] + nodes[tn[1] * row + off] +
                              nodes[tn[2] * row + off] + nodes[tn[3] * row + off]);
        }
    }
};

// tet_nodes and g are read once for all K placements. v_block is row-major (n_nodes, K) float64,
// so a node's K potentials are one vector load. A null e_out or magn_out column is not written.
template <int K, typename Dadt>
__global__ void reconstruct_block_kernel(const double* __restrict__ v_block,
                                         const int* __restrict__ tet_nodes,
                                         const float* __restrict__ g, Dadt dadt, PtrPack e_out,
                                         PtrPack magn_out, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;
    const TetGeometry t =
        load_tet(Tet4View<const int>(tet_nodes, n_tet), TetGradView<const float>(g, n_tet), e);
    const auto v = unsized<WidthView<const double, K>>(v_block);
    cuda::std::array<double, K> vn[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) load_row<K>(&v(t.nodes[i], 0), vn[i]);
#pragma unroll
    for (int c = 0; c < K; ++c) {
        const double vc[4] = {vn[0][c], vn[1][c], vn[2][c], vn[3][c]};
        float da[3], ec[3];
        dadt.load(e, t.nodes, c, da);
        const float m = tet_field(t, vc, da, ec);
        if (e_out[c] != nullptr) {
            const Vec3View<float> out(e_out[c], n_tet);
#pragma unroll
            for (int d = 0; d < 3; ++d) out(e, d) = ec[d];
        }
        if (magn_out[c] != nullptr) magn_out[c][e] = m;
    }
}

template <typename Dadt>
void launch_reconstruct_with(const double* v_block, const int* tet_nodes, const float* g,
                             const Dadt& dadt, float* const* e_out, float* const* magn_out,
                             int n_tet, int k, cudaStream_t stream) {
    const PtrPack eo = ptr_pack(e_out, k);
    const PtrPack mo = ptr_pack(magn_out, k);
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(k, "reconstruct supports k in [1, 8]", [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(n_tet)) {
            reconstruct_block_kernel<K, Dadt>
                <<<blocks, kBlock, 0, stream>>>(v_block, tet_nodes, g, dadt, eo, mo, n_tet);
        }
    });
}

}  // namespace

void launch_reconstruct_block(const double* v_block, const int* tet_nodes, const float* g,
                              const float* const* dadt_elm, float* const* e_out,
                              float* const* magn_out, int n_tet, int k, cudaStream_t stream) {
    launch_reconstruct_with(v_block, tet_nodes, g, ElementDadt{ptr_pack(dadt_elm, k), n_tet},
                            e_out, magn_out, n_tet, k, stream);
}

void launch_reconstruct_nodal_block(const double* v_block, const int* tet_nodes, const float* g,
                                    const float* dadt_nodes, std::int64_t row_stride,
                                    int column_stride, float* const* e_out,
                                    float* const* magn_out, int n_tet, int k,
                                    cudaStream_t stream) {
    launch_reconstruct_with(v_block, tet_nodes, g,
                            NodalDadt{dadt_nodes, row_stride, column_stride}, e_out, magn_out,
                            n_tet, k, stream);
}

void launch_reconstruct(const double* v, const int* tet_nodes, const float* g,
                        const float* dadt_elm, float* e_out, float* magn_out, int n_tet,
                        cudaStream_t stream) {
    launch_reconstruct_block(v, tet_nodes, g, &dadt_elm, &e_out, &magn_out, n_tet, 1, stream);
}
