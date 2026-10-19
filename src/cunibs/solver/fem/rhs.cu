#include "core/device_math.cuh"
#include "fem/fem.hpp"

// Assign one thread per node to avoid atomic updates and fix the sum order.
//   b[node] = Σ_{(e,i): tet_nodes[e,i]=node}  neg_vc[e] · dot(dadt_elm[e], g[e,i])
// ``neg_vc[e] = -vols[e] * cond[e]``.

namespace {

__global__ void rhs_kernel(const float* __restrict__ dadt_elm, const float* __restrict__ g,
                           const float* __restrict__ neg_vc, const int* __restrict__ ptr,
                           const int* __restrict__ idx, float* __restrict__ b, int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const auto dadt = unsized<Vec3View<const float>>(dadt_elm);
    const auto gv = unsized<Vec3View<const float>>(g);

    const int begin = ptr[node];
    const int end = ptr[node + 1];
    float acc = 0.f;
    for (int p = begin; p < end; ++p) {
        const int c = idx[p];
        const int e = c >> 2;
        const float dot = dadt(e, 0) * gv(c, 0) + dadt(e, 1) * gv(c, 1) + dadt(e, 2) * gv(c, 2);
        acc += neg_vc[e] * dot;
    }
    b[node] = acc;
}

// The forward path's RHS as one assembled operator on the nodal field. dA/dt reaches an element
// as the mean of its four nodes, and every step after that is linear, so
//     b_i = Σ_j B_ij · a_j,   B_ij = Σ_{e ∋ i, j} (neg_vc_e / 4) g[e, i_loc]  (a 3-vector)
// with a the nodal dA/dt. B has the stiffness pattern and depends only on the mesh and the
// conductivities, so it is built once per context. A placement then costs one pass over B and a
// gather of a cache-resident nodal field. B is contracted in fp64 at setup, so each entry is
// rounded once.
//
// A nonzero is packed as (Bx, By, Bz, column bits) so that it is one 16-byte load.

// Setup: one thread per row. Each entry walks the row's incident corners in node2corner order and
// takes the ones whose element also holds its column, so the sum order is fixed by the mesh.
__global__ void rhs_operator_fill_kernel(const double* __restrict__ g,
                                         const double* __restrict__ scale,
                                         const int* __restrict__ tet_nodes,
                                         const int* __restrict__ ptr, const int* __restrict__ idx,
                                         const int* __restrict__ indptr,
                                         const int* __restrict__ indices, float4* __restrict__ op,
                                         int n_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const auto gv = unsized<Vec3View<const double, std::int64_t>>(g);
    const auto tn = unsized<Tet4View<const int, std::int64_t>>(tet_nodes);
    const int begin = ptr[row];
    const int end = ptr[row + 1];
    for (int p = indptr[row]; p < indptr[row + 1]; ++p) {
        const int col = indices[p];
        double bx = 0.0, by = 0.0, bz = 0.0;
        for (int q = begin; q < end; ++q) {
            const int corner = idx[q];
            const int e = corner >> 2;
            // Multiplicity rather than membership, so the operator is the exact linear map even
            // for an element that names a node twice.
            const int mult = (tn(e, 0) == col) + (tn(e, 1) == col) + (tn(e, 2) == col) +
                             (tn(e, 3) == col);
            if (mult == 0) continue;
            const double s = scale[e] * mult;
            bx += s * gv(corner, 0);
            by += s * gv(corner, 1);
            bz += s * gv(corner, 2);
        }
        op[p] = make_float4(static_cast<float>(bx), static_cast<float>(by),
                            static_cast<float>(bz), __int_as_float(col));
    }
}

// Lanes per row in the apply. Four lanes read a 64-byte stretch of a ~14-nonzero row per step;
// measured against 1, 2 and 8 it is fastest at k = 1 and tied with 8 at k = 8. It must not depend
// on k: it fixes each column's sum order, which is what keeps a placement's b the same at every
// block width.
constexpr int kRhsOpTpr = 4;

static_assert(kWarp % kRhsOpTpr == 0 && kBlock % kRhsOpTpr == 0,
              "the apply's lanes per row must divide the warp and the block");

// a is the nodal field interleaved (n_nodes, K, 3), so a neighbour's K placements are one
// contiguous gather rather than K scattered ones.
template <int K>
__global__ void __launch_bounds__(kBlock)
    rhs_operator_kernel(int n, const int* __restrict__ indptr, const float4* __restrict__ op,
                        const float* __restrict__ a, float* __restrict__ b) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / kRhsOpTpr;
    const int lane = static_cast<int>(threadIdx.x) % kRhsOpTpr;
    cuda::std::array<float, K> acc;
#pragma unroll
    for (int c = 0; c < K; ++c) acc[c] = 0.f;
    if (row < n) {
        const int row_e = indptr[row + 1];
        for (int p = indptr[row] + lane; p < row_e; p += kRhsOpTpr) {
            const float4 v = op[p];
            cuda::std::array<float, 3 * K> t;
            load_row<3 * K>(a + static_cast<std::int64_t>(__float_as_int(v.w)) * (3 * K), t);
            // Spelled out so that every K contracts the same way.
#pragma unroll
            for (int c = 0; c < K; ++c) {
                const float bt = __fmaf_rn(
                    v.z, t[3 * c + 2], __fmaf_rn(v.x, t[3 * c], __fmul_rn(v.y, t[3 * c + 1])));
                acc[c] = __fadd_rn(acc[c], bt);
            }
        }
    }
    warp_reduce_sum<kRhsOpTpr>(acc);
    if (row < n && lane == 0) {
        const WidthView<float, K> out(b, n);
        store_row<K>(&out(row, 0), acc);
    }
}

}  // namespace

void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_nodes)) {
        rhs_kernel<<<blocks, kBlock, 0, stream>>>(dadt_elm, g, neg_vc, ptr, idx, b, n_nodes);
    }
}

void launch_rhs_operator_fill(const double* g, const double* scale, const int* tet_nodes,
                              const int* ptr, const int* idx, const int* indptr,
                              const int* indices, float* op, int n_rows, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_rows)) {
        rhs_operator_fill_kernel<<<blocks, kBlock, 0, stream>>>(
            g, scale, tet_nodes, ptr, idx, indptr, indices, reinterpret_cast<float4*>(op), n_rows);
    }
}

void launch_rhs_operator(const int* indptr, const float* op, const float* a, float* b,
                         int n_nodes, int k, cudaStream_t stream) {
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(k, "RHS operator supports k in [1, 8]", [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks =
                grid_for(static_cast<std::int64_t>(n_nodes) * kRhsOpTpr)) {
            rhs_operator_kernel<K><<<blocks, kBlock, 0, stream>>>(
                n_nodes, indptr, reinterpret_cast<const float4*>(op), a, b);
        }
    });
}
