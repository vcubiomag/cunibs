#include "device_math.cuh"
#include "kernels.hpp"

#include <cstdint>

// Recovery post-processing: turn the raw per-tetrahedron E into a smoothed field by fitting
// local patches around each slot and interpolating back.
//
// A slot is the unit a patch is fitted around: a node for the global mode, a (node, tissue) pair
// for the tissue-restricted ones. Either way the patch is a segment of a CSR, so one kernel
// serves every mode and only the operator's contents differ. The SPR kernels walk a corner CSR
// (corner ids c = 4e + i); the harmonic ones walk a CSR of patch nodes.
//
// SPR is linear in the per-element field, so the whole fit collapses to one scalar weight per
// corner, built once per mesh:
//     E*[s] = Σ_{c ∋ s} w[c] · E[c>>2]
// See fem/recovery.py for the derivation and for how w is built.
//
// Every reduction here is a fixed-order walk of ptr/idx with one thread per output slot, which
// is what keeps the recovered field bit-reproducible and independent of the block width. The
// only atomic is the integer fallback counter.

namespace {

// Largest sum|w| a patch fit may have before it is rejected for the volume-weighted average.
// Weights sum to one, so this bounds the recovery's amplification; 8 leaves ample room for a
// legitimately anisotropic patch while catching the near-degenerate ones.
constexpr double kMaxLebesgue = 8.0;

// --- outer-boundary detection ---------------------------------------------------------------

// One thread per face. The four faces of a tet are its four 3-subsets of local corners; which
// subset maps to which face does not matter, only that every tet enumerates the same four.
//
// A face is on the outer boundary when exactly one tet owns it. Every owner of face (n, p, q)
// holds n, so every owner sits in n's segment of the node2corner CSR and counting them is one
// walk of that segment.
//
// Marking is idempotent (every writer stores 1), so the concurrent stores race benignly and need
// no atomic. The count does not depend on the order the segment is walked, so nothing here rests
// on build_node2corner's stable sort.
__global__ void mark_outer_boundary_kernel(const int* __restrict__ tet_nodes,
                                           const int* __restrict__ ptr,
                                           const int* __restrict__ idx,
                                           int* __restrict__ is_boundary, int n_face) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_face) return;
    const int e = f >> 2, skip = f & 3;

    int a[3];
    int m = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i != skip) a[m++] = tet_nodes[e * 4 + i];
    }

    // Rotate the shortest incidence list into the pivot slot with scalar selects: indexing a[] by
    // a runtime value would spill the triple out of registers into local memory.
    int n = a[0], p = a[1], q = a[2];
    int len = ptr[a[0] + 1] - ptr[a[0]];
    const int len1 = ptr[a[1] + 1] - ptr[a[1]];
    if (len1 < len) { len = len1; n = a[1]; p = a[2]; q = a[0]; }
    const int len2 = ptr[a[2] + 1] - ptr[a[2]];
    if (len2 < len) { len = len2; n = a[2]; p = a[0]; q = a[1]; }

    // Only "exactly one" against "more than one" matters, so the walk stops at the second owner.
    const int begin = ptr[n], end = begin + len;
    int owners = 0;
    for (int k = begin; k < end && owners < 2; ++k) {
        // A 16-byte load: tet_nodes is one contiguous 4 * n_tet block, so a tet's four ids never
        // straddle the boundary and the base is an allocation start rather than a strided view.
        const int4 t = *reinterpret_cast<const int4*>(tet_nodes + (idx[k] >> 2) * 4);
        owners += (t.x == p || t.y == p || t.z == p || t.w == p) &&
                  (t.x == q || t.y == q || t.z == q || t.w == q);
    }
    if (owners != 1) return;

    is_boundary[n] = 1;
    is_boundary[p] = 1;
    is_boundary[q] = 1;
}

// --- patch fitting --------------------------------------------------------------------------

// Cholesky-solve A X = [e_r : r in rhs_rows] for an N x N SPD A held in registers. Returns false
// on a non-positive pivot, which is how a coplanar or under-filled patch reports itself. N <= 9,
// so the factor sits in local memory.
template <int N>
__device__ bool solve_spd_columns(const double a[N][N], const int* rhs_rows, int n_rhs,
                                  double out[][N]) {
    double l[N][N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) l[i][j] = 0.0;
    }
    // a[0][0] is the patch size (the constant basis row), so it sets the pivot scale.
    const double floor_pivot = 1e-12 * a[0][0];
    for (int j = 0; j < N; ++j) {
        double d = a[j][j];
        for (int k = 0; k < j; ++k) d -= l[j][k] * l[j][k];
        if (!(d > floor_pivot)) return false;
        l[j][j] = sqrt(d);
        for (int i = j + 1; i < N; ++i) {
            double s = a[i][j];
            for (int k = 0; k < j; ++k) s -= l[i][k] * l[j][k];
            l[i][j] = s / l[j][j];
        }
    }
    for (int c = 0; c < n_rhs; ++c) {
        double y[N];
        for (int i = 0; i < N; ++i) {
            double s = (i == rhs_rows[c]) ? 1.0 : 0.0;
            for (int k = 0; k < i; ++k) s -= l[i][k] * y[k];
            y[i] = s / l[i][i];
        }
        for (int i = N - 1; i >= 0; --i) {
            double s = y[i];
            for (int k = i + 1; k < N; ++k) s -= l[k][i] * out[c][k];
            out[c][i] = s / l[i][i];
        }
    }
    return true;
}

__device__ __forceinline__ void tet_barycentre(const double* __restrict__ nodes_mm,
                                               const int* __restrict__ tet_nodes, int e,
                                               double b[3]) {
    b[0] = b[1] = b[2] = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = tet_nodes[e * 4 + i];
        b[0] += nodes_mm[n * 3 + 0];
        b[1] += nodes_mm[n * 3 + 1];
        b[2] += nodes_mm[n * 3 + 2];
    }
    b[0] *= 0.25;
    b[1] *= 0.25;
    b[2] *= 0.25;
}

// w_c = z . p~(b_e), the fit evaluated at one patch element in the node-centred, radius-scaled
// frame p~ = [1, (b_e - x_n)/h].
__device__ __forceinline__ double spr_corner_weight(const double z[4], const double b[3],
                                                    const double xn[3], double h) {
    return z[0] + z[1] * (b[0] - xn[0]) / h + z[2] * (b[1] - xn[1]) / h +
           z[3] * (b[2] - xn[2]) / h;
}

// One thread per slot. Four passes over the patch: the scaling radius, the normal matrix, the
// Lebesgue constant, then the weights.
//
// Fitting in p~ rather than SimNIBS's absolute [1, x, y, z] is an affine change of basis, so the
// value recovered at the node is unchanged in exact arithmetic, but the normal matrix is far
// better conditioned on head-mesh coordinates. In that frame p~(x_n) = e_0, so only the first row
// of the inverse is needed: solve A z = e_0 and take w_c = z . p~_c.
//
// Nodes on the outer boundary of the tet volume take a volume-weighted average instead, matching
// SimNIBS, and so does any patch whose fit is ill-posed. Both are a select over the whole slot,
// never a blend: a failed solve must not leak into a good one.
__global__ void spr_weights_kernel(const double* __restrict__ nodes_mm,
                                   const int* __restrict__ tet_nodes,
                                   const float* __restrict__ vols, const int* __restrict__ ptr,
                                   const int* __restrict__ idx, const int* __restrict__ slot_node,
                                   const int* __restrict__ is_boundary, float* __restrict__ w,
                                   int* __restrict__ n_fallback, int n_slots) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;

    const int node = slot_node ? slot_node[s] : s;
    const int begin = ptr[s], end = ptr[s + 1];
    const int count = end - begin;
    if (count <= 0) return;

    const double xn[3] = {nodes_mm[node * 3 + 0], nodes_mm[node * 3 + 1],
                          nodes_mm[node * 3 + 2]};

    bool fit = (count >= 4) && !(is_boundary && is_boundary[node]);
    double z[1][4] = {{1.0, 0.0, 0.0, 0.0}};
    double h = 1.0;

    if (fit) {
        h = 0.0;
        for (int p = begin; p < end; ++p) {
            double b[3];
            tet_barycentre(nodes_mm, tet_nodes, idx[p] >> 2, b);
#pragma unroll
            for (int c = 0; c < 3; ++c) h = fmax(h, fabs(b[c] - xn[c]));
        }
        fit = h > 0.0;
    }
    if (fit) {
        double a[4][4] = {};
        for (int p = begin; p < end; ++p) {
            double b[3];
            tet_barycentre(nodes_mm, tet_nodes, idx[p] >> 2, b);
            const double q[4] = {1.0, (b[0] - xn[0]) / h, (b[1] - xn[1]) / h,
                                 (b[2] - xn[2]) / h};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) a[i][j] += q[i] * q[j];
            }
        }
        constexpr int kConstantRow = 0;
        fit = solve_spd_columns<4>(a, &kConstantRow, 1, z);
    }

    // The weights sum to one, so sum|w| -- the patch's Lebesgue constant -- is how much the fit
    // can amplify its inputs. A healthy patch sits near 1. A pivot test alone does not catch a
    // patch that is merely very flat: it passes, and then returns weights of order thousands.
    // Restricting patches to one tissue makes such thin structures common.
    if (fit) {
        double lebesgue = 0.0;
        for (int p = begin; p < end; ++p) {
            double b[3];
            tet_barycentre(nodes_mm, tet_nodes, idx[p] >> 2, b);
            lebesgue += fabs(spr_corner_weight(z[0], b, xn, h));
        }
        fit = lebesgue <= kMaxLebesgue;
    }

    if (!fit) {
        if (n_fallback) atomicAdd(n_fallback, 1);
        double total = 0.0;
        for (int p = begin; p < end; ++p) total += static_cast<double>(vols[idx[p] >> 2]);
        // A zero-volume patch cannot be averaged either; fall back once more to a plain mean so
        // the weights still sum to one and no slot is left holding a NaN.
        const bool by_volume = total > 0.0;
        const double inv = by_volume ? 1.0 / total : 1.0 / static_cast<double>(count);
        for (int p = begin; p < end; ++p) {
            const double num = by_volume ? static_cast<double>(vols[idx[p] >> 2]) : 1.0;
            w[p] = static_cast<float>(num * inv);
        }
        return;
    }

    for (int p = begin; p < end; ++p) {
        double b[3];
        tet_barycentre(nodes_mm, tet_nodes, idx[p] >> 2, b);
        w[p] = static_cast<float>(spr_corner_weight(z[0], b, xn, h));
    }
}

// --- harmonic-constrained potential recovery --------------------------------------------------

// The 9 harmonic quadratics in the node-centred, radius-scaled frame. Dropping the tenth basis
// function is the whole method: a general quadratic has three independent second derivatives,
// but at an interface or boundary node the patch is a half-ball and cannot determine the one
// normal to the flat side. Laplace's equation ties the three together, so the in-plane curvature
// the patch does resolve pins the one it does not. v is harmonic inside a tissue because the
// source is a divergence-free dipole potential outside the head.
struct HarmonicBasis {
    static constexpr int kTerms = 9;
    __device__ __forceinline__ static void eval(double u, double v, double t, double p[kTerms]) {
        p[0] = 1.0;
        p[1] = u;
        p[2] = v;
        p[3] = t;
        p[4] = u * u - v * v;
        p[5] = v * v - t * t;
        p[6] = u * v;
        p[7] = u * t;
        p[8] = v * t;
    }
};

// The fallback rung: a plain linear fit needs only four non-coplanar patch nodes, so it is
// reachable from any non-degenerate tetrahedron.
struct LinearBasis {
    static constexpr int kTerms = 4;
    __device__ __forceinline__ static void eval(double u, double v, double t, double p[kTerms]) {
        p[0] = 1.0;
        p[1] = u;
        p[2] = v;
        p[3] = t;
    }
};

// Fit the potential over one slot's node patch in Basis, then write the three gradient weights
// per patch node. The gradient at the node is the linear coefficients divided by the scaling
// radius, because every quadratic term has zero derivative at the centre. Leaves w untouched and
// returns false when the normal matrix is not positive definite.
template <typename Basis>
__device__ bool fit_gradient_weights(const double* __restrict__ nodes_mm,
                                     const int* __restrict__ pidx, int begin, int end,
                                     const double xn[3], double h, float* __restrict__ w) {
    constexpr int N = Basis::kTerms;
    if (end - begin < N) return false;

    double a[N][N] = {};
    for (int p = begin; p < end; ++p) {
        const int m = pidx[p];
        double q[N];
        Basis::eval((nodes_mm[m * 3 + 0] - xn[0]) / h, (nodes_mm[m * 3 + 1] - xn[1]) / h,
                    (nodes_mm[m * 3 + 2] - xn[2]) / h, q);
#pragma unroll
        for (int i = 0; i < N; ++i) {
#pragma unroll
            for (int j = 0; j < N; ++j) a[i][j] += q[i] * q[j];
        }
    }

    // Rows 1..3 of the inverse are the coefficients of u, v, t, i.e. the gradient in the scaled
    // frame.
    const int grad_rows[3] = {1, 2, 3};
    double z[3][N];
    if (!solve_spd_columns<N>(a, grad_rows, 3, z)) return false;

    for (int p = begin; p < end; ++p) {
        const int m = pidx[p];
        double q[N];
        Basis::eval((nodes_mm[m * 3 + 0] - xn[0]) / h, (nodes_mm[m * 3 + 1] - xn[1]) / h,
                    (nodes_mm[m * 3 + 2] - xn[2]) / h, q);
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            double acc = 0.0;
#pragma unroll
            for (int i = 0; i < N; ++i) acc += z[c][i] * q[i];
            w[p * 3 + c] = static_cast<float>(acc / h);
        }
    }
    return true;
}

// One thread per slot over the node patch, taking the harmonic fit where it is well posed and a
// plain linear one where it is not. status records which rung each slot took: 0 harmonic,
// 1 linear, 2 no gradient determined.
__global__ void hpr_weights_kernel(const double* __restrict__ nodes_mm,
                                   const int* __restrict__ pptr, const int* __restrict__ pidx,
                                   const int* __restrict__ slot_node, float* __restrict__ w,
                                   int* __restrict__ status, int n_slots) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;

    const int begin = pptr[s], end = pptr[s + 1];
    const int node = slot_node[s];
    const double xn[3] = {nodes_mm[node * 3 + 0], nodes_mm[node * 3 + 1],
                          nodes_mm[node * 3 + 2]};

    double h = 0.0;
    for (int p = begin; p < end; ++p) {
        const int m = pidx[p];
#pragma unroll
        for (int c = 0; c < 3; ++c) h = fmax(h, fabs(nodes_mm[m * 3 + c] - xn[c]));
    }

    if (h > 0.0) {
        if (fit_gradient_weights<HarmonicBasis>(nodes_mm, pidx, begin, end, xn, h, w)) {
            status[s] = 0;
            return;
        }
        if (fit_gradient_weights<LinearBasis>(nodes_mm, pidx, begin, end, xn, h, w)) {
            status[s] = 1;
            return;
        }
    }

    // Only reachable when the patch is degenerate enough that no gradient is determined, e.g.
    // every node of it coplanar. Zero weights leave E = -dA/dt at that slot rather than a NaN.
    for (int p = begin; p < end; ++p) {
        w[p * 3 + 0] = w[p * 3 + 1] = w[p * 3 + 2] = 0.f;
    }
    status[s] = 2;
}

// One thread per slot, K placements at a time.
//   grad_v[s] = Σ_m W[s,m] · v[m]        (fp64: the difference against dA/dt cancels)
//   E[s]      = −grad_v[s] − dA/dt(x_n)
// The nodal dA/dt is used exactly here rather than the element average the raw path builds, so
// the only approximated part of E is the one the FEM actually approximates.
template <int K>
__global__ void hpr_grad_kernel(const double* __restrict__ v_block, const float* __restrict__ w,
                                const int* __restrict__ pptr, const int* __restrict__ pidx,
                                const int* __restrict__ slot_node, ConstPtrPack dadt_nodes,
                                PtrPack e_slots, int n_slots, int stride) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;

    double acc[K][3];
#pragma unroll
    for (int c = 0; c < K; ++c) acc[c][0] = acc[c][1] = acc[c][2] = 0.0;

    const int begin = pptr[s], end = pptr[s + 1];
    for (int p = begin; p < end; ++p) {
        const double wx = static_cast<double>(w[p * 3 + 0]);
        const double wy = static_cast<double>(w[p * 3 + 1]);
        const double wz = static_cast<double>(w[p * 3 + 2]);
        const std::int64_t base = static_cast<std::int64_t>(pidx[p]) * stride;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            const double vm = v_block[base + c];
            acc[c][0] += wx * vm;
            acc[c][1] += wy * vm;
            acc[c][2] += wz * vm;
        }
    }

    const int node = slot_node[s];
#pragma unroll
    for (int c = 0; c < K; ++c) {
        const float* da = dadt_nodes.p[c] + node * 3;
        float* dst = e_slots.p[c] + s * 3;
        dst[0] = static_cast<float>(-acc[c][0] - static_cast<double>(da[0]));
        dst[1] = static_cast<float>(-acc[c][1] - static_cast<double>(da[1]));
        dst[2] = static_cast<float>(-acc[c][2] - static_cast<double>(da[2]));
    }
}

// --- apply ------------------------------------------------------------------------------------

// One thread per slot, K placements at a time. The shared reads (w, ptr, idx) are paid once for
// the whole block. K is a template parameter so the accumulators stay in registers; a runtime k
// would put them in local memory.
//
// The walk order over ptr/idx does not depend on K, so column c comes back bit-identical at every
// block width, and the K=1 instantiation is what the serial launcher calls.
template <int K>
__global__ void recover_nodes_kernel(ConstPtrPack e_in, const float* __restrict__ w,
                                     const int* __restrict__ ptr, const int* __restrict__ idx,
                                     PtrPack e_slots, int n_slots) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;

    float acc[K][3];
#pragma unroll
    for (int c = 0; c < K; ++c) acc[c][0] = acc[c][1] = acc[c][2] = 0.f;

    const int begin = ptr[s], end = ptr[s + 1];
    for (int p = begin; p < end; ++p) {
        const float wp = w[p];
        const int e = idx[p] >> 2;
#pragma unroll
        for (int c = 0; c < K; ++c) {
            const float* src = e_in.p[c] + e * 3;
            acc[c][0] += wp * src[0];
            acc[c][1] += wp * src[1];
            acc[c][2] += wp * src[2];
        }
    }
#pragma unroll
    for (int c = 0; c < K; ++c) {
        float* dst = e_slots.p[c] + s * 3;
        dst[0] = acc[c][0];
        dst[1] = acc[c][1];
        dst[2] = acc[c][2];
    }
}

// One thread per tet: sample the recovered field at the barycentre, which for P1 is the mean of
// the four corner slots. slot_of_corner is the identity on tet_nodes for the node-slot modes and
// the (node, tissue) map for the tissue-restricted ones. Accumulated in fp64 before the fp32
// store, matching reconstruct.cu.
template <int K>
__global__ void recover_elements_kernel(ConstPtrPack e_slots,
                                        const int* __restrict__ slot_of_corner, PtrPack e_out,
                                        PtrPack magn_out, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    int slots[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) slots[i] = slot_of_corner[e * 4 + i];

#pragma unroll
    for (int c = 0; c < K; ++c) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float* src = e_slots.p[c] + slots[i] * 3;
            sx += static_cast<double>(src[0]);
            sy += static_cast<double>(src[1]);
            sz += static_cast<double>(src[2]);
        }
        sx *= 0.25;
        sy *= 0.25;
        sz *= 0.25;
        float* dst = e_out.p[c] + e * 3;
        dst[0] = static_cast<float>(sx);
        dst[1] = static_cast<float>(sy);
        dst[2] = static_cast<float>(sz);
        magn_out.p[c][e] = static_cast<float>(sqrt(sx * sx + sy * sy + sz * sz));
    }
}

ConstPtrPack pack_const(const float* const* src, int k) {
    ConstPtrPack out{};
    for (int c = 0; c < k; ++c) out.p[c] = src[c];
    return out;
}

PtrPack pack(float* const* src, int k) {
    PtrPack out{};
    for (int c = 0; c < k; ++c) out.p[c] = src[c];
    return out;
}

}  // namespace

void launch_mark_outer_boundary(const int* tet_nodes, const int* ptr, const int* idx,
                                int* is_boundary, int n_tet, cudaStream_t stream) {
    const std::int64_t n_face = static_cast<std::int64_t>(n_tet) * 4;
    if (const unsigned blocks = grid_for(n_face)) {
        mark_outer_boundary_kernel<<<blocks, kBlock, 0, stream>>>(
            tet_nodes, ptr, idx, is_boundary, static_cast<int>(n_face));
    }
}

void launch_spr_weights(const double* nodes_mm, const int* tet_nodes, const float* vols,
                        const int* ptr, const int* idx, const int* slot_node,
                        const int* is_boundary, float* w, int* n_fallback, int n_slots,
                        cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_slots)) {
        spr_weights_kernel<<<blocks, kBlock, 0, stream>>>(nodes_mm, tet_nodes, vols, ptr, idx,
                                                          slot_node, is_boundary, w, n_fallback,
                                                          n_slots);
    }
}

void launch_hpr_weights(const double* nodes_mm, const int* pptr, const int* pidx,
                        const int* slot_node, float* w, int* status, int n_slots,
                        cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_slots)) {
        hpr_weights_kernel<<<blocks, kBlock, 0, stream>>>(nodes_mm, pptr, pidx, slot_node, w,
                                                          status, n_slots);
    }
}

void launch_hpr_grad(const double* v_block, const float* w, const int* pptr, const int* pidx,
                     const int* slot_node, const float* const* dadt_nodes, float* const* e_slots,
                     int n_slots, int k, int stride, cudaStream_t stream) {
    const ConstPtrPack da = pack_const(dadt_nodes, k);
    const PtrPack out = pack(e_slots, k);
    const unsigned blocks = grid_for(n_slots);
    if (!blocks) return;
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(k, "hpr_grad: unsupported block width", [&](auto kk) {
        hpr_grad_kernel<decltype(kk)::value><<<blocks, kBlock, 0, stream>>>(
            v_block, w, pptr, pidx, slot_node, da, out, n_slots, stride);
    });
}

void launch_recover_nodes(const float* const* e_in, const float* w, const int* ptr, const int* idx,
                          float* const* e_slots, int n_slots, int k, cudaStream_t stream) {
    const ConstPtrPack in = pack_const(e_in, k);
    const PtrPack out = pack(e_slots, k);
    const unsigned blocks = grid_for(n_slots);
    if (!blocks) return;
    // Every width up to kMaxStageBlock is compiled, not just the solver's {1,2,4,8}: a chunk
    // carries however many placements the sweep had left, and padding the pack by replicating a
    // pointer would recompute a column to throw it away.
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(k, "recover_nodes: unsupported block width", [&](auto kk) {
        recover_nodes_kernel<decltype(kk)::value>
            <<<blocks, kBlock, 0, stream>>>(in, w, ptr, idx, out, n_slots);
    });
}

void launch_recover_elements(const float* const* e_slots, const int* slot_of_corner,
                             float* const* e_out, float* const* magn_out, int n_tet, int k,
                             cudaStream_t stream) {
    const ConstPtrPack in = pack_const(e_slots, k);
    const PtrPack eo = pack(e_out, k);
    const PtrPack mo = pack(magn_out, k);
    const unsigned blocks = grid_for(n_tet);
    if (!blocks) return;
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(
        k, "recover_elements: unsupported block width", [&](auto kk) {
            recover_elements_kernel<decltype(kk)::value>
                <<<blocks, kBlock, 0, stream>>>(in, slot_of_corner, eo, mo, n_tet);
        });
}
