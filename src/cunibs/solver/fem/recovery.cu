#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/std/array>

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
// Every reduction here walks ptr/idx in a fixed order, which is what keeps the recovered field
// bit-reproducible and independent of the block width. The SPR kernels give each output slot a
// thread and walk its segment in one loop; the harmonic fit gives each slot a warp and deals the
// normal matrix's entries round the lanes, so each entry is still summed over the segment in order
// while no lane has to hold the whole matrix. The only atomic is the integer fallback counter.

namespace {

// Largest sum|w| a patch fit may have before it is rejected for the volume-weighted average.
// Weights sum to one, so this bounds the recovery's amplification; 8 leaves ample room for a
// legitimately anisotropic patch while catching the near-degenerate ones.
constexpr double kMaxLebesgue = 8.0;

// The same idea for the gradient fits. A gradient functional carries a length unit, so the
// dimensionless quantity is h * max_c sum_m |w[m,c]|: how far the recovered gradient can travel
// outside the range of the potential it is fitted to, relative to the patch's own scale. Both
// harmonic rungs sit an order of magnitude under this on real patches, so the bound catches only
// the tail, where a determined-but-ill-conditioned patch turns the potential's rounding into
// field error. Rejection drops a rung rather than failing; the rung below is still a fit.
constexpr double kMaxGradientAmplification = 50.0;

// The potential fit runs one warp per slot. Two warps per block, because the fit's shared scratch
// is what caps blocks per SM and a narrower block leaves room for more of them. Patch entries are
// staged a chunk at a time so the basis is evaluated once per entry and read by every lane.
constexpr int kFitWarps = 2;
constexpr int kFitBlock = kWarp * kFitWarps;
constexpr int kFitChunk = kWarp;

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

    const Tet4View<const int> tn(tet_nodes, n_face >> 2);
    int a[3];
    int m = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i != skip) a[m++] = tn(e, i);
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

// Cholesky-solve A z = e_0 for an N x N SPD A held in registers. Returns false on a non-positive
// pivot, which is how a coplanar or under-filled patch reports itself. Only the SPR modes use
// this, at N = 4, where the factor is 16 doubles and stays in registers. The harmonic rungs go up
// to N = 16, whose factor does not, and factor per warp instead; see fit_gradient_weights.
template <int N>
__device__ bool solve_spd_e0(const double a[N][N], double out[N]) {
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
    double y[N];
    for (int i = 0; i < N; ++i) {
        double s = (i == 0) ? 1.0 : 0.0;
        for (int k = 0; k < i; ++k) s -= l[i][k] * y[k];
        y[i] = s / l[i][i];
    }
    for (int i = N - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < N; ++k) s -= l[k][i] * out[k];
        out[i] = s / l[i][i];
    }
    return true;
}

__device__ __forceinline__ void tet_barycentre(Vec3View<const double> nodes_mm,
                                               Tet4View<const int> tet_nodes, int e,
                                               double b[3]) {
    b[0] = b[1] = b[2] = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = tet_nodes(e, i);
        b[0] += nodes_mm(n, 0);
        b[1] += nodes_mm(n, 1);
        b[2] += nodes_mm(n, 2);
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
// Boundary nodes take a volume-weighted average instead, matching SimNIBS, and so does any patch
// whose fit is ill-posed. Both are a select over the whole slot, never a blend: a failed solve
// must not leak into a good one.
//
// is_boundary is indexed by node, not by slot, and is read through slot_node. Which nodes it
// marks is the caller's choice of boundary, one per recovery mode; see fem/recovery.py.
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

    const auto nodes = unsized<Vec3View<const double>>(nodes_mm);
    const auto tn = unsized<Tet4View<const int>>(tet_nodes);
    const double xn[3] = {nodes(node, 0), nodes(node, 1), nodes(node, 2)};

    bool fit = (count >= 4) && !(is_boundary && is_boundary[node]);
    double z[4] = {1.0, 0.0, 0.0, 0.0};
    double h = 1.0;

    if (fit) {
        h = 0.0;
        for (int p = begin; p < end; ++p) {
            double b[3];
            tet_barycentre(nodes, tn, idx[p] >> 2, b);
#pragma unroll
            for (int c = 0; c < 3; ++c) h = fmax(h, fabs(b[c] - xn[c]));
        }
        fit = h > 0.0;
    }
    if (fit) {
        double a[4][4] = {};
        for (int p = begin; p < end; ++p) {
            double b[3];
            tet_barycentre(nodes, tn, idx[p] >> 2, b);
            const double q[4] = {1.0, (b[0] - xn[0]) / h, (b[1] - xn[1]) / h,
                                 (b[2] - xn[2]) / h};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) a[i][j] += q[i] * q[j];
            }
        }
        fit = solve_spd_e0<4>(a, z);
    }

    // The weights sum to one, so sum|w|, the patch's Lebesgue constant, is how much the fit
    // can amplify its inputs. A healthy patch sits near 1. A pivot test alone does not catch a
    // patch that is merely very flat: it passes, and then returns weights of order thousands.
    // Every patch reaching here is a full ball in its own volume, the half-balls having gone to
    // the average above, so this is a guard against a malformed mesh rather than a routine path.
    if (fit) {
        double lebesgue = 0.0;
        for (int p = begin; p < end; ++p) {
            double b[3];
            tet_barycentre(nodes, tn, idx[p] >> 2, b);
            lebesgue += fabs(spr_corner_weight(z, b, xn, h));
        }
        fit = lebesgue <= kMaxLebesgue;
    }

    if (!fit) {
        // Diagnostic only, and read after the stream drains, so relaxed is the whole ordering.
        if (n_fallback) {
            cuda::atomic_ref<int, cuda::thread_scope_device> fallbacks(*n_fallback);
            fallbacks.fetch_add(1, cuda::std::memory_order_relaxed);
        }
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
        tet_barycentre(nodes, tn, idx[p] >> 2, b);
        w[p] = static_cast<float>(spr_corner_weight(z, b, xn, h));
    }
}

// The 9 harmonic quadratics in the node-centred, radius-scaled frame. Dropping the tenth basis
// function is the whole method: a general quadratic has three independent second derivatives,
// but at an interface or boundary node the patch is a half-ball and cannot determine the one
// normal to the flat side. Laplace's equation ties the three together, so the in-plane curvature
// the patch does resolve pins the one it does not. v is harmonic inside a tissue because the
// source is a divergence-free dipole potential outside the head.
struct HarmonicBasis {
    static constexpr int kTerms = 9;
    // Smallest patch that may attempt the fit. This rung carries no margin of its own because the
    // escalation threshold supplies one; see _MIN_PATCH_NODES in fem/recovery.py.
    static constexpr int kMinPatch = 9;
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

// The top rung. Harmonic polynomials of degree exactly l span 2l+1 dimensions, so degree <= 3 is
// 1 + 3 + 5 + 7 = 16 against the quadratic's 9. Terms 0..8 are HarmonicBasis unchanged, so the two
// agree wherever the cubic coefficients vanish, and terms 9..15 are the seven real solid harmonics
// of degree 3. Gradient extraction is the same either way: every term above linear still has zero
// gradient at the patch centre, so it remains rows 1..3 of the inverse. What the extra terms buy
// is truncation, O(h^3) against O(h^2).
//
// kMinPatch carries a real margin here. A patch with exactly kTerms points makes the normal matrix
// square, and a square fit interpolates rather than averages, which is how a patch passes the
// pivot test and still returns weights of order 1e10.
struct CubicHarmonicBasis {
    static constexpr int kTerms = 16;
    static constexpr int kMinPatch = 20;
    __device__ __forceinline__ static void eval(double u, double v, double t, double p[kTerms]) {
        HarmonicBasis::eval(u, v, t, p);
        const double uu = u * u, vv = v * v, tt = t * t;
        p[9] = t * (2.0 * tt - 3.0 * uu - 3.0 * vv);
        p[10] = u * (4.0 * tt - uu - vv);
        p[11] = v * (4.0 * tt - uu - vv);
        p[12] = t * (uu - vv);
        p[13] = u * v * t;
        p[14] = u * (uu - 3.0 * vv);
        p[15] = v * (3.0 * uu - vv);
    }
};

// The fallback rung: a plain linear fit needs only four non-coplanar patch nodes, so it is
// reachable from any non-degenerate tetrahedron.
struct LinearBasis {
    static constexpr int kTerms = 4;
    static constexpr int kMinPatch = 4;
    __device__ __forceinline__ static void eval(double u, double v, double t, double p[kTerms]) {
        p[0] = 1.0;
        p[1] = u;
        p[2] = v;
        p[3] = t;
    }
};

// The scaling radius: the patch's half-width in the infinity norm, which is what puts the fit in a
// frame where the offsets lie in [-1, 1] and the normal matrix is conditioned on shape rather than
// on head-mesh coordinates. One warp walks the patch; fmax is exact, so splitting the walk across
// lanes and folding the results reaches the same radius a single thread's walk would.
__device__ __forceinline__ double patch_radius(Vec3View<const double> nodes_mm,
                                               const int* __restrict__ pidx, int begin, int end,
                                               const Vec3& xn, int lane) {
    double h = 0.0;
    for (int p = begin + lane; p < end; p += kWarp) {
        const int m = pidx[p];
#pragma unroll
        for (int c = 0; c < 3; ++c) h = fmax(h, fabs(nodes_mm(m, c) - xn[c]));
    }
#pragma unroll
    for (int d = kWarp / 2; d; d >>= 1) h = fmax(h, __shfl_xor_sync(kFullWarp, h, d));
    return h;
}

// Per-warp scratch for one rung of the fit. Sized for the widest basis and reused by the narrower
// rungs, since a warp runs them one after another and never needs two at once.
// Row stride of a basis of N terms in the fit's shared arrays. Lanes that each take a row of their
// own would otherwise sit N doubles apart, which at N = 16 is one bank group for every lane; two
// doubles of padding give the rows distinct bank groups and keep them 16-byte aligned.
__host__ __device__ constexpr int fit_stride(int n_terms) { return n_terms + 2; }

struct FitScratch {
    double* q;     // kFitChunk rows, the staged basis of one chunk of patch entries
    double* a;     // N rows, the normal matrix, overwritten by its Cholesky factor
    double* yz;    // 3 forward-substitution rows, then 3 solution rows
    int* singular; // set when a pivot fails, so the whole warp leaves the rung together
};

// Fit the potential over one slot's node patch in Basis with one warp, then write the three
// gradient weights per patch node. The gradient at the node is the linear coefficients divided by
// the scaling radius, because every term above linear has zero derivative at the centre. Returns
// false when the normal matrix is not positive definite or the fit amplifies beyond
// kMaxGradientAmplification; w may have been written either way, and the caller's next rung
// overwrites the same entries.
//
// The normal matrix lives in shared memory, one copy per warp, because the cubic rung's 16x16 does
// not fit in one thread's registers and spills to local memory if it is held there. Only its lower
// triangle is ever read, so 136 entries are dealt four or five to a lane.
//
// Each a[i][j] is summed over patch entries in pptr order and each substitution runs its k in
// order, so the arithmetic and its sequence are what a per-thread fit would do, not a
// reassociation of it. Only the Lebesgue sums are reassociated, and they decide a threshold rather
// than enter a weight.
template <typename Basis>
__device__ bool fit_gradient_weights(Vec3View<const double> nodes_mm,
                                     const int* __restrict__ pidx, int begin, int end,
                                     const Vec3& xn, double h, Vec3View<float> wv,
                                     const FitScratch& s, int lane) {
    constexpr int N = Basis::kTerms;
    constexpr int Ld = fit_stride(N);
    const double inv_h = 1.0 / h;
    // Lower-triangle entries, and how many of them a lane owns once they are dealt round the warp.
    constexpr int kEntries = N * (N + 1) / 2;
    constexpr int kPerLane = cuda::ceil_div(kEntries, kWarp);
    if (end - begin < Basis::kMinPatch) return false;

    cuda::std::array<int, kPerLane> row, col;
    cuda::std::array<double, kPerLane> acc;
    int mine = 0;
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
        const int k = lane + kWarp * e;
        acc[e] = 0.0;
        row[e] = 0;
        col[e] = 0;
        if (k < kEntries) {
            // Invert k = i(i+1)/2 + j.
            const int i = (cuda::isqrt(8 * k + 1) - 1) / 2;
            row[e] = i;
            col[e] = k - (i * (i + 1)) / 2;
            ++mine;
        }
    }

    for (int base = begin; base < end; base += kFitChunk) {
        const int n = min(kFitChunk, end - base);
        if (lane < n) {
            const int m = pidx[base + lane];
            Basis::eval((nodes_mm(m, 0) - xn[0]) * inv_h, (nodes_mm(m, 1) - xn[1]) * inv_h,
                        (nodes_mm(m, 2) - xn[2]) * inv_h, s.q + lane * Ld);
        }
        __syncwarp();
        for (int p = 0; p < n; ++p) {
            const double* __restrict__ q = s.q + p * Ld;
#pragma unroll
            for (int e = 0; e < kPerLane; ++e) {
                if (e < mine) acc[e] = fma(q[row[e]], q[col[e]], acc[e]);
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int e = 0; e < kPerLane; ++e) {
        if (e < mine) s.a[row[e] * Ld + col[e]] = acc[e];
    }
    if (lane == 0) *s.singular = 0;
    __syncwarp();

    // Cholesky in place: a[i][j] is read exactly once, at the moment l[i][j] is formed, so the
    // factor may overwrite it. Lane i owns row i. a[0][0] is the patch size, so it sets the pivot
    // scale, and it has to be read before the first column overwrites it. 1 / l[i][i] goes in the
    // row's padding at a[i][N]: the column update and both substitutions multiply by it, which
    // keeps fp64 division, a long dependent sequence, off their serial chains.
    const double floor_pivot = 1e-12 * s.a[0];
    __syncwarp();
    for (int j = 0; j < N; ++j) {
        if (lane == j) {
            double d = s.a[j * Ld + j];
            for (int k = 0; k < j; ++k) d -= s.a[j * Ld + k] * s.a[j * Ld + k];
            if (!(d > floor_pivot)) {
                *s.singular = 1;
            } else {
                const double l = sqrt(d);
                s.a[j * Ld + j] = l;
                s.a[j * Ld + N] = 1.0 / l;
            }
        }
        __syncwarp();
        if (*s.singular) return false;
        const double inv_ljj = s.a[j * Ld + N];
        if (lane > j && lane < N) {
            double v = s.a[lane * Ld + j];
            for (int k = 0; k < j; ++k) v -= s.a[lane * Ld + k] * s.a[j * Ld + k];
            s.a[lane * Ld + j] = v * inv_ljj;
        }
        __syncwarp();
    }

    // Rows 1..3 of the inverse are the coefficients of u, v, t, i.e. the gradient in the scaled
    // frame. One right-hand side per lane; the three are independent given the factor.
    double* const z = s.yz + 3 * Ld;
    if (lane < 3) {
        double* const y = s.yz + lane * Ld;
        double* const zc = z + lane * Ld;
        const int rhs_row = lane + 1;
        for (int i = 0; i < N; ++i) {
            double v = (i == rhs_row) ? 1.0 : 0.0;
            for (int k = 0; k < i; ++k) v -= s.a[i * Ld + k] * y[k];
            y[i] = v * s.a[i * Ld + N];
        }
        for (int i = N - 1; i >= 0; --i) {
            double v = y[i];
            for (int k = i + 1; k < N; ++k) v -= s.a[k * Ld + i] * zc[k];
            zc[i] = v * s.a[i * Ld + N];
        }
    }
    __syncwarp();

    // sum accumulates h * sum_m |w[m,c]|, since w is acc / h, in the walk that writes w, so the
    // guard costs no second evaluation of the basis over the patch.
    Vec3 sum{};
    for (int p = begin + lane; p < end; p += kWarp) {
        const int m = pidx[p];
        double q[N];
        Basis::eval((nodes_mm(m, 0) - xn[0]) * inv_h, (nodes_mm(m, 1) - xn[1]) * inv_h,
                    (nodes_mm(m, 2) - xn[2]) * inv_h, q);
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            double grad = 0.0;
#pragma unroll
            for (int i = 0; i < N; ++i) grad = fma(z[c * Ld + i], q[i], grad);
            sum[c] += fabs(grad);
            wv(p, c) = static_cast<float>(grad * inv_h);
        }
    }
#pragma unroll
    for (int d = kWarp / 2; d; d >>= 1) {
#pragma unroll
        for (int c = 0; c < 3; ++c) sum[c] += __shfl_xor_sync(kFullWarp, sum[c], d);
    }
    return fmax(sum[0], fmax(sum[1], sum[2])) <= kMaxGradientAmplification;
}

// One warp per slot over the node patch, taking the widest basis the patch can carry and
// dropping a rung where it cannot. status records which rung each slot took, counted down from
// the widest: 0 cubic, 1 quadratic, 2 linear, 3 no gradient determined. The slots are active[0,
// n), or [0, n) when active is null; a slot's fit reads only its own patch.
__global__ __launch_bounds__(kFitBlock) void hpr_weights_kernel(
    const double* __restrict__ nodes_mm, const int* __restrict__ pptr,
    const int* __restrict__ pidx, const int* __restrict__ slot_node, float* __restrict__ w,
    int* __restrict__ status, const int* __restrict__ active, int n) {
    constexpr int kMaxTerms = CubicHarmonicBasis::kTerms;
    constexpr int kMaxLd = fit_stride(kMaxTerms);
    __shared__ double warp_q[kFitWarps][kFitChunk * kMaxLd];
    __shared__ double warp_a[kFitWarps][kMaxTerms * kMaxLd];
    __shared__ double warp_yz[kFitWarps][6 * kMaxLd];
    __shared__ int warp_singular[kFitWarps];

    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    // Every lane of a warp takes the same slot, so this exit and every rung decision below are
    // warp-uniform and the __syncwarp calls inside a fit always meet the whole warp.
    const int i = blockIdx.x * kFitWarps + warp;
    if (i >= n) return;
    const int s = active ? active[i] : i;
    const FitScratch scratch{warp_q[warp], warp_a[warp], warp_yz[warp], &warp_singular[warp]};

    const int begin = pptr[s], end = pptr[s + 1];
    const int node = slot_node[s];
    const auto nodes = unsized<Vec3View<const double>>(nodes_mm);
    // One 3-vector per patch entry, so its rows are entries of pptr/pidx, not slots.
    const auto wv = unsized<Vec3View<float>>(w);
    const Vec3 xn = {nodes(node, 0), nodes(node, 1), nodes(node, 2)};
    const double h = patch_radius(nodes, pidx, begin, end, xn, lane);

    int rung = 3;
    if (h > 0.0) {
        if (fit_gradient_weights<CubicHarmonicBasis>(nodes, pidx, begin, end, xn, h, wv, scratch,
                                                     lane)) {
            rung = 0;
        } else if (fit_gradient_weights<HarmonicBasis>(nodes, pidx, begin, end, xn, h, wv, scratch,
                                                       lane)) {
            rung = 1;
        } else if (fit_gradient_weights<LinearBasis>(nodes, pidx, begin, end, xn, h, wv, scratch,
                                                     lane)) {
            rung = 2;
        }
    }

    if (rung == 3) {
        // Only reachable when the patch is degenerate enough that no gradient is determined, e.g.
        // every node of it coplanar. Zero weights leave E = -dA/dt at that slot rather than a NaN.
        for (int p = begin + lane; p < end; p += kWarp) {
            wv(p, 0) = wv(p, 1) = wv(p, 2) = 0.f;
        }
    }
    if (lane == 0) status[s] = rung;
}

// The potential as a pair of floats per value, v ~ hi + lo to 2^-48 of v. Built once per call so
// that hpr_grad's inner loop has no fp64 in it at all.
template <int K>
__global__ void split_potential_kernel(const double* __restrict__ v_block, int stride, int n,
                                       float* __restrict__ hi, float* __restrict__ lo) {
    const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::int64_t>(n) * K) return;
    const std::int64_t node = t / K;
    const int c = static_cast<int>(t % K);
    const double v = v_block[node * stride + c];
    const float h = static_cast<float>(v);
    hi[t] = h;
    lo[t] = static_cast<float>(v - static_cast<double>(h));
}

// kHprLanes lanes per slot, K placements at a time.
//   grad_v[s] = Σ_m W[s,m] · (v[m] − v[n_s])
//   E[s]      = −grad_v[s] − dA/dt(x_n)
// The nodal dA/dt is used exactly here rather than the element average the raw path builds, so
// the only approximated part of E is the one the FEM actually approximates.
//
// A patch's gradient weights reproduce constants, so differencing against the slot's own node
// changes nothing in exact arithmetic and removes the potential's common level, which the fp32
// weights' rounding otherwise leaks into the gradient at |v| |W| eps. The difference comes from
// the float pairs: hi_m − hi_0 is exact whenever the two are within a factor of two (Sterbenz)
// and otherwise the difference is as large as the values, so either way it carries a float's
// relative precision. Each lane's share of the patch is a Kahan sum and the lanes combine by
// TwoSum, so the error does not grow with the patch, and nothing here needs the 1:64 fp64 pipe.
//
// The lanes take a slot's entries in turn, so a warp's loads of w and idx are a few contiguous
// runs. A thread walking a whole patch alone would touch a line of its own on every load, and the
// kernel would move several times the bytes it reads.
//
// Every operation is an explicit round-to-nearest intrinsic. nvcc may otherwise fuse a multiply
// into the add after it in one instantiation and not another, which made the result depend on K.
constexpr int kHprLanes = 8;
template <int K>
__global__ void hpr_grad_kernel(const float* __restrict__ v_hi, const float* __restrict__ v_lo,
                                const float* __restrict__ w, const int* __restrict__ pptr,
                                const int* __restrict__ pidx, const int* __restrict__ slot_node,
                                const float* __restrict__ dadt_nodes, std::int64_t dadt_row,
                                int dadt_column, PtrPack e_slots, int n_slots,
                                const int* __restrict__ active, int n) {
    const std::int64_t gt = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int t = static_cast<int>(gt / kHprLanes);
    const int lane = static_cast<int>(gt % kHprLanes);
    const bool live = t < n;
    const int s = !live ? 0 : active ? active[t] : t;

    const auto wv = unsized<Vec3View<const float>>(w);
    const auto hi = unsized<WidthView<const float, K>>(v_hi);
    const auto lo = unsized<WidthView<const float, K>>(v_lo);
    float acc[K][3], comp[K][3];
#pragma unroll
    for (int c = 0; c < K; ++c) {
        acc[c][0] = acc[c][1] = acc[c][2] = 0.f;
        comp[c][0] = comp[c][1] = comp[c][2] = 0.f;
    }

    int node = 0;
    if (live) {
        node = slot_node[s];
        cuda::std::array<float, K> h0, l0;
        load_row<K>(&hi(node, 0), h0);
        load_row<K>(&lo(node, 0), l0);
        const int end = pptr[s + 1];
        for (int p = pptr[s] + lane; p < end; p += kHprLanes) {
            const float wx = wv(p, 0), wy = wv(p, 1), wz = wv(p, 2);
            const int m = pidx[p];
            cuda::std::array<float, K> hm, lm;
            load_row<K>(&hi(m, 0), hm);
            load_row<K>(&lo(m, 0), lm);
#pragma unroll
            for (int c = 0; c < K; ++c) {
                const float d = __fadd_rn(__fsub_rn(hm[c], h0[c]), __fsub_rn(lm[c], l0[c]));
                const float wd[3] = {__fmul_rn(wx, d), __fmul_rn(wy, d), __fmul_rn(wz, d)};
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    // Kahan: comp carries what the running sum dropped, so a lane holds acc - comp.
                    const float y = __fsub_rn(wd[i], comp[c][i]);
                    const float u = __fadd_rn(acc[c][i], y);
                    comp[c][i] = __fsub_rn(__fsub_rn(u, acc[c][i]), y);
                    acc[c][i] = u;
                }
            }
        }
    }

    // A fixed tree over the slot's lanes. TwoSum splits acc_a + acc_b exactly into a sum and its
    // error, which joins the compensations.
#pragma unroll
    for (int off = kHprLanes / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int c = 0; c < K; ++c) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const float a2 = __shfl_down_sync(kFullWarp, acc[c][i], off, kHprLanes);
                const float c2 = __shfl_down_sync(kFullWarp, comp[c][i], off, kHprLanes);
                const float sum = __fadd_rn(acc[c][i], a2);
                const float bb = __fsub_rn(sum, acc[c][i]);
                const float err =
                    __fadd_rn(__fsub_rn(acc[c][i], __fsub_rn(sum, bb)), __fsub_rn(a2, bb));
                comp[c][i] = __fsub_rn(__fadd_rn(comp[c][i], c2), err);
                acc[c][i] = sum;
            }
        }
    }
    if (!live || lane != 0) return;

#pragma unroll
    for (int c = 0; c < K; ++c) {
        const float* const da =
            dadt_nodes + node * dadt_row + static_cast<std::int64_t>(c) * dadt_column;
        const Vec3View<float> dst(e_slots[c], n_slots);
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            dst(s, i) = __fadd_rn(__fsub_rn(-acc[c][i], da[i]), comp[c][i]);
        }
    }
}

// One thread per slot, K placements at a time. The shared reads (w, ptr, idx) are paid once for
// the whole block. K is a template parameter so the accumulators stay in registers; a runtime k
// would put them in local memory.
//
// The walk order over ptr/idx does not depend on K, so column c comes back bit-identical at every
// block width, K = 1 included.
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
            const auto src = unsized<Vec3View<const float>>(e_in[c]);
            acc[c][0] = __fmaf_rn(wp, src(e, 0), acc[c][0]);
            acc[c][1] = __fmaf_rn(wp, src(e, 1), acc[c][1]);
            acc[c][2] = __fmaf_rn(wp, src(e, 2), acc[c][2]);
        }
    }
#pragma unroll
    for (int c = 0; c < K; ++c) {
        const Vec3View<float> dst(e_slots[c], n_slots);
        dst(s, 0) = acc[c][0];
        dst(s, 1) = acc[c][1];
        dst(s, 2) = acc[c][2];
    }
}

// TwoSum: s + e == a + b exactly.
__device__ __forceinline__ float two_sum(float a, float b, float& e) {
    const float s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
    return s;
}

// One thread per tet: sample the recovered field at the barycentre, which for P1 is the mean of
// the four corner slots. slot_of_corner is the identity on tet_nodes for the node-slot modes and
// the (node, tissue) map for the tissue-restricted ones.
//
// The four-term sum is compensated: each pairwise sum keeps its rounding error as a separate term,
// and the errors are added back at the end, which comes close to an fp64 accumulation followed by
// an fp32 store without using the 1:64 fp64 pipe. |E| is the norm of the stored components.
template <int K>
__global__ void recover_elements_kernel(ConstPtrPack e_slots,
                                        const int* __restrict__ slot_of_corner, PtrPack e_out,
                                        PtrPack magn_out, int n_tet,
                                        const int* __restrict__ active, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int e = active ? active[t] : t;

    const Tet4View<const int> soc(slot_of_corner, n_tet);
    int slots[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) slots[i] = soc(e, i);

#pragma unroll
    for (int c = 0; c < K; ++c) {
        const auto src = unsized<Vec3View<const float>>(e_slots[c]);
        float m[3];
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            float e1, e2, e3;
            const float s1 = two_sum(src(slots[0], d), src(slots[1], d), e1);
            const float s2 = two_sum(src(slots[2], d), src(slots[3], d), e2);
            const float s = two_sum(s1, s2, e3);
            m[d] = 0.25f * (s + ((e1 + e2) + e3));
        }
        if (e_out[c] != nullptr) {
            const Vec3View<float> dst(e_out[c], n_tet);
#pragma unroll
            for (int d = 0; d < 3; ++d) dst(e, d) = m[d];
        }
        if (magn_out[c] != nullptr) {
            magn_out[c][e] = sqrtf(fmaf(m[0], m[0], fmaf(m[1], m[1], m[2] * m[2])));
        }
    }
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
                        const int* slot_node, float* w, int* status, const int* active, int n,
                        cudaStream_t stream) {
    // One warp per slot, so the grid is sized in slots per block rather than in threads.
    if (const unsigned blocks = grid_for(n, kFitWarps)) {
        hpr_weights_kernel<<<blocks, kFitBlock, 0, stream>>>(nodes_mm, pptr, pidx, slot_node, w,
                                                             status, active, n);
    }
}

std::size_t hpr_grad_work_floats(int n_nodes, int k) {
    return 2 * static_cast<std::size_t>(n_nodes) * k;
}

void launch_hpr_grad(const double* v_block, const float* w, const int* pptr, const int* pidx,
                     const int* slot_node, const float* dadt_nodes, std::int64_t dadt_row,
                     int dadt_column, float* const* e_slots, int n_slots, const int* active,
                     int n, int k, int stride, int n_nodes, float* work, cudaStream_t stream) {
    const PtrPack out = ptr_pack(e_slots, k);
    if (n <= 0) return;
    const std::size_t count = static_cast<std::size_t>(n_nodes) * k;
    float* const v_hi = work;
    float* const v_lo = v_hi + count;
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(k, "hpr_grad: unsupported block width", [&](auto kk) {
        constexpr int K = decltype(kk)::value;
        const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * kHprLanes);
        if (const unsigned sb = grid_for(static_cast<std::int64_t>(count))) {
            split_potential_kernel<K>
                <<<sb, kBlock, 0, stream>>>(v_block, stride, n_nodes, v_hi, v_lo);
        }
        hpr_grad_kernel<K><<<blocks, kBlock, 0, stream>>>(v_hi, v_lo, w, pptr, pidx, slot_node,
                                                          dadt_nodes, dadt_row, dadt_column, out,
                                                          n_slots, active, n);
    });
}

void launch_recover_nodes(const float* const* e_in, const float* w, const int* ptr, const int* idx,
                          float* const* e_slots, int n_slots, int k, cudaStream_t stream) {
    const ConstPtrPack in = ptr_pack(e_in, k);
    const PtrPack out = ptr_pack(e_slots, k);
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
                             float* const* e_out, float* const* magn_out, int n_tet,
                             const int* active, int n, int k, cudaStream_t stream) {
    const ConstPtrPack in = ptr_pack(e_slots, k);
    const PtrPack eo = ptr_pack(e_out, k);
    const PtrPack mo = ptr_pack(magn_out, k);
    const unsigned blocks = grid_for(n);
    if (!blocks) return;
    dispatch_k<1, 2, 3, 4, 5, 6, 7, 8>(
        k, "recover_elements: unsupported block width", [&](auto kk) {
            recover_elements_kernel<decltype(kk)::value>
                <<<blocks, kBlock, 0, stream>>>(in, slot_of_corner, eo, mo, n_tet, active, n);
        });
}
