#include "coil/coil.hpp"
#include "core/device_math.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda/cmath>
#include <cuda/functional>

#include <math_constants.h>

#include <cfloat>
#include <climits>
#include <stdexcept>

// Batched coil placement: for each target centre, find the closest point on the scalp surface
// (closest-point-on-triangle over all skin triangles, Ericson), then build the coil-to-head
// frame. All double precision.
//
// The scan runs as a block per (placement, triangle chunk) and the frame as a thread per
// placement. Selection is a min over the total order (distance^2, triangle index), so the
// chunk count never changes the result.

namespace {

// Smallest sin^2 of the angle between the handle direction and the scalp tangent plane that
// still defines an in-plane axis: (1e-6 rad)^2. Below it the tangential component is numerical
// noise and the coil's rotation about the normal is arbitrary.
constexpr double kMinSin2 = 1e-12;

// Loses the (distance^2, index) comparison to every real triangle, so threads that scan nothing
// need no special case.
constexpr int kNoTri = INT_MAX;

// The corners (n_tri, 3, 3) and outward normals (n_tri, 3) of the scalp triangles.
struct Scalp {
    TriVertView<const double> verts;
    Vec3View<const double> normals;

    __device__ double closest(const double* p, int j, double* q) const;
};

// (n_pl, 4, 4) row-major coil-to-head transforms.
using TransformView =
    cuda::std::mdspan<double, cuda::std::extents<int, cuda::std::dynamic_extent, 4, 4>>;

// Ordered by (d2, triangle index), so ties break to the lower index, matching cupy's argmin. A
// minimum over this total order is associative and commutative, so cub may combine it in
// whatever order it likes: unlike the float sums elsewhere in this directory, no association can
// change what it returns.
struct Candidate {
    double d2;
    int tri;

    __device__ friend bool operator<(Candidate x, Candidate y) {
        return x.d2 < y.d2 || (x.d2 == y.d2 && x.tri < y.tri);
    }
};

// Closest point q on triangle (a,b,c) to p; returns squared distance. Full Ericson ordering:
// vertex A, vertex B, edge AB, vertex C, edge AC, edge BC, interior.
__device__ double closest_on_tri(const double* p, const double* a, const double* b,
                                 const double* c, double* q) {
    double ab[3], ac[3], ap[3];
    for (int i = 0; i < 3; ++i) {
        ab[i] = b[i] - a[i];
        ac[i] = c[i] - a[i];
        ap[i] = p[i] - a[i];
    }
    const double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    bool done = false;
    if (d1 <= 0.0 && d2 <= 0.0) {
        for (int i = 0; i < 3; ++i) q[i] = a[i];
        done = true;
    }
    double bp[3];
    for (int i = 0; i < 3; ++i) bp[i] = p[i] - b[i];
    const double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (!done && d3 >= 0.0 && d4 <= d3) {
        for (int i = 0; i < 3; ++i) q[i] = b[i];
        done = true;
    }
    const double vc = d1 * d4 - d3 * d2;
    if (!done && vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double t = d1 / (d1 - d3);
        for (int i = 0; i < 3; ++i) q[i] = a[i] + t * ab[i];
        done = true;
    }
    double cp[3];
    for (int i = 0; i < 3; ++i) cp[i] = p[i] - c[i];
    const double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (!done && d6 >= 0.0 && d5 <= d6) {
        for (int i = 0; i < 3; ++i) q[i] = c[i];
        done = true;
    }
    const double vb = d5 * d2 - d1 * d6;
    if (!done && vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double t = d2 / (d2 - d6);
        for (int i = 0; i < 3; ++i) q[i] = a[i] + t * ac[i];
        done = true;
    }
    const double va = d3 * d6 - d5 * d4;
    if (!done && va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        for (int i = 0; i < 3; ++i) q[i] = b[i] + t * (c[i] - b[i]);
        done = true;
    }
    if (!done) {
        double denom = 1.0 / (va + vb + vc);
        double v = vb * denom, w = vc * denom;
        for (int i = 0; i < 3; ++i) q[i] = a[i] + v * ab[i] + w * ac[i];
    }
    double dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
    return dx * dx + dy * dy + dz * dz;
}

__device__ double Scalp::closest(const double* p, int j, double* q) const {
    return closest_on_tri(p, &verts(j, 0, 0), &verts(j, 1, 0), &verts(j, 2, 0), q);
}

// One block per (placement, chunk): the closest triangle in this block's slice of the scan.
__global__ void place_scan_kernel(const double* __restrict__ centers, Scalp scalp,
                                  Candidate* __restrict__ cand, int n_tri, int n_chunks) {
    using BlockReduce = cub::BlockReduce<Candidate, kBlock>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int p = blockIdx.x;
    const int chunk = blockIdx.y;
    const int per = cuda::ceil_div(n_tri, n_chunks);
    const int lo = chunk * per;
    const int hi = min(lo + per, n_tri);
    const auto ctr = unsized<Vec3View<const double>>(centers);
    const double center[3] = {ctr(p, 0), ctr(p, 1), ctr(p, 2)};

    // A thread's own candidates arrive in increasing j, so only a strict improvement can win
    // here; ties across threads are settled by the reduction below.
    Candidate best{DBL_MAX, kNoTri};
    double q[3];
    for (int j = lo + threadIdx.x; j < hi; j += kBlock) {
        const double d2 = scalp.closest(center, j, q);
        if (d2 < best.d2) best = {d2, j};
    }

    const Candidate won = BlockReduce(temp).Reduce(best, cuda::minimum<>{});

    if (threadIdx.x == 0) {
        cand[static_cast<size_t>(p) * n_chunks + chunk] = won;
    }
}

// One thread per placement: pick the winning chunk candidate, then build that placement's frame.
__global__ void place_frame_kernel(const double* __restrict__ centers,
                                   const double* __restrict__ handles,
                                   const double* __restrict__ dists, Scalp scalp,
                                   const Candidate* __restrict__ cand, double* __restrict__ out,
                                   int* __restrict__ degenerate, int n_pl, int n_chunks) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pl) return;

    const Candidate* mine = cand + static_cast<size_t>(p) * n_chunks;
    Candidate best = mine[0];
    for (int i = 1; i < n_chunks; ++i) {
        if (mine[i] < best) best = mine[i];
    }
    const int tri = best.tri;
    const TransformView o(out, n_pl);
    // Only a non-finite centre finds no triangle, since every distance to it compares false.
    if (tri == kNoTri) {
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col) o(p, r, col) = CUDART_NAN;
        }
        if (degenerate != nullptr) degenerate[p] = 1;
        return;
    }

    const Vec3View<const double> ctr(centers, n_pl);
    const double center[3] = {ctr(p, 0), ctr(p, 1), ctr(p, 2)};
    double proj[3];
    scalp.closest(center, tri, proj);
    const double normal[3] = {scalp.normals(tri, 0), scalp.normals(tri, 1),
                              scalp.normals(tri, 2)};
    const double z[3] = {-normal[0], -normal[1], -normal[2]};

    // In-plane axis from the handle. It is undefined when the handle direction has no
    // component in the tangent plane, in which case fall back to a canonical axis and say so
    // through `degenerate`: the frame stays orthonormal either way, so no caller can be handed
    // a NaN, and the ones that supplied a handle can reject the placement.
    double y[3];
    bool from_handle = false;
    if (handles != nullptr) {
        const Vec3View<const double> hnd(handles, n_pl);
        for (int i = 0; i < 3; ++i) y[i] = hnd(p, i) - proj[i];
        if (dot3(y, y) > 0.0) {
            const double inv = rnorm3d(y[0], y[1], y[2]);
            for (int i = 0; i < 3; ++i) y[i] *= inv;
            const double cz = dot3(y, z);
            // After removing the out-of-plane part, |y|^2 = 1 - cos^2 = sin^2 of the angle to
            // the tangent plane, so this test is exact and scale-free. kMinSin2 is a
            // numerical-annihilation threshold, not a geometry-quality one: on a curved scalp a
            // legitimate handle can sit quite close to the normal.
            if ((1.0 - cz * cz) > kMinSin2) {
                for (int i = 0; i < 3; ++i) y[i] -= z[i] * cz;
                const double inv2 = rnorm3d(y[0], y[1], y[2]);
                for (int i = 0; i < 3; ++i) y[i] *= inv2;
                from_handle = true;
            }
        }
    }
    if (!from_handle) {
        // Any deterministic in-plane axis serves; crossing z against the canonical direction it
        // is least aligned with keeps that cross product far from zero.
        int m = 0;
        if (fabs(z[1]) < fabs(z[m])) m = 1;
        if (fabs(z[2]) < fabs(z[m])) m = 2;
        double ref[3] = {0.0, 0.0, 0.0};
        ref[m] = 1.0;
        cross3(z, ref, y);
        const double inv = rnorm3d(y[0], y[1], y[2]);
        for (int i = 0; i < 3; ++i) y[i] *= inv;
    }
    if (degenerate != nullptr) degenerate[p] = from_handle ? 0 : 1;
    double x[3];
    cross3(y, z, x);

    for (int r = 0; r < 3; ++r) {
        o(p, r, 0) = x[r];
        o(p, r, 1) = y[r];
        o(p, r, 2) = z[r];
        o(p, r, 3) = proj[r] + dists[p] * normal[r];
        o(p, 3, r) = 0.0;
    }
    o(p, 3, 3) = 1.0;
}

}  // namespace

void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* verts, const double* tnorm, double* out, int* degenerate,
                  int n_pl, int n_tri, cudaStream_t stream) {
    if (n_tri <= 0) throw std::invalid_argument("place: the scalp has no triangles");
    if (n_pl <= 0) return;

    const int n_chunks = scan_chunks(n_pl, n_tri);
    const Scalp scalp{TriVertView<const double>(verts, n_tri),
                      Vec3View<const double>(tnorm, n_tri)};

    const ScratchBuffer<Candidate> cand = scratch_alloc<Candidate>(
        static_cast<std::size_t>(n_pl) * n_chunks, stream, "place", "candidates");

    // Placements go on x, the only grid axis not capped at 65535 blocks.
    place_scan_kernel<<<dim3(n_pl, n_chunks), kBlock, 0, stream>>>(centers, scalp, cand.get(),
                                                                   n_tri, n_chunks);
    place_frame_kernel<<<grid_for(n_pl), kBlock, 0, stream>>>(
        centers, handles, dists, scalp, cand.get(), out, degenerate, n_pl, n_chunks);
    check_cuda(cudaGetLastError(), "place", "launch");
}
