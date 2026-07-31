#include "device_math.cuh"
#include "kernels.hpp"

#include <cfloat>

// Batched coil placement: for each target centre, find the closest point on the scalp surface
// (closest-point-on-triangle over all skin triangles, Ericson), then build the coil-to-head frame.
// One block per placement; threads stride over triangles and reduce to the nearest triangle (ties
// broken by lowest index, matching cupy's argmin). All double precision.

namespace {

// Smallest sin^2 of the angle between the handle direction and the scalp tangent plane that
// still defines an in-plane axis: (1e-6 rad)^2. Below it the tangential component is numerical
// noise and the coil's rotation about the normal is arbitrary.
constexpr double kMinSin2 = 1e-12;

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

__global__ void place_kernel(const double* __restrict__ centers, const double* __restrict__ handles,
                             const double* __restrict__ dists, const double* __restrict__ av,
                             const double* __restrict__ bv, const double* __restrict__ cv,
                             const double* __restrict__ tnorm, double* __restrict__ out,
                             int* __restrict__ degenerate, int n_pl, int n_tri) {
    __shared__ double sdist[kBlock];
    __shared__ int stri[kBlock];

    const int p = blockIdx.x;
    if (p >= n_pl) return;
    const double center[3] = {centers[p * 3], centers[p * 3 + 1], centers[p * 3 + 2]};

    double best = DBL_MAX;
    int btri = -1;
    double q[3];
    for (int j = threadIdx.x; j < n_tri; j += kBlock) {
        double d2 = closest_on_tri(center, &av[j * 3], &bv[j * 3], &cv[j * 3], q);
        if (d2 < best || (d2 == best && j < btri)) {
            best = d2;
            btri = j;
        }
    }
    sdist[threadIdx.x] = best;
    stri[threadIdx.x] = btri;
    __syncthreads();

    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double db = sdist[threadIdx.x + s];
            int tb = stri[threadIdx.x + s];
            if (db < sdist[threadIdx.x] ||
                (db == sdist[threadIdx.x] && tb < stri[threadIdx.x])) {
                sdist[threadIdx.x] = db;
                stri[threadIdx.x] = tb;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x != 0) return;
    const int tri = stri[0];
    double proj[3];
    closest_on_tri(center, &av[tri * 3], &bv[tri * 3], &cv[tri * 3], proj);
    const double normal[3] = {tnorm[tri * 3], tnorm[tri * 3 + 1], tnorm[tri * 3 + 2]};
    const double z[3] = {-normal[0], -normal[1], -normal[2]};

    // In-plane axis from the handle. It is undefined when the handle direction has no
    // component in the tangent plane, in which case fall back to a canonical axis and say so
    // through `degenerate`: the frame stays orthonormal either way, so no caller can be handed
    // a NaN, and the ones that supplied a handle can reject the placement.
    double y[3];
    bool from_handle = false;
    if (handles != nullptr) {
        for (int i = 0; i < 3; ++i) y[i] = handles[p * 3 + i] - proj[i];
        if (dot3(y, y) > 0.0) {
            const double inv = rnorm3d(y[0], y[1], y[2]);
            for (int i = 0; i < 3; ++i) y[i] *= inv;
            const double cz = dot3(y, z);
            // After removing the out-of-plane part, |y|^2 = 1 - cos^2 = sin^2 of the angle to
            // the tangent plane, so this test is exact and scale-free. kMinSin2 is (1e-6 rad)^2
            // -- a numerical-annihilation threshold, not a geometry-quality one: on a curved
            // scalp a legitimate handle can sit quite close to the normal.
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
        y[0] = z[1] * ref[2] - z[2] * ref[1];
        y[1] = z[2] * ref[0] - z[0] * ref[2];
        y[2] = z[0] * ref[1] - z[1] * ref[0];
        const double inv = rnorm3d(y[0], y[1], y[2]);
        for (int i = 0; i < 3; ++i) y[i] *= inv;
    }
    if (degenerate != nullptr) degenerate[p] = from_handle ? 0 : 1;
    const double x[3] = {y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2],
                         y[0] * z[1] - y[1] * z[0]};

    double* o = &out[p * 16];
    for (int i = 0; i < 16; ++i) o[i] = 0.0;
    for (int r = 0; r < 3; ++r) {
        o[r * 4 + 0] = x[r];
        o[r * 4 + 1] = y[r];
        o[r * 4 + 2] = z[r];
        o[r * 4 + 3] = proj[r] + dists[p] * normal[r];
    }
    o[15] = 1.0;
}

}  // namespace

void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* a, const double* b, const double* c, const double* tnorm,
                  double* out, int* degenerate, int n_pl, int n_tri, cudaStream_t stream) {
    // One block per placement; threads inside it stride over triangles.
    if (const unsigned blocks = grid_for(n_pl, 1)) {
        place_kernel<<<blocks, kBlock, 0, stream>>>(centers, handles, dists, a, b, c, tnorm, out,
                                                    degenerate, n_pl, n_tri);
    }
}
