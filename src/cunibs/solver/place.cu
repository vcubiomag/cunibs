#include "kernels.hpp"

#include <cuda_runtime.h>
#include <cfloat>

// Batched coil placement: for each target centre, find the closest point on the scalp surface
// (closest-point-on-triangle over all skin triangles, Ericson), then build the coil-to-head frame.
// One block per placement; threads stride over triangles and reduce to the nearest triangle (ties
// broken by lowest index, matching cupy's argmin). All double precision.

namespace {

#ifndef KPLACE_BLOCK
#define KPLACE_BLOCK 256
#endif
constexpr int kBlock = KPLACE_BLOCK;

__device__ inline double dot3(const double* u, const double* v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

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
                             const double* __restrict__ tnorm, double* __restrict__ out, int n_pl,
                             int n_tri) {
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

    double y[3];
    for (int i = 0; i < 3; ++i) y[i] = handles[p * 3 + i] - proj[i];
    double yn = rnorm3d(y[0], y[1], y[2]);
    for (int i = 0; i < 3; ++i) y[i] *= yn;
    double yz = dot3(y, z);
    for (int i = 0; i < 3; ++i) y[i] -= z[i] * yz;
    yn = rnorm3d(y[0], y[1], y[2]);
    for (int i = 0; i < 3; ++i) y[i] *= yn;
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
                  double* out, int n_pl, int n_tri, cudaStream_t stream) {
    place_kernel<<<n_pl, kBlock, 0, stream>>>(centers, handles, dists, a, b, c, tnorm, out, n_pl,
                                              n_tri);
}
