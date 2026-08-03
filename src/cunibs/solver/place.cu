#include "device_math.cuh"
#include "kernels.hpp"

#include <algorithm>
#include <cfloat>
#include <climits>

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

// Fewest triangles worth a block of its own; below this the block reduction outweighs the scan.
constexpr int kMinTrisPerChunk = 1024;

// Loses the (distance^2, index) comparison to every real triangle, so threads that scan nothing
// need no special case.
constexpr int kNoTri = INT_MAX;

struct Scalp {
    const double* a;
    const double* b;
    const double* c;
    const double* normals;
};

struct Candidate {
    double d2;
    int tri;
};

// Ties break to the lower triangle index, matching cupy's argmin.
__device__ __forceinline__ bool closer(Candidate x, Candidate y) {
    return x.d2 < y.d2 || (x.d2 == y.d2 && x.tri < y.tri);
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

// One block per (placement, chunk): the closest triangle in this block's slice of the scan.
__global__ void place_scan_kernel(const double* __restrict__ centers, Scalp scalp,
                                  Candidate* __restrict__ cand, int n_tri, int n_chunks) {
    __shared__ double sdist[kBlock];
    __shared__ int stri[kBlock];

    const int p = blockIdx.x;
    const int chunk = blockIdx.y;
    const int per = (n_tri + n_chunks - 1) / n_chunks;
    const int lo = chunk * per;
    const int hi = min(lo + per, n_tri);
    const double center[3] = {centers[p * 3], centers[p * 3 + 1], centers[p * 3 + 2]};

    // A thread's own candidates arrive in increasing j, so only a strict improvement can win
    // here; ties across threads are settled by the reduction below.
    Candidate best{DBL_MAX, kNoTri};
    double q[3];
    for (int j = lo + threadIdx.x; j < hi; j += kBlock) {
        const double d2 =
            closest_on_tri(center, &scalp.a[j * 3], &scalp.b[j * 3], &scalp.c[j * 3], q);
        if (d2 < best.d2) best = {d2, j};
    }

    sdist[threadIdx.x] = best.d2;
    stri[threadIdx.x] = best.tri;
    __syncthreads();
    for (int step = kBlock / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            const Candidate other{sdist[threadIdx.x + step], stri[threadIdx.x + step]};
            if (closer(other, {sdist[threadIdx.x], stri[threadIdx.x]})) {
                sdist[threadIdx.x] = other.d2;
                stri[threadIdx.x] = other.tri;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        cand[static_cast<size_t>(p) * n_chunks + chunk] = {sdist[0], stri[0]};
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
        if (closer(mine[i], best)) best = mine[i];
    }
    const int tri = best.tri;

    const double center[3] = {centers[p * 3], centers[p * 3 + 1], centers[p * 3 + 2]};
    double proj[3];
    closest_on_tri(center, &scalp.a[tri * 3], &scalp.b[tri * 3], &scalp.c[tri * 3], proj);
    const double normal[3] = {scalp.normals[tri * 3], scalp.normals[tri * 3 + 1],
                              scalp.normals[tri * 3 + 2]};
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

// Cached because nothing changes the current device.
int sm_count() {
    static const int count = [] {
        int device = 0, sms = 0;
        check_cuda(cudaGetDevice(&device), "place", "getDevice");
        check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "place",
                   "getAttribute(multiProcessorCount)");
        return sms;
    }();
    return count;
}

}  // namespace

void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* a, const double* b, const double* c, const double* tnorm,
                  double* out, int* degenerate, int n_pl, int n_tri, cudaStream_t stream) {
    if (n_pl <= 0 || n_tri <= 0) return;

    // Split a placement's scan across blocks only while the placements alone leave SMs idle, and
    // never past a block per kMinTrisPerChunk triangles. Both terms are at least 1.
    const int for_occupancy = (sm_count() + n_pl - 1) / n_pl;
    const int for_size = (n_tri + kMinTrisPerChunk - 1) / kMinTrisPerChunk;
    const int n_chunks = std::min(for_occupancy, for_size);
    const Scalp scalp{a, b, c, tnorm};

    // cudaMallocAsync is illegal inside a CUDA-graph capture; placement runs outside the
    // solver's captured region.
    Candidate* cand = nullptr;
    check_cuda(
        cudaMallocAsync(&cand, static_cast<size_t>(n_pl) * n_chunks * sizeof(Candidate), stream),
        "place", "mallocAsync(cand)");

    // Placements go on x, the only grid axis not capped at 65535 blocks.
    place_scan_kernel<<<dim3(n_pl, n_chunks), kBlock, 0, stream>>>(centers, scalp, cand, n_tri,
                                                                   n_chunks);
    place_frame_kernel<<<grid_for(n_pl), kBlock, 0, stream>>>(centers, handles, dists, scalp, cand,
                                                              out, degenerate, n_pl, n_chunks);

    check_cuda(cudaFreeAsync(cand, stream), "place", "freeAsync(cand)");
}
