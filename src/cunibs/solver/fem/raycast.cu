#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cub/block/block_reduce.cuh>
#include <cuda/cmath>

#include <limits>

// Moller-Trumbore first-hit ray casting against a triangle surface. There is no acceleration
// structure: every ray is tested against every triangle, as a block per (ray, triangle chunk).
// The reduction is a minimum over distance, so the chunk count never changes the result.

namespace {

// Below this the ray is parallel to the triangle plane and the intersection is undefined.
constexpr double kParallelEps = 1e-12;

constexpr double kMiss = std::numeric_limits<double>::infinity();

// (n_tri, 3, 3): the three corner coordinates of each triangle.
using VertsView =
    cuda::std::mdspan<const double, cuda::std::extents<int, cuda::std::dynamic_extent, 3, 3>>;

struct MinOp {
    __device__ __forceinline__ double operator()(double x, double y) const { return fmin(x, y); }
};

// Distance along `dir` to triangle j, or kMiss where the ray misses it or hits it nearer than
// min_t. `dir` need not be normalised; the distance comes back in its units.
__device__ double intersect(const double* origin, const double* dir, VertsView verts, int j,
                            double min_t) {
    double e1[3], e2[3], s[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        e1[i] = verts(j, 1, i) - verts(j, 0, i);
        e2[i] = verts(j, 2, i) - verts(j, 0, i);
        s[i] = origin[i] - verts(j, 0, i);
    }

    double h[3];
    cross3(dir, e2, h);
    const double a = dot3(e1, h);
    if (fabs(a) < kParallelEps) return kMiss;

    const double f = 1.0 / a;
    const double u = f * dot3(s, h);
    if (u < 0.0) return kMiss;

    double q[3];
    cross3(s, e1, q);
    const double v = f * dot3(dir, q);
    if (v < 0.0 || u + v > 1.0) return kMiss;

    const double t = f * dot3(e2, q);
    return t > min_t ? t : kMiss;
}

// One block per (ray, chunk): the nearest hit in this block's slice of the scan.
__global__ void ray_scan_kernel(const double* __restrict__ origins,
                                const double* __restrict__ directions, VertsView verts,
                                double* __restrict__ chunk_t, int n_tri, int n_chunks,
                                double min_t) {
    using BlockReduce = cub::BlockReduce<double, kBlock>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int r = blockIdx.x;
    const int per = cuda::ceil_div(n_tri, n_chunks);
    const int lo = blockIdx.y * per;
    const int hi = min(lo + per, n_tri);

    const Vec3View<const double> org(origins, kUnsizedRows);
    const Vec3View<const double> dir(directions, kUnsizedRows);
    const double o[3] = {org(r, 0), org(r, 1), org(r, 2)};
    const double d[3] = {dir(r, 0), dir(r, 1), dir(r, 2)};

    double best = kMiss;
    for (int j = lo + threadIdx.x; j < hi; j += kBlock) {
        best = fmin(best, intersect(o, d, verts, j, min_t));
    }

    const double won = BlockReduce(temp).Reduce(best, MinOp());
    if (threadIdx.x == 0) chunk_t[static_cast<size_t>(r) * n_chunks + blockIdx.y] = won;
}

// One thread per ray: the nearest of its chunks' hits.
__global__ void ray_select_kernel(const double* __restrict__ chunk_t, double* __restrict__ out_t,
                                  int n_rays, int n_chunks) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rays) return;

    const double* mine = chunk_t + static_cast<size_t>(r) * n_chunks;
    double best = mine[0];
    for (int i = 1; i < n_chunks; ++i) best = fmin(best, mine[i]);
    out_t[r] = best;
}

}  // namespace

void launch_ray_first_hit(const double* origins, const double* directions, const double* verts,
                          double* out_t, int n_rays, int n_tri, double min_t,
                          cudaStream_t stream) {
    if (n_rays <= 0) return;

    const int n_chunks = scan_chunks(n_rays, n_tri);

    double* chunk_t = nullptr;
    check_cuda(
        cudaMallocAsync(&chunk_t, static_cast<size_t>(n_rays) * n_chunks * sizeof(double), stream),
        "raycast", "mallocAsync(chunk_t)");

    // Rays go on x, the only grid axis not capped at 65535 blocks.
    ray_scan_kernel<<<dim3(n_rays, n_chunks), kBlock, 0, stream>>>(
        origins, directions, VertsView(verts, n_tri), chunk_t, n_tri, n_chunks, min_t);
    ray_select_kernel<<<grid_for(n_rays), kBlock, 0, stream>>>(chunk_t, out_t, n_rays, n_chunks);

    check_cuda(cudaGetLastError(), "raycast", "first hit launch");
    check_cuda(cudaFreeAsync(chunk_t, stream), "raycast", "freeAsync(chunk_t)");
}
