#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cuda/std/array>
#include <cuda/std/cmath>

// Smoothed outward normals for the skin surface.
//
// Each triangle's normal is the area-weighted face normal spread to its three nodes, gathered back
// per face, and smoothed once more; the result is the normalised mean of the three node normals.
// It is what a coil placement projects against, so it is built once per subject with the rest of
// the context.
//
// The spread is a reduction over the triangles meeting at a node, and its order is what makes the
// answer reproducible. Every one of them walks a node-to-corner CSR with one thread per node, in
// the CSR's own order. An atomicAdd over faces would be shorter and would not be reproducible run
// to run, which the field's contract does not allow.

namespace {

void check_cuda(cudaError_t err, const char* what) { ::check_cuda(err, "skin", what); }

using Vec3 = cuda::std::array<double, 3>;

// Unit vector, leaving a zero vector alone. Only skin nodes are reached through the CSR, but the
// accumulators are indexed by mesh node, so the rest stay zero and must not become NaN.
__device__ __forceinline__ void normalise(Vec3& v) {
    const double n = cuda::std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n > 0.0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) v[i] /= n;
    }
}

__device__ __forceinline__ Vec3 load_vec3(Vec3View<const double> rows, int r) {
    return {rows(r, 0), rows(r, 1), rows(r, 2)};
}

__device__ __forceinline__ void store_vec3(Vec3View<double> rows, int r, const Vec3& v) {
#pragma unroll
    for (int i = 0; i < 3; ++i) rows(r, i) = v[i];
}

// One thread per triangle: the unnormalised face normal, whose length carries twice the area and
// so weights the smoothing that follows.
__global__ void face_normals_kernel(const double* __restrict__ nodes_mm,
                                    Tri3View<const int> tris, double* __restrict__ out,
                                    int n_tri) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;
    const Vec3View<const double> nodes(nodes_mm, kUnsizedRows);
    const Vec3 a = load_vec3(nodes, tris(t, 0));
    const Vec3 b = load_vec3(nodes, tris(t, 1));
    const Vec3 c = load_vec3(nodes, tris(t, 2));

    Vec3 u, v;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        u[i] = b[i] - a[i];
        v[i] = c[i] - a[i];
    }
    const Vec3 cross = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0]};
    store_vec3(Vec3View<double>(out, kUnsizedRows), t, cross);
}

// One thread per node, summing the per-face vector of every triangle corner in the node's segment.
// idx holds corner ids c = 3t + i, so c / 3 is the triangle. Walking the segment in order is what
// pins the summation order.
__global__ void spread_to_nodes_kernel(const double* __restrict__ face, const int* __restrict__ ptr,
                                       const int* __restrict__ idx, double* __restrict__ out,
                                       int n_nodes) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_nodes) return;
    const Vec3View<const double> src(face, kUnsizedRows);

    Vec3 acc{};
    const int end = ptr[n + 1];
    for (int p = ptr[n]; p < end; ++p) {
        const int t = idx[p] / 3;
#pragma unroll
        for (int i = 0; i < 3; ++i) acc[i] += src(t, i);
    }
    store_vec3(Vec3View<double>(out, kUnsizedRows), n, acc);
}

// One thread per triangle, gathering its three node vectors back. The sum is normalised rather
// than averaged: on the final pass the mean and the sum differ only by a factor of three, which
// normalising removes.
__global__ void gather_to_faces_kernel(const double* __restrict__ node_vec,
                                       Tri3View<const int> tris, double* __restrict__ out,
                                       int n_tri) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;
    const Vec3View<const double> src(node_vec, kUnsizedRows);

    Vec3 acc;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        acc[i] = src(tris(t, 0), i) + src(tris(t, 1), i) + src(tris(t, 2), i);
    }
    normalise(acc);
    store_vec3(Vec3View<double>(out, kUnsizedRows), t, acc);
}

__global__ void normalise_rows_kernel(double* __restrict__ v, int n) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const Vec3View<double> rows(v, kUnsizedRows);
    Vec3 a = load_vec3(rows, r);
    normalise(a);
    store_vec3(rows, r, a);
}

// Smoothing rounds. Two is what the surface a coil is placed against was tuned on; it is not a
// convergence parameter.
constexpr int kSmoothingRounds = 2;

}  // namespace

void launch_skin_normals(const double* nodes_mm, const int* tris_ptr, const int* node_ptr,
                         const int* node_idx, double* out_tri_normals, int n_tri, int n_nodes,
                         cudaStream_t stream) {
    if (n_tri <= 0) return;
    DeviceBuffer<double> face =
        device_alloc<double>(static_cast<std::size_t>(n_tri) * 3, "skin", "face normals");
    DeviceBuffer<double> node =
        device_alloc<double>(static_cast<std::size_t>(n_nodes) * 3, "skin", "node normals");

    const Tri3View<const int> tris(tris_ptr, n_tri);
    const unsigned tri_blocks = grid_for(n_tri);
    const unsigned node_blocks = grid_for(n_nodes);
    face_normals_kernel<<<tri_blocks, kBlock, 0, stream>>>(nodes_mm, tris, face.get(), n_tri);
    spread_to_nodes_kernel<<<node_blocks, kBlock, 0, stream>>>(face.get(), node_ptr, node_idx,
                                                               node.get(), n_nodes);
    // The gather that closes a round feeds the next one's spread, so the last round does not need
    // one: what survives to the end is the node vector, not the face vector.
    for (int round = 1; round < kSmoothingRounds; ++round) {
        gather_to_faces_kernel<<<tri_blocks, kBlock, 0, stream>>>(node.get(), tris, face.get(),
                                                                  n_tri);
        spread_to_nodes_kernel<<<node_blocks, kBlock, 0, stream>>>(face.get(), node_ptr, node_idx,
                                                                   node.get(), n_nodes);
    }
    normalise_rows_kernel<<<node_blocks, kBlock, 0, stream>>>(node.get(), n_nodes);
    gather_to_faces_kernel<<<tri_blocks, kBlock, 0, stream>>>(node.get(), tris, out_tri_normals,
                                                              n_tri);
    check_cuda(cudaGetLastError(), "skin normals launch");
    // The scratch dies with this frame, so the stream must be done with it first.
    check_cuda(cudaStreamSynchronize(stream), "sync(skin normals)");
}
