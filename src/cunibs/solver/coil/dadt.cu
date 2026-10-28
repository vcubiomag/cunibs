#include "coil/coil.hpp"

#include <algorithm>
#include <cstdint>

// Evaluate dA/dt without storing the node-by-dipole weight matrix.
// Each thread accumulates:
//   w_j = |r - s_j|^-3 = (|r|^2 + |s_j|^2 - 2 r·s_j)^-3/2
//   W = Σ_j w_j m_j
//   P = Σ_j w_j (m_j × s_j)
//   dA/dt = didt · mu0_4pi · (W × r - P)
//
// Offsets are spelled flat here: this translation unit alone compiles with --use_fast_math, so it
// must not include device_math.cuh, whose fp64 bodies would pick up contraction rules no other
// caller gets.

namespace {

// Local block size: this kernel is register-bound and wants a narrower block than the
// shared kBlock. Named apart from it so nothing here silently shadows the shared constant.
constexpr int kDadtBlock = 128;
// One dipole per thread of the block, so a tile loads in one pass.
constexpr int kDipTile = kDadtBlock;
// Nodes each thread carries, the register tiling that lets one dipole tile load amortise over
// K nodes.
constexpr int kNodesPerThread = 8;

// Tolerance on |m_j x u| / |m_j| below which a coil counts as having one moment direction u.
constexpr double kPlanarTol = 1e-9;

// planar stays nonzero only if every moment is parallel to the first, which flat coil models are
// by construction: their windings lie in one plane and every dipole points along its normal. The
// test only ever clears the flag, so the order threads reach it in cannot change the outcome.
__global__ void dadt_planar_kernel(const double* __restrict__ mom, int n_dip,
                                   int* __restrict__ planar) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_dip) return;
    const double ax = mom[0], ay = mom[1], az = mom[2];
    const double a2 = ax * ax + ay * ay + az * az;
    const double mx = mom[3 * j + 0], my = mom[3 * j + 1], mz = mom[3 * j + 2];
    const double cx = my * az - mz * ay, cy = mz * ax - mx * az, cz = mx * ay - my * ax;
    const double c2 = cx * cx + cy * cy + cz * cz;
    const double m2 = mx * mx + my * my + mz * mz;
    if (a2 == 0.0 || c2 > kPlanarTol * kPlanarTol * m2 * a2) atomicAnd(planar, 0);
}

// One thread per (column, dipole): the coil-frame dipoles placed by column c's transform, centred
// on the coil origin and narrowed to fp32 once. Centring is what keeps the expanded |r - s|^2
// accurate, and with the origin at the transform's translation the centred position is the
// rotation alone: R s_coil, rather than (s_coil R^T + t) - t with its cancellation.
//
// General coils stage (s, |s|^2), (m, (m x s).x) and ((m x s).y, (m x s).z). A coil with one moment
// direction u stages (s, |s|^2) and (mu, mu s) with mu = m . u, which is all the planar loop reads.
// Every fp64 operation is an explicit round-to-nearest intrinsic: this file compiles with fast
// math, and contraction left to nvcc would move the staged values by an ulp between architectures.
//
// A coil with no dipoles still gets its origins, one thread per column, and stages nothing.
__global__ void place_dipoles_kernel(const double* __restrict__ pos_m,
                                     const double* __restrict__ mom,
                                     const double* __restrict__ tf, int n_dip, int k,
                                     const int* __restrict__ planar, float4* __restrict__ s4,
                                     float4* __restrict__ m4, float2* __restrict__ m2,
                                     double* __restrict__ origin, float* __restrict__ axis) {
    const int rows = n_dip > 0 ? n_dip : 1;
    const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::int64_t>(rows) * k) return;
    const int c = static_cast<int>(t / rows);
    const int j = static_cast<int>(t % rows);
    const double* R = tf + 16 * c;
    if (j == 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) origin[3 * c + i] = __dmul_rn(R[4 * i + 3], 1e-3);
    }
    if (n_dip == 0) return;
    const double* p = pos_m + 3 * j;
    const double* m = mom + 3 * j;
    double sp[3], mr[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        sp[i] = __dadd_rn(__dadd_rn(__dmul_rn(R[4 * i], p[0]), __dmul_rn(R[4 * i + 1], p[1])),
                          __dmul_rn(R[4 * i + 2], p[2]));
        mr[i] = __dadd_rn(__dadd_rn(__dmul_rn(R[4 * i], m[0]), __dmul_rn(R[4 * i + 1], m[1])),
                          __dmul_rn(R[4 * i + 2], m[2]));
    }
    const float sx = static_cast<float>(sp[0]), sy = static_cast<float>(sp[1]),
                sz = static_cast<float>(sp[2]);
    s4[t] = make_float4(sx, sy, sz,
                        __fadd_rn(__fadd_rn(__fmul_rn(sx, sx), __fmul_rn(sy, sy)),
                                  __fmul_rn(sz, sz)));
    const auto dot = [](const double* a, const double* b) {
        return __dadd_rn(__dadd_rn(__dmul_rn(a[0], b[0]), __dmul_rn(a[1], b[1])),
                         __dmul_rn(a[2], b[2]));
    };
    const auto cross = [&](int a, int b) {
        return static_cast<float>(__dsub_rn(__dmul_rn(mr[a], sp[b]), __dmul_rn(mr[b], sp[a])));
    };
    if (*planar) {
        const double un = __dsqrt_rn(dot(mom, mom));
        const double mu = __ddiv_rn(dot(m, mom), un);
        m4[t] = make_float4(static_cast<float>(mu), static_cast<float>(__dmul_rn(mu, sp[0])),
                            static_cast<float>(__dmul_rn(mu, sp[1])),
                            static_cast<float>(__dmul_rn(mu, sp[2])));
        if (j == 0) {
#pragma unroll
            for (int i = 0; i < 3; ++i) axis[3 * c + i] = static_cast<float>(__ddiv_rn(mr[i], un));
        }
    } else {
        m4[t] = make_float4(static_cast<float>(mr[0]), static_cast<float>(mr[1]),
                            static_cast<float>(mr[2]), cross(1, 2));
        m2[t] = make_float2(cross(2, 0), cross(0, 1));
    }
}

template <int K, bool kFlat>
__device__ __forceinline__ void dadt_column(const double* __restrict__ nodes_mm,
                                            const float4* __restrict__ sc,
                                            const float4* __restrict__ mc,
                                            const float2* __restrict__ hc,
                                            const double* __restrict__ o,
                                            const float* __restrict__ u, float* __restrict__ out,
                                            int out_stride, int n_dip, int n_nodes, float scale) {
    __shared__ float4 s_pos[kDipTile];
    __shared__ float4 s_mlo[kDipTile];
    __shared__ float2 s_mhi[kDipTile];
    const int T = gridDim.x * blockDim.x;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    int node[K];
    float rx[K], ry[K], rz[K], rn[K], qx[K], qy[K], qz[K];
    // General: W = a0..a2, P = a3..a5. Flat: A = a0, V = a1..a3.
    float a0[K], a1[K], a2[K], a3[K], a4[K], a5[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        node[k] = t + k * T;
        rx[k] = ry[k] = rz[k] = rn[k] = 0.f;
        if (node[k] < n_nodes) {
            const double* x = nodes_mm + 3 * static_cast<std::int64_t>(node[k]);
            // In metres about the coil origin, rounded once from fp64.
            rx[k] = static_cast<float>(__dsub_rn(__dmul_rn(x[0], 1e-3), o[0]));
            ry[k] = static_cast<float>(__dsub_rn(__dmul_rn(x[1], 1e-3), o[1]));
            rz[k] = static_cast<float>(__dsub_rn(__dmul_rn(x[2], 1e-3), o[2]));
            rn[k] = rx[k] * rx[k] + ry[k] * ry[k] + rz[k] * rz[k];
        }
        qx[k] = -2.f * rx[k];
        qy[k] = -2.f * ry[k];
        qz[k] = -2.f * rz[k];
        a0[k] = a1[k] = a2[k] = a3[k] = a4[k] = a5[k] = 0.f;
    }

    for (int base = 0; base < n_dip; base += kDipTile) {
        const int tile = min(kDipTile, n_dip - base);
        if (const int i = threadIdx.x; i < tile) {
            s_pos[i] = sc[base + i];
            s_mlo[i] = mc[base + i];
            if constexpr (!kFlat) s_mhi[i] = hc[base + i];
        }
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < tile; ++j) {
            const float4 sp = s_pos[j];
            const float4 ml = s_mlo[j];
            float2 mh;
            if constexpr (!kFlat) mh = s_mhi[j];
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const float d2 =
                    fmaf(qx[k], sp.x, fmaf(qy[k], sp.y, fmaf(qz[k], sp.z, rn[k] + sp.w)));
                // The coil-scalp gap keeps d2 positive.
                const float inv = rsqrtf(d2);
                const float w = inv * inv * inv;
                a0[k] += w * ml.x;
                a1[k] += w * ml.y;
                a2[k] += w * ml.z;
                a3[k] += w * ml.w;
                if constexpr (!kFlat) {
                    a4[k] += w * mh.x;
                    a5[k] += w * mh.y;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < K; ++k) {
        if (node[k] >= n_nodes) continue;
        float* const p = out + static_cast<std::int64_t>(node[k]) * out_stride;
        if constexpr (kFlat) {
            const float cx = a0[k] * rx[k] - a1[k];
            const float cy = a0[k] * ry[k] - a2[k];
            const float cz = a0[k] * rz[k] - a3[k];
            p[0] = scale * (u[1] * cz - u[2] * cy);
            p[1] = scale * (u[2] * cx - u[0] * cz);
            p[2] = scale * (u[0] * cy - u[1] * cx);
        } else {
            p[0] = scale * ((a1[k] * rz[k] - a2[k] * ry[k]) - a3[k]);
            p[1] = scale * ((a2[k] * rx[k] - a0[k] * rz[k]) - a4[k]);
            p[2] = scale * ((a0[k] * ry[k] - a1[k] * rx[k]) - a5[k]);
        }
    }
}

// dA/dt at every node for k placements of one coil, blockIdx.y the placement. Node n of column c
// lands at out[n * out_stride + 3 c].
//
// The pair loop is issue-bound: 13 instructions per pair (four for |r - s|^2, the rsqrt, two
// multiplies for |r - s|^-3, six accumulations), all but the rsqrt on the FP32 pipe. A coil with
// one moment direction u needs only A = sum w mu and V = sum w mu s, since W = u A and P = u x V
// give dA/dt = u x (A r - V): four accumulations instead of six. The flag is uniform, and each
// branch is a whole column's loop, so neither path is scheduled around the other's registers.
template <int K>
__global__ void dadt_block_kernel(const double* __restrict__ nodes_mm,
                                  const float4* __restrict__ s4, const float4* __restrict__ m4,
                                  const float2* __restrict__ m2,
                                  const double* __restrict__ origin,
                                  const float* __restrict__ axis, const int* __restrict__ planar,
                                  float* __restrict__ out, int out_stride, int n_dip,
                                  int n_nodes, float scale) {
    const int c = blockIdx.y;
    const std::int64_t off = static_cast<std::int64_t>(c) * n_dip;
    if (*planar != 0) {
        dadt_column<K, true>(nodes_mm, s4 + off, m4 + off, m2 + off, origin + 3 * c, axis + 3 * c,
                             out + 3 * c, out_stride, n_dip, n_nodes, scale);
    } else {
        dadt_column<K, false>(nodes_mm, s4 + off, m4 + off, m2 + off, origin + 3 * c,
                              axis + 3 * c, out + 3 * c, out_stride, n_dip, n_nodes, scale);
    }
}

}  // namespace

void launch_dadt_block(const double* pos_m, const double* mom, const double* transforms,
                       int n_dip, int k, const double* nodes_mm, int n_nodes, float* out,
                       int out_stride, float didt, float mu0_4pi, cudaStream_t stream) {
    if (k <= 0 || n_nodes <= 0) return;
    const std::size_t kn = static_cast<std::size_t>(k) * std::max(n_dip, 1);
    const ScratchBuffer<int> planar = scratch_alloc<int>(1, stream, "dadt", "planar");
    const ScratchBuffer<float4> s4 = scratch_alloc<float4>(kn, stream, "dadt", "positions");
    const ScratchBuffer<float4> m4 = scratch_alloc<float4>(kn, stream, "dadt", "moments");
    const ScratchBuffer<float2> m2 = scratch_alloc<float2>(kn, stream, "dadt", "moments");
    const ScratchBuffer<double> origin = scratch_alloc<double>(3 * k, stream, "dadt", "origin");
    const ScratchBuffer<float> axis = scratch_alloc<float>(3 * k, stream, "dadt", "axis");
    check_cuda(cudaMemsetAsync(planar.get(), 1, sizeof(int), stream), "dadt", "planar");
    if (n_dip > 0) {
        dadt_planar_kernel<<<grid_for(n_dip), kBlock, 0, stream>>>(mom, n_dip, planar.get());
    }
    const std::int64_t placed = static_cast<std::int64_t>(std::max(n_dip, 1)) * k;
    place_dipoles_kernel<<<grid_for(placed), kBlock, 0, stream>>>(
        pos_m, mom, transforms, n_dip, k, planar.get(), s4.get(), m4.get(), m2.get(),
        origin.get(), axis.get());
    const std::int64_t threads = grid_for(n_nodes, kNodesPerThread);
    const dim3 grid(grid_for(threads, kDadtBlock), k);
    dadt_block_kernel<kNodesPerThread><<<grid, kDadtBlock, 0, stream>>>(
        nodes_mm, s4.get(), m4.get(), m2.get(), origin.get(), axis.get(), planar.get(), out,
        out_stride, n_dip, n_nodes, didt * mu0_4pi);
    check_cuda(cudaGetLastError(), "dadt", "launch");
}
