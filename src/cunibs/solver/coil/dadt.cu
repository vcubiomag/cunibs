#include "coil/coil.hpp"

#include <cstdint>
#include <stdexcept>

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
// The same for three moment sets, which carry three times the accumulators per node.
constexpr int kNodesPerThreadMulti = 4;

// D moment sets share every source and target position, and so every weight w_j: the coil's single
// set for dA/dt, or the reciprocity field's one set per adjoint direction. The weight is the
// rsqrt and two multiplies of an interaction, so sharing it makes the three-direction Q-field a
// single pass. Set d of node n lands at out[n * out_stride + 3 d + c], and each set's arithmetic is
// exactly the D = 1 kernel's.
template <int D, int K>
__global__ void dadt_kernel(const float* __restrict__ s, const float* __restrict__ mp,
                            const float* __restrict__ sn, const float* __restrict__ r,
                            float* __restrict__ out, int out_stride, int n_dip, int n_nodes,
                            float didt, float mu0_4pi) {
    // A dipole is three vector reads from the tile, which keeps shared-memory issue from stalling
    // the FMA pipe.
    __shared__ float4 s_pos[kDipTile];     // (s, |s|^2)
    __shared__ float4 s_mlo[D][kDipTile];  // m, (m x s).x
    __shared__ float2 s_mhi[D][kDipTile];  // (m x s).y, (m x s).z

    // This mapping keeps node loads contiguous within each warp and preserves dipole sum order.
    const int T = gridDim.x * blockDim.x;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    int node[K];
    // q = -2 r, so |r - s|^2 = (|r|^2 + |s|^2) + q.s is one add and three FMAs.
    float rx[K], ry[K], rz[K], rn[K], qx[K], qy[K], qz[K];
    float wx[D][K], wy[D][K], wz[D][K];
    float px[D][K], py[D][K], pz[D][K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        node[k] = t + k * T;
        rx[k] = ry[k] = rz[k] = rn[k] = 0.f;
        if (node[k] < n_nodes) {
            rx[k] = r[node[k] * 3 + 0];
            ry[k] = r[node[k] * 3 + 1];
            rz[k] = r[node[k] * 3 + 2];
            rn[k] = rx[k] * rx[k] + ry[k] * ry[k] + rz[k] * rz[k];
        }
        qx[k] = -2.f * rx[k];
        qy[k] = -2.f * ry[k];
        qz[k] = -2.f * rz[k];
#pragma unroll
        for (int d = 0; d < D; ++d) {
            wx[d][k] = wy[d][k] = wz[d][k] = px[d][k] = py[d][k] = pz[d][k] = 0.f;
        }
    }

    for (int base = 0; base < n_dip; base += kDipTile) {
        const int tile = min(kDipTile, n_dip - base);
        if (const int i = threadIdx.x; i < tile) {
            const int j = base + i;
            s_pos[i] = make_float4(s[j * 3 + 0], s[j * 3 + 1], s[j * 3 + 2], sn[j]);
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const float* m = mp + (static_cast<std::int64_t>(d) * n_dip + j) * 6;
                s_mlo[d][i] = make_float4(m[0], m[1], m[2], m[3]);
                s_mhi[d][i] = make_float2(m[4], m[5]);
            }
        }
        __syncthreads();

#pragma unroll 4
        for (int j = 0; j < tile; ++j) {
            const float4 sp = s_pos[j];
            float4 ml[D];
            float2 mh[D];
#pragma unroll
            for (int d = 0; d < D; ++d) {
                ml[d] = s_mlo[d][j];
                mh[d] = s_mhi[d][j];
            }
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const float d2 =
                    fmaf(qx[k], sp.x, fmaf(qy[k], sp.y, fmaf(qz[k], sp.z, rn[k] + sp.w)));
                // The coil-scalp gap keeps d2 positive.
                const float inv = rsqrtf(d2);
                const float w = inv * inv * inv;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    wx[d][k] += w * ml[d].x;
                    wy[d][k] += w * ml[d].y;
                    wz[d][k] += w * ml[d].z;
                    px[d][k] += w * ml[d].w;
                    py[d][k] += w * mh[d].x;
                    pz[d][k] += w * mh[d].y;
                }
            }
        }
        __syncthreads();
    }

    const float scale = didt * mu0_4pi;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if (node[k] < n_nodes) {
            float* const o = out + static_cast<std::int64_t>(node[k]) * out_stride;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                o[3 * d + 0] = scale * ((wy[d][k] * rz[k] - wz[d][k] * ry[k]) - px[d][k]);
                o[3 * d + 1] = scale * ((wz[d][k] * rx[k] - wx[d][k] * rz[k]) - py[d][k]);
                o[3 * d + 2] = scale * ((wx[d][k] * ry[k] - wy[d][k] * rx[k]) - pz[d][k]);
            }
        }
    }
}

}  // namespace

void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int out_stride, int n_sets, int n_dip, int n_nodes, float didt, float mu0_4pi,
                 cudaStream_t stream) {
    const auto launch = [&](auto kernel, int nodes_per_thread) {
        const std::int64_t threads = grid_for(n_nodes, nodes_per_thread);
        if (const unsigned blocks = grid_for(threads, kDadtBlock)) {
            kernel<<<blocks, kDadtBlock, 0, stream>>>(s, mp, sn, r, out, out_stride, n_dip,
                                                      n_nodes, didt, mu0_4pi);
        }
    };
    if (n_sets == 1) {
        launch(dadt_kernel<1, kNodesPerThread>, kNodesPerThread);
    } else if (n_sets == 3) {
        launch(dadt_kernel<3, kNodesPerThreadMulti>, kNodesPerThreadMulti);
    } else {
        throw std::invalid_argument("dadt: moment sets must be 1 or 3");
    }
}
