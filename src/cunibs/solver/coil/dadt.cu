#include "coil/coil.hpp"

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

// Node n lands at out[n * out_stride + c].
template <int K>
__global__ void dadt_kernel(const float* __restrict__ s, const float* __restrict__ mp,
                            const float* __restrict__ sn, const float* __restrict__ r,
                            float* __restrict__ out, int out_stride, int n_dip, int n_nodes,
                            float didt, float mu0_4pi) {
    // A dipole is three vector reads from the tile, which keeps shared-memory issue from stalling
    // the FMA pipe.
    __shared__ float4 s_pos[kDipTile];  // (s, |s|^2)
    __shared__ float4 s_mlo[kDipTile];  // m, (m x s).x
    __shared__ float2 s_mhi[kDipTile];  // (m x s).y, (m x s).z

    // This mapping keeps node loads contiguous within each warp and preserves dipole sum order.
    const int T = gridDim.x * blockDim.x;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    int node[K];
    // q = -2 r, so |r - s|^2 = (|r|^2 + |s|^2) + q.s is one add and three FMAs.
    float rx[K], ry[K], rz[K], rn[K], qx[K], qy[K], qz[K];
    float wx[K], wy[K], wz[K];
    float px[K], py[K], pz[K];
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
        wx[k] = wy[k] = wz[k] = px[k] = py[k] = pz[k] = 0.f;
    }

    for (int base = 0; base < n_dip; base += kDipTile) {
        const int tile = min(kDipTile, n_dip - base);
        if (const int i = threadIdx.x; i < tile) {
            const int j = base + i;
            s_pos[i] = make_float4(s[j * 3 + 0], s[j * 3 + 1], s[j * 3 + 2], sn[j]);
            const float* m = mp + static_cast<std::int64_t>(j) * 6;
            s_mlo[i] = make_float4(m[0], m[1], m[2], m[3]);
            s_mhi[i] = make_float2(m[4], m[5]);
        }
        __syncthreads();

#pragma unroll 4
        for (int j = 0; j < tile; ++j) {
            const float4 sp = s_pos[j];
            const float4 ml = s_mlo[j];
            const float2 mh = s_mhi[j];
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const float d2 =
                    fmaf(qx[k], sp.x, fmaf(qy[k], sp.y, fmaf(qz[k], sp.z, rn[k] + sp.w)));
                // The coil-scalp gap keeps d2 positive.
                const float inv = rsqrtf(d2);
                const float w = inv * inv * inv;
                wx[k] += w * ml.x;
                wy[k] += w * ml.y;
                wz[k] += w * ml.z;
                px[k] += w * ml.w;
                py[k] += w * mh.x;
                pz[k] += w * mh.y;
            }
        }
        __syncthreads();
    }

    const float scale = didt * mu0_4pi;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if (node[k] < n_nodes) {
            float* const o = out + static_cast<std::int64_t>(node[k]) * out_stride;
            o[0] = scale * ((wy[k] * rz[k] - wz[k] * ry[k]) - px[k]);
            o[1] = scale * ((wz[k] * rx[k] - wx[k] * rz[k]) - py[k]);
            o[2] = scale * ((wx[k] * ry[k] - wy[k] * rx[k]) - pz[k]);
        }
    }
}

}  // namespace

void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int out_stride, int n_dip, int n_nodes, float didt, float mu0_4pi,
                 cudaStream_t stream) {
    const std::int64_t threads = grid_for(n_nodes, kNodesPerThread);
    if (const unsigned blocks = grid_for(threads, kDadtBlock)) {
        dadt_kernel<kNodesPerThread><<<blocks, kDadtBlock, 0, stream>>>(
            s, mp, sn, r, out, out_stride, n_dip, n_nodes, didt, mu0_4pi);
    }
}
