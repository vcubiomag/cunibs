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
constexpr int kDipTile = 128;
// Nodes each thread carries, not a node count: the register tiling that lets one dipole
// tile load amortize over K nodes.
constexpr int kNodesPerThread = 4;

template <int K>
__global__ void dadt_kernel(const float* __restrict__ s, const float* __restrict__ mp,
                            const float* __restrict__ sn, const float* __restrict__ r,
                            float* __restrict__ out, int n_dip, int n_nodes, float didt,
                            float mu0_4pi) {
    __shared__ float s_s[kDipTile * 3];
    __shared__ float s_mp[kDipTile * 6];
    __shared__ float s_sn[kDipTile];

    // This mapping keeps node loads contiguous within each warp and preserves dipole sum order.
    const int T = gridDim.x * blockDim.x;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    int node[K];
    float rx[K], ry[K], rz[K], rn[K];
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
        wx[k] = wy[k] = wz[k] = px[k] = py[k] = pz[k] = 0.f;
    }

    for (int base = 0; base < n_dip; base += kDipTile) {
        const int tile = min(kDipTile, n_dip - base);
        // Flat copies make adjacent lanes read adjacent values.
        for (int i = threadIdx.x; i < tile * 3; i += kDadtBlock) s_s[i] = s[base * 3 + i];
        for (int i = threadIdx.x; i < tile * 6; i += kDadtBlock) s_mp[i] = mp[base * 6 + i];
        for (int i = threadIdx.x; i < tile; i += kDadtBlock) s_sn[i] = sn[base + i];
        __syncthreads();

#pragma unroll 4
        for (int j = 0; j < tile; ++j) {
            const float sx = s_s[j * 3 + 0];
            const float sy = s_s[j * 3 + 1];
            const float sz = s_s[j * 3 + 2];
            const float snj = s_sn[j];
            const float m0 = s_mp[j * 6 + 0];
            const float m1 = s_mp[j * 6 + 1];
            const float m2 = s_mp[j * 6 + 2];
            const float m3 = s_mp[j * 6 + 3];
            const float m4 = s_mp[j * 6 + 4];
            const float m5 = s_mp[j * 6 + 5];
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const float d2 = rn[k] + snj - 2.f * (rx[k] * sx + ry[k] * sy + rz[k] * sz);
                // The coil-scalp gap keeps d2 positive.
                const float inv = rsqrtf(d2);
                const float w = inv * inv * inv;
                wx[k] += w * m0;
                wy[k] += w * m1;
                wz[k] += w * m2;
                px[k] += w * m3;
                py[k] += w * m4;
                pz[k] += w * m5;
            }
        }
        __syncthreads();
    }

    const float scale = didt * mu0_4pi;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if (node[k] < n_nodes) {
            out[node[k] * 3 + 0] = scale * ((wy[k] * rz[k] - wz[k] * ry[k]) - px[k]);
            out[node[k] * 3 + 1] = scale * ((wz[k] * rx[k] - wx[k] * rz[k]) - py[k]);
            out[node[k] * 3 + 2] = scale * ((wx[k] * ry[k] - wy[k] * rx[k]) - pz[k]);
        }
    }
}

}  // namespace

void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int n_dip, int n_nodes, float didt, float mu0_4pi, cudaStream_t stream) {
    const std::int64_t threads = grid_for(n_nodes, kNodesPerThread);
    if (const unsigned blocks = grid_for(threads, kDadtBlock)) {
        dadt_kernel<kNodesPerThread><<<blocks, kDadtBlock, 0, stream>>>(s, mp, sn, r, out, n_dip,
                                                                        n_nodes, didt, mu0_4pi);
    }
}
