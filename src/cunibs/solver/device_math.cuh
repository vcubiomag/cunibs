#pragma once
#include "common.hpp"

// __device__ helpers shared across kernels. .cu only. Keep this out of dadt.cu: that
// translation unit alone compiles with --use_fast_math, which would give these fp64
// bodies different contraction rules than every other caller gets.

struct Vec3d {
    double x, y, z;
};

__device__ __forceinline__ double dot3(const double* u, const double* v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

// grad_v = Σ_i v[tet_nodes[e,i]] · g[e,i,:] for one P1 tetrahedron. Accumulated in fp64
// because callers subtract a comparable term from it and lose the leading digits.
__device__ __forceinline__ Vec3d tet_grad4(const double* __restrict__ v,
                                           const int* __restrict__ tet_nodes,
                                           const float* __restrict__ g, int e) {
    double gx = 0.0, gy = 0.0, gz = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const double vi = v[tet_nodes[e * 4 + i]];
        const int base = (e * 4 + i) * 3;
        gx += vi * static_cast<double>(g[base + 0]);
        gy += vi * static_cast<double>(g[base + 1]);
        gz += vi * static_cast<double>(g[base + 2]);
    }
    return {gx, gy, gz};
}

// Per-placement pointers for the block stages, passed by value as a kernel argument.
struct ConstPtrPack {
    const float* p[kMaxStageBlock];
};
struct PtrPack {
    float* p[kMaxStageBlock];
};

// Move the K contiguous values of one row of a (n, K) row-major operand as vector accesses
// rather than K scalar ones.
//
// This is what keeps a wide block off the L1 request limit: scalar accesses ask for K separate
// slices per thread, so a warp generates up to 32 distinct wavefronts for data occupying a
// handful of sectors and the kernel stalls on request throughput long before DRAM. K is a power
// of two and every operand starts at an allocation boundary, so a row's base is aligned to the
// width being moved. The values and their order are unaffected.
//
// store_row deliberately takes a plain pointer: the V-cycle's smoother writes a buffer it also
// reads at the row being written, and __restrict__ there would be a promise that call pattern
// breaks.

template <int K>
__device__ __forceinline__ void load_row(const float* __restrict__ p, float (&v)[K]) {
    if constexpr (K == 2) {
        const float2 t = *reinterpret_cast<const float2*>(p);
        v[0] = t.x;
        v[1] = t.y;
    } else if constexpr (K % 4 == 0) {
#pragma unroll
        for (int b = 0; b < K / 4; ++b) {
            const float4 t = *reinterpret_cast<const float4*>(p + b * 4);
            v[b * 4 + 0] = t.x;
            v[b * 4 + 1] = t.y;
            v[b * 4 + 2] = t.z;
            v[b * 4 + 3] = t.w;
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) v[c] = __ldg(p + c);
    }
}

template <int K>
__device__ __forceinline__ void load_row(const double* __restrict__ p, double (&v)[K]) {
    if constexpr (K % 2 == 0) {
#pragma unroll
        for (int b = 0; b < K / 2; ++b) {
            const double2 t = *reinterpret_cast<const double2*>(p + b * 2);
            v[b * 2 + 0] = t.x;
            v[b * 2 + 1] = t.y;
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) v[c] = __ldg(p + c);
    }
}

template <int K>
__device__ __forceinline__ void store_row(float* p, const float (&v)[K]) {
    if constexpr (K == 2) {
        *reinterpret_cast<float2*>(p) = make_float2(v[0], v[1]);
    } else if constexpr (K % 4 == 0) {
#pragma unroll
        for (int b = 0; b < K / 4; ++b) {
            *reinterpret_cast<float4*>(p + b * 4) =
                make_float4(v[b * 4 + 0], v[b * 4 + 1], v[b * 4 + 2], v[b * 4 + 3]);
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) p[c] = v[c];
    }
}

template <int K>
__device__ __forceinline__ void store_row(double* p, const double (&v)[K]) {
    if constexpr (K % 2 == 0) {
#pragma unroll
        for (int b = 0; b < K / 2; ++b) {
            *reinterpret_cast<double2*>(p + b * 2) = make_double2(v[b * 2 + 0], v[b * 2 + 1]);
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) p[c] = v[c];
    }
}
