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
