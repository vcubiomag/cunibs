#include "fem/fem.hpp"

#include <cstdint>

// P1 basis-function gradients and element volumes, one thread per tetrahedron.
//
// With T the edge matrix whose rows are n_i - n_0 for i = 1..3, and
// A = [[-1,1,0,0], [-1,0,1,0], [-1,0,0,1]], the operator is G = (T^-1 A)^T. Columns 1..3 of A
// are the identity, so rows 1..3 of G are the columns of T^-1 and row 0 is minus their sum.
// For a 3x3 the columns of T^-1 are the cross products of the edge pairs over the determinant,
// so no factorisation and no pivoting are involved.
//
// Nodes arrive in millimetres and the result is in the metre units the FEM works in: the cross
// products scale as mm^2 and the determinant as mm^3, so G picks up a factor of 1e3 and the
// volume one of 1e-9.

namespace {

constexpr double kGradScale = 1e3;
constexpr double kVolumeScale = 1e-9 / 6.0;

__device__ __forceinline__ void cross3(const double* a, const double* b, double* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__global__ void p1_gradients_kernel(const double* __restrict__ nodes_mm,
                                    const int* __restrict__ tet_nodes, double* __restrict__ g,
                                    double* __restrict__ vols, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    const double* n0 = nodes_mm + static_cast<std::int64_t>(tet_nodes[e * 4 + 0]) * 3;
    double t[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const double* ni = nodes_mm + static_cast<std::int64_t>(tet_nodes[e * 4 + i + 1]) * 3;
#pragma unroll
        for (int c = 0; c < 3; ++c) t[i][c] = ni[c] - n0[c];
    }

    double adj[3][3];
    cross3(t[1], t[2], adj[0]);
    cross3(t[2], t[0], adj[1]);
    cross3(t[0], t[1], adj[2]);
    const double det = t[0][0] * adj[0][0] + t[0][1] * adj[0][1] + t[0][2] * adj[0][2];
    const double inv = kGradScale / det;

    double* row = g + static_cast<std::int64_t>(e) * 12;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const double g1 = adj[0][c] * inv;
        const double g2 = adj[1][c] * inv;
        const double g3 = adj[2][c] * inv;
        row[0 * 3 + c] = -(g1 + g2 + g3);
        row[1 * 3 + c] = g1;
        row[2 * 3 + c] = g2;
        row[3 * 3 + c] = g3;
    }
    vols[e] = fabs(det) * kVolumeScale;
}

}  // namespace

void launch_p1_gradients(const double* nodes_mm, const int* tet_nodes, double* g, double* vols,
                         int n_tet, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_tet)) {
        p1_gradients_kernel<<<blocks, kBlock, 0, stream>>>(nodes_mm, tet_nodes, g, vols, n_tet);
    }
}
