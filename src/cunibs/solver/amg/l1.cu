#include "amg/l1.hpp"

// One thread per row, the same shape rhs.cu uses for the node gather and for the same reason:
// it fixes the summation order.
//   d[i]    = sign(a_ii) · Σ_j |a_ij|,  the diagonal included
//   dinv[i] = 1 / d[i]
//
// An SpMV of |A| against a vector of ones computes the same sum, but splits a row across a thread
// count cuSPARSE picks at runtime, so the last bits vary between processes. dinv scales every
// entry of the smoothed prolongator and reaches every V-cycle level, so any drift in it reaches
// the whole hierarchy.

namespace {

__global__ void l1_dinv_kernel(const int* __restrict__ indptr, const int* __restrict__ indices,
                               const float* __restrict__ data, float* __restrict__ dinv,
                               int n_rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const int end = indptr[row + 1];
    float acc = 0.f;
    float diag = 0.f;
    for (int p = indptr[row]; p < end; ++p) {
        acc += fabsf(data[p]);
        if (indices[p] == row) diag = data[p];
    }

    // The sign follows the diagonal, which is what keeps the smoother positive definite. A row
    // with no stored diagonal reads as zero and so keeps the positive sum.
    if (diag < 0.f) acc = -acc;
    // Cannot fire for the SPD reduced stiffness, but keeps the reciprocal finite for any input.
    if (acc == 0.f) acc = 1.f;
    dinv[row] = 1.f / acc;
}

}  // namespace

void launch_l1_dinv(const int* indptr, const int* indices, const float* data, float* dinv,
                    int n_rows, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_rows)) {
        l1_dinv_kernel<<<blocks, kBlock, 0, stream>>>(indptr, indices, data, dinv, n_rows);
    }
}
