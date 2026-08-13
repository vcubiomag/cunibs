#pragma once
#include "core/common.hpp"

// --- l1.cu: l1-Jacobi smoother scaling over a CSR operator ----------------------------------
// dinv[i] = 1 / (sign(a_ii) · Σ_j |a_ij|), the diagonal included in the row sum, and 1 where that
// sum is zero. indices need not be sorted; the row is scanned for the diagonal, and a row without
// one keeps the positive sign. dinv is overwritten and must hold n_rows entries.
void launch_l1_dinv(const int* indptr, const int* indices, const float* data, float* dinv,
                    int n_rows, cudaStream_t stream);
