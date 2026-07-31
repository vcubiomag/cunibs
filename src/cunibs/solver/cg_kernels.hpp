#pragma once
#include "common.hpp"

// Launchers for the mixed-precision CG kernels (block_cg.cu), for k in {1, 2, 4, 8}. All dense
// operands are row-major (n, k). k = 1 is a compiled width like any other, so the single-RHS
// solve runs the same kernels in the same summation order as a wider block.

int bcg_partials_blocks(int n);
void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const int* col_idx,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream);
void launch_bcg_dot(int n, int k, const double* x, const double* y, double* partials,
                    double* out, cudaStream_t stream);
void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream);
// converged is a k-wide 0/1 device mask the caller starts all-zero and mark_converged rewrites
// each iteration; a null pointer means no column is frozen. A frozen column gets
// alpha = beta = 0, which leaves its x and r exactly where they were.
void launch_bcg_alpha(int k, const double* rz, const double* pap, const double* converged,
                      double* alpha, double* neg_alpha, cudaStream_t stream);
void launch_bcg_mark_converged(int k, const double* norm_sq, const double* ref_sq,
                               double tolerance, double* converged, cudaStream_t stream);
void launch_bcg_update_xr_norm(int n, int k, const double* alpha, const double* neg_alpha,
                               const double* p, const double* ap, double* x, double* r,
                               float* rf, double* partials, double* norms,
                               cudaStream_t stream);
void launch_bcg_cast_dot_beta(int n, int k, const float* zf, const double* r,
                              const double* converged, double* partials, double* rz, double* beta,
                              cudaStream_t stream);
void launch_bcg_cast_dot_init(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, cudaStream_t stream);
void launch_bcg_update_p(int n, int k, const double* beta, const float* zf, double* p,
                         cudaStream_t stream);
void launch_bcg_d2f(std::int64_t n_total, const double* in, float* out, cudaStream_t stream);
void launch_bcg_f2d(std::int64_t n_total, const float* in, double* out, cudaStream_t stream);
void launch_bcg_residual(std::int64_t n_total, const double* b, const double* ap, double* r,
                         cudaStream_t stream);
