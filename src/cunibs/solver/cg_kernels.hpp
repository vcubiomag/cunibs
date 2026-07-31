#pragma once
#include "common.hpp"

// Launchers for the mixed-precision CG kernels (block_cg.cu). All dense operands are
// row-major (n, k); the single-RHS set below is the K = 1 instantiation of the same kernels,
// so the two groups differ in signature only.

// --- single RHS ----------------------------------------------------------------------------
void launch_double_to_float(const double* in, float* out, int n, cudaStream_t stream);
void launch_float_to_double(const float* in, double* out, int n, cudaStream_t stream);
void launch_cg_alpha(const double* rz, const double* pap, double* alpha, double* neg_alpha,
                     cudaStream_t stream);
void launch_cg_update_p(const double* beta, const float* zf, double* p, int n,
                        cudaStream_t stream);
void launch_csrmv_f64(int n, const int* row_ptr, const int* col_idx, const double* vals,
                      const double* x, double* y, cudaStream_t stream);
int cg_partials_size(int n);
void launch_cg_update_xr_norm(const double* alpha, const double* neg_alpha, const double* p,
                              const double* ap, double* x, double* r, float* rf,
                              double* partials, double* norm_sq, int n, cudaStream_t stream);
void launch_cg_cast_dot_beta(const float* zf, const double* r, double* partials,
                             double* rz, double* beta, int n, cudaStream_t stream);

// --- k in {2, 4, 8} --------------------------------------------------------------------------
int bcg_partials_blocks(int n);
void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const int* col_idx,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream);
void launch_bcg_dot(int n, int k, const double* x, const double* y, double* partials,
                    double* out, cudaStream_t stream);
void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream);
void launch_bcg_alpha(int k, const double* rz, const double* pap, double* alpha,
                      double* neg_alpha, cudaStream_t stream);
void launch_bcg_update_xr_norm(int n, int k, const double* alpha, const double* neg_alpha,
                               const double* p, const double* ap, double* x, double* r,
                               float* rf, double* partials, double* norms,
                               cudaStream_t stream);
void launch_bcg_cast_dot_beta(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, double* beta,
                              cudaStream_t stream);
void launch_bcg_cast_dot_init(int n, int k, const float* zf, const double* r,
                              double* partials, double* rz, cudaStream_t stream);
void launch_bcg_update_p(int n, int k, const double* beta, const float* zf, double* p,
                         cudaStream_t stream);
void launch_bcg_d2f(std::int64_t n_total, const double* in, float* out, cudaStream_t stream);
void launch_bcg_f2d(std::int64_t n_total, const float* in, double* out, cudaStream_t stream);
void launch_bcg_residual(std::int64_t n_total, const double* b, const double* ap, double* r,
                         cudaStream_t stream);
