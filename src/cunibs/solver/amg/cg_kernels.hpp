#pragma once
#include <cstdint>
#include "core/common.hpp"

// Launchers for the mixed-precision CG kernels (block_cg.cu), for k in {1, 2, 4, 8}. All dense
// operands are row-major (n, k). k = 1 is a compiled width like any other, so the single-RHS
// solve runs the same kernels in the same summation order as a wider block.

int bcg_partials_blocks(int n);
// row_ptr/cols/vals are a slot CSR (amg/slot_csr.hpp), vals laid out in its slots.
void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const std::int16_t* cols,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream);
// An operator affine in per-column coefficients, A(sigma) = sum_t sigma_t A_t, stored as per-row
// segments, one per part touching the row: segments [seg_ptr[i], seg_ptr[i+1]) of row i, segment
// g holding slots [seg_start[g], seg_start[g+1]) of part seg_term[g].
struct AffineOperator {
    int n_terms = 0;
    const int* order = nullptr;  // the order rows are taken in; any permutation gives the same y
    const int* seg_ptr = nullptr;
    const int* seg_start = nullptr;
    const std::uint8_t* seg_term = nullptr;
    const std::int16_t* cols = nullptr;
    const double* vals = nullptr;
};
// sigma is [column][term] for kMaxStageBlock columns.
void launch_affine_spmv_block(int n, int k, const AffineOperator& op, const double* sigma,
                              const double* x, double* y, cudaStream_t stream);
void launch_bcg_dot(int n, int k, const double* x, const double* y, double* partials,
                    double* out, cudaStream_t stream);
void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream);
void launch_bcg_sum(int n, int k, const double* x, double* partials, double* out,
                    cudaStream_t stream);
// Deflation of the constant vector (see block_cg.cu). The state a solve reads: the rows of the
// support of A 1, Aw on them laid out [row][column] for kMaxStageBlock columns, and 1/E per column
// (0 switches that column's deflation off).
struct Deflation {
    int n_rows = 0;
    const int* rows = nullptr;
    const double* aw = nullptr;
    const double* inv_e = nullptr;
};
// Setup only: every row's sum and l1 norm, from which the caller picks the support.
void launch_bcg_row_sums(int n, const int* row_ptr, const double* vals, double* sum, double* l1,
                         cudaStream_t stream);
// Aw on the support from the operator's values, the same in every column, and 1/E.
void launch_bcg_defl_from_values(int n_s, const int* rows, const int* row_ptr,
                                 const double* vals, double* aw, double* inv_e,
                                 cudaStream_t stream);
// The same for an affine operator: column c's Aw is sum_t sigma[c, t] aw_t[t], sigma laid out
// [column][term] for kMaxStageBlock columns.
void launch_bcg_defl_affine(int n_s, int n_terms, const double* aw_t, const double* sigma,
                            double* aw, double* inv_e, cudaStream_t stream);
// x <- x + Q r and r <- r - A Q r for the starting pair; also writes rf and the new sum(r).
void launch_bcg_deflate_start(int n, int k, Deflation defl, double* x, double* r, float* rf,
                              double* partials, double* rsum, cudaStream_t stream);
// converged is a k-wide 0/1 device mask the caller starts all-zero and mark_converged rewrites
// each iteration; a null pointer means no column is frozen. A frozen column gets
// alpha = beta = 0, which leaves its x and r exactly where they were.
void launch_bcg_alpha(int k, const double* rz, const double* pap, const double* converged,
                      double* alpha, double* neg_alpha, cudaStream_t stream);
void launch_bcg_mark_converged(int k, const double* norm_sq, const double* ref_sq,
                               const double* tolerance, double* converged, cudaStream_t stream);
// r -= alpha Ap and rf = (float)r. norms_sums is 2k wide: |r|^2 per column, then sum(r).
void launch_bcg_update_r_norm(int n, int k, const double* neg_alpha, const double* ap, double* r,
                              float* rf, double* partials, double* norms_sums,
                              cudaStream_t stream);
// The preconditioned residual is z = zf + shift per column; these write that shift and the
// deflated r . z, and the init form also sets p0 = z0.
void launch_bcg_cast_dot_beta(int n, int k, const float* zf, const double* r, Deflation defl,
                              const double* converged, const double* rsum, double* partials,
                              double* rz, double* beta, double* shift, cudaStream_t stream);
void launch_bcg_cast_dot_init(int n, int k, const float* zf, const double* r, Deflation defl,
                              const double* rsum, double* partials, double* rz, double* shift,
                              double* p, cudaStream_t stream);
// x += alpha p, then p = beta p + z, in one pass over p.
void launch_bcg_update_p(int n, int k, const double* alpha, const double* beta,
                         const double* shift, const float* zf, double* p, double* x,
                         cudaStream_t stream);
void launch_bcg_residual(std::int64_t n_total, const double* b, const double* ap, double* r,
                         cudaStream_t stream);
