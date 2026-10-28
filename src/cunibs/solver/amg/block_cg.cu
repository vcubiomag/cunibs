// Mixed-precision CG kernels, templated on the number of right-hand sides K, following the
// policy vcycle.cu sets: every step is written once and the single-RHS path is the K = 1
// instantiation. K chains run in lockstep sharing each matrix read. All dense operands are
// row-major (n, k) so the k values of one mesh node are contiguous; col_idx/vals are read
// once per nnz and reused across the k columns. Every reduction is a fixed-order two-stage
// tree per column, so results are run-to-run deterministic and each column's arithmetic is
// independent of its neighbours.
#include "amg/cg_kernels.hpp"
#include "core/device_math.cuh"
#include "core/slot_csr.cuh"

#include <cub/device/device_select.cuh>
#include <cuda/iterator>

#include <cstdint>

namespace {

// Threads cooperating on one row in the strided SpMV, which also fixes the association every width
// must reproduce: the column-lane kernel carries this many partial sums per (row, column) and pays
// kTpr - 1 fp64 adds to combine them. At k = 8 that kernel is bound by the 1:64 fp64 pipe, so the
// combine is real cost. Two lanes keep the k = 1 gather coalesced, which one lane does not, and
// spend a single add on the combine.
constexpr int kTpr = 2;

// Blocks-per-SM floor for the SpMV kernels, which is what bounds the register budget ptxas
// will spend. They are bandwidth-bound and want the residency for latency hiding; without the
// floor ptxas unrolls a wide instantiation until only two blocks fit per SM.
constexpr int kSpmvMinBlocks = 6;

template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    bcsrmv_f64_kernel(int n, const int* __restrict__ row_ptr,
                      const std::int16_t* __restrict__ cols, const double* __restrict__ vals,
                      const double* __restrict__ x, double* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = threadIdx.x % kTpr;
    cuda::std::array<double, K> sum;
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.0;
    if (row < n) {
        const WidthView<const double, K> xv(x, n);
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += kTpr) {
            const double v = vals[j];
            const double* xrow = &xv(slot_column(cols, j, row), 0);
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] += v * __ldg(xrow + c);
        }
    }
    warp_reduce_sum<kTpr>(sum);
    if (row < n && lane == 0) {
        const WidthView<double, K> yv(y, n);
#pragma unroll
        for (int c = 0; c < K; ++c) yv(row, c) = sum[c];
    }
}

// One thread per (row, column), which is what K >= 4 uses.
//
// The strided form above gives each thread a K-wide accumulator, so it issues K separate
// eight-byte loads at x[col * K + c] per nonzero, to addresses unrelated to its warp
// neighbours'. A warp then generates up to 32 distinct wavefronts for data occupying eight and
// the kernel saturates L1 request throughput well before DRAM. Giving each lane one column
// instead makes col_idx[j] and vals[j] broadcasts across a row's K lanes and turns the gather
// into one coalesced wavefront. Below K = 4 the strided form still wins, because two lanes per
// row cannot hide the row loop.
//
// A lane owns its column outright, so no cross-lane reduction is needed, but it still carries the
// strided kernel's kTpr accumulators and combines them with the same tree: which kernel runs is a
// throughput choice, and block_k must not move a result.
template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    bcsrmv_f64_collane_kernel(int n, const int* __restrict__ row_ptr,
                              const std::int16_t* __restrict__ cols,
                              const double* __restrict__ vals, const double* __restrict__ x,
                              double* __restrict__ y) {
    const std::int64_t tid =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = static_cast<int>(tid / K);
    if (row >= n) return;
    const int c = static_cast<int>(tid % K);
    double part[kTpr];
#pragma unroll
    for (int l = 0; l < kTpr; ++l) part[l] = 0.0;

    // Walk the row in groups of kTpr so the accumulator index stays a compile-time constant;
    // a running index would spill `part` to local memory. Group g slot l is nonzero
    // row_s + kTpr*g + l, which is exactly the strided kernel's lane-l subsequence.
    const WidthView<const double, K> xv(x, n);
    const int row_s = row_ptr[row];
    const int row_e = row_ptr[row + 1];
    int j = row_s;
    for (; j + kTpr <= row_e; j += kTpr) {
#pragma unroll
        for (int l = 0; l < kTpr; ++l) {
            part[l] += vals[j + l] * xv(slot_column(cols, j + l, row), c);
        }
    }
#pragma unroll
    for (int l = 0; l < kTpr; ++l) {
        if (j + l < row_e) part[l] += vals[j + l] * xv(slot_column(cols, j + l, row), c);
    }

    // The same pairing warp_reduce_sum performs above, in registers rather than across lanes,
    // so both kernels land on the same association (p0 + p1 at kTpr = 2).
#pragma unroll
    for (int off = kTpr / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int l = 0; l < off; ++l) part[l] += part[l + off];
    }
    WidthView<double, K>(y, n)(row, c) = part[0];
}

// The iteration's operator in difference form, in fp32: (A x)_i = sum_j a_ij (x_j - x_i) + s_i x_i,
// with s_i the row sum. A conductivity operator's rows sum to zero, so in the plain form a smooth x
// makes a row's terms cancel down to a small fraction of their size, and fp32 rounding of the
// terms leaves Ap with a relative error of order 1e-5: the CG then reports a residual 10x below
// the true one. Each difference x_j - x_i is exact for neighbouring values, and what remains is
// the fp32 rounding of terms already as small as the result.
//
// The diagonal slot's value is replaced by the row sum (diff_values_kernel), which is nonzero
// only beside the grounded node, and contributes s_i x_i; every other slot contributes
// v (x_col - x_row). A low-half slot decodes to its own row and carries a zero value, so it adds
// zero either way. The operator iterated on is therefore the fp64 one with its off-diagonals
// rounded to fp32: an exact conductivity operator for edge weights perturbed by 6e-8. The fp64
// operator still decides convergence, through the true-residual check in solver.cpp.
__device__ __forceinline__ float diff_term(const std::int16_t* __restrict__ cols, int j, float v,
                                           float xc, float xr, float acc) {
    const float d = (cols[j] == 0) ? xc : __fsub_rn(xc, xr);
    return __fmaf_rn(v, d, acc);
}

// Every width runs this strided form. With fp32 operands a row's K values are one or two vector
// loads, so the L1 request limit that sends the fp64 kernel to column lanes at K >= 4 does not
// bind, and one kernel gives every width the same association by construction.
template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    bcsrmv_diff_kernel(int n, const int* __restrict__ row_ptr,
                       const std::int16_t* __restrict__ cols, const float* __restrict__ vals,
                       const float* __restrict__ x, float* __restrict__ y) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kTpr;
    const int lane = threadIdx.x % kTpr;
    cuda::std::array<float, K> sum;
#pragma unroll
    for (int c = 0; c < K; ++c) sum[c] = 0.f;
    if (row < n) {
        const WidthView<const float, K> xv(x, n);
        cuda::std::array<float, K> xr;
        load_row<K>(&xv(row, 0), xr);
        const int row_e = row_ptr[row + 1];
        for (int j = row_ptr[row] + lane; j < row_e; j += kTpr) {
            const float v = vals[j];
            cuda::std::array<float, K> xc;
            load_row<K>(&xv(slot_column(cols, j, row), 0), xc);
#pragma unroll
            for (int c = 0; c < K; ++c) sum[c] = diff_term(cols, j, v, xc[c], xr[c], sum[c]);
        }
    }
    warp_reduce_sum<kTpr>(sum);
    if (row < n && lane == 0) {
        const WidthView<float, K> yv(y, n);
#pragma unroll
        for (int c = 0; c < K; ++c) yv(row, c) = sum[c];
    }
}

// The affine operator in difference form, each part's segment summed as bcsrmv_diff_kernel sums a
// row, with that part's own row sum on its diagonal slot.
template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    affine_diff_kernel(int n, const int* __restrict__ order, const int* __restrict__ seg_ptr,
                       const int* __restrict__ seg_start, const std::uint8_t* __restrict__ seg_term,
                       const std::int16_t* __restrict__ cols, const float* __restrict__ vals,
                       const float* __restrict__ sigma, int n_terms, const float* __restrict__ x,
                       float* __restrict__ y) {
    const std::int64_t tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int slot = static_cast<int>(tid / K);
    if (slot >= n) return;
    const int row = order[slot];
    const int c = static_cast<int>(tid % K);
    const WidthView<const float, K> xv(x, n);
    const float xr = xv(row, c);
    const float* sig = sigma + c * n_terms;
    float acc = 0.f;
    for (int g = seg_ptr[row]; g < seg_ptr[row + 1]; ++g) {
        float part = 0.f;
        for (int j = seg_start[g]; j < seg_start[g + 1]; ++j) {
            part = diff_term(cols, j, vals[j], xv(slot_column(cols, j, row), c), xr, part);
        }
        acc = __fmaf_rn(sig[seg_term[g]], part, acc);
    }
    WidthView<float, K>(y, n)(row, c) = acc;
}

// Setup only. Whether segment g, [bounds[g], bounds[g + 1]), sums to more than a rounding residue
// of its l1 norm, with the sum left in `s`. The fp32 diagonal and the deflation support both take
// the test from here, so the rows that keep a sum are the deflation support.
__device__ __forceinline__ bool keeps_sum(const int* __restrict__ bounds,
                                          const double* __restrict__ vals, int g, double tol,
                                          double& s) {
    double a = 0.0;
    s = 0.0;
    for (int j = bounds[g]; j < bounds[g + 1]; ++j) {
        s += vals[j];
        a += fabs(vals[j]);
    }
    return fabs(s) > tol * a;
}

// Setup only. The fp32 difference-form values of a slot CSR whose rows (or affine segments) are
// [bounds[i], bounds[i + 1]): every slot's value rounded, except the diagonal's, which becomes the
// segment's sum where it keeps one and zero elsewhere.
__global__ void diff_values_kernel(int n_segs, const int* __restrict__ bounds,
                                   const std::int16_t* __restrict__ cols,
                                   const double* __restrict__ vals, double tol,
                                   float* __restrict__ out) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_segs) return;
    double s;
    const float diag = keeps_sum(bounds, vals, g, tol, s) ? static_cast<float>(s) : 0.f;
    for (int j = bounds[g]; j < bounds[g + 1]; ++j) {
        out[j] = (cols[j] == 0) ? diag : static_cast<float>(vals[j]);
    }
}

// y = A(sigma_c) x_c for an operator affine in per-column coefficients,
// A(sigma) = sum_t sigma_t A_t on one pattern: the conductivity-UQ stiffness, a sum of per-tissue
// parts. A row's entries are grouped into one segment per part that touches it, so each column
// applies its own sigma once per segment and no column's matrix is ever materialised. The parts
// are read once for all K columns, which is what lets independent draws share a block.
//
// One thread per (row, column), each walking its row in order, so a column's arithmetic does not
// depend on K. Entries are slots (core/slot_csr.cuh) and sigma is laid out [column][term].
//
// A warp holds 32 / K rows, and the nested loops cost it, per segment, the longest of its rows'
// segments, so rows are taken in `order`: each window of consecutive rows grouped by segment count,
// then longest first. That leaves every row's arithmetic as it was, and keeps rows near their
// spatial neighbours for the gather. The segment loop stays nested: closing segments inside one
// loop over the slots would put predicated fp64 work on every slot of a kernel that is fp64-bound.
template <int K>
__global__ void __launch_bounds__(kBlock, kSpmvMinBlocks)
    affine_spmv_kernel(int n, const int* __restrict__ order, const int* __restrict__ seg_ptr,
                       const int* __restrict__ seg_start, const std::uint8_t* __restrict__ seg_term,
                       const std::int16_t* __restrict__ cols, const double* __restrict__ vals,
                       const double* __restrict__ sigma, int n_terms,
                       const double* __restrict__ x, double* __restrict__ y) {
    const std::int64_t tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int slot = static_cast<int>(tid / K);
    if (slot >= n) return;
    const int row = order[slot];
    const int c = static_cast<int>(tid % K);
    const WidthView<const double, K> xv(x, n);
    const double* sig = sigma + c * n_terms;
    double acc = 0.0;
    for (int g = seg_ptr[row]; g < seg_ptr[row + 1]; ++g) {
        double part = 0.0;
        for (int j = seg_start[g]; j < seg_start[g + 1]; ++j) {
            part += vals[j] * xv(slot_column(cols, j, row), c);
        }
        acc += sig[seg_term[g]] * part;
    }
    WidthView<double, K>(y, n)(row, c) = acc;
}

// Rows each thread of a reducing kernel carries. A warp's shuffle tree spends five fp64 adds per
// lane and per column to combine 32 values, and fp64 runs at a sixty-fourth of fp32 on a GeForce
// part, so the tree, not the loads, is what bounds these kernels. Folding rows into a thread's
// accumulator first replaces eight trees with seven sequential adds and one tree. The rows a
// thread takes are a grid stride apart, so every access stays as coalesced as it was at one row per
// thread. Past eight the grid stops covering the SMs.
constexpr int kRowsPerThread = 8;

// Columns the deflation state is held for, one per possible block column.
constexpr int kDeflCols = kMaxStageBlock;

// Per-block reduction of K register accumulators into one partial per (column, block): warp
// shuffles, one cross-warp pass through shared memory, a single __syncthreads(). Keeping it to
// one barrier matters because the streaming kernels that embed it are bandwidth-bound.
//
// The order is the rows a thread folded in, then the shuffle tree within each warp, then a
// sequential sum over the kBlock/kWarp warp results. None of that depends on K, which is what
// makes the same placement solved at block_k=1 and block_k=8 come back bitwise identical.
template <int K>
__device__ __forceinline__ void bcg_block_reduce_cols(cuda::std::array<double, K>& local,
                                                      double* __restrict__ partials,
                                                      int n_blocks) {
    constexpr int kWarps = kBlock / kWarp;
    __shared__ double s[kWarps][K];
    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    warp_reduce_sum<kWarp>(local);
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < K; ++c) s[warp][c] = local[c];
    }
    __syncthreads();
    if (threadIdx.x < K) {
        double v = 0.0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) v += s[w][threadIdx.x];
        partials[static_cast<std::int64_t>(threadIdx.x) * n_blocks + blockIdx.x] = v;
    }
}

// What a column-wise reduction folds each entry into.
enum class Fold { kDot, kNorm, kSum };

// A thread's fold over the rows it carries: fp64 for fp64 operands, and fp32 for the fp32 CG
// vectors, whose products and kRowsPerThread-term sums are then widened once for the fp64 tree.
// Converting every element instead would put two conversions and an fp64 add per element on the
// 1:64 fp64 pipe, as many as the loads can feed.
template <typename T>
__device__ __forceinline__ T fold_mul_add(T a, T b, T acc) {
    if constexpr (cuda::std::is_same_v<T, float>) {
        return __fmaf_rn(a, b, acc);
    } else {
        return acc + a * b;
    }
}

template <typename T>
__device__ __forceinline__ T fold_add(T a, T acc) {
    if constexpr (cuda::std::is_same_v<T, float>) {
        return __fadd_rn(acc, a);
    } else {
        return acc + a;
    }
}

template <int K, typename T>
__device__ __forceinline__ cuda::std::array<double, K> widen(const cuda::std::array<T, K>& v) {
    cuda::std::array<double, K> out;
#pragma unroll
    for (int c = 0; c < K; ++c) out[c] = static_cast<double>(v[c]);
    return out;
}

// Column-wise reductions, stage 1: one partial per (column, block). y is read only by kDot. x and
// y are both fp64 or both fp32, and the fold runs in their precision. A grid-stride fold that
// loads and adds one row at a time serialises its loads behind the adds, so the rows are loaded a
// batch at a time and then folded in row order.
template <int K, Fold F, typename T = double>
__global__ void bcg_partials_kernel(int n, const T* __restrict__ x, const T* __restrict__ y,
                                    double* __restrict__ partials, int n_blocks) {
    const WidthView<const T, K> xv(x, n);
    const WidthView<const T, K> yv(y, n);
    cuda::std::array<T, K> local;
#pragma unroll
    for (int c = 0; c < K; ++c) local[c] = T(0);
    const int stride = gridDim.x * blockDim.x;
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int kBatch = 4;
    static_assert(kRowsPerThread % kBatch == 0, "batches must tile the rows a thread carries");
#pragma unroll
    for (int t0 = 0; t0 < kRowsPerThread; t0 += kBatch) {
        cuda::std::array<T, K> xb[kBatch];
        cuda::std::array<T, K> yb[kBatch];
#pragma unroll
        for (int t = 0; t < kBatch; ++t) {
            const int i = i0 + (t0 + t) * stride;
            if (i < n) {
                load_row<K>(&xv(i, 0), xb[t]);
                if constexpr (F == Fold::kDot) load_row<K>(&yv(i, 0), yb[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < kBatch; ++t) {
            if (i0 + (t0 + t) * stride >= n) break;
#pragma unroll
            for (int c = 0; c < K; ++c) {
                const T xc = xb[t][c];
                if constexpr (F == Fold::kDot) {
                    local[c] = fold_mul_add(xc, yb[t][c], local[c]);
                } else if constexpr (F == Fold::kNorm) {
                    local[c] = fold_mul_add(xc, xc, local[c]);
                } else {
                    local[c] = fold_add(xc, local[c]);
                }
            }
        }
    }
    cuda::std::array<double, K> wide = widen<K>(local);
    bcg_block_reduce_cols<K>(wide, partials, n_blocks);
}

// Stage 2: one block per column folds that column's partials into a single total, returned to
// every thread of the block. The two kernels below differ only in what they then do with it, and
// the sum order here is the same for both, so which one ran cannot move a result. Every thread of
// the block must reach this.
__device__ __forceinline__ double bcg_reduce_column(const double* __restrict__ partials,
                                                    int n_blocks, int c) {
    __shared__ double sdata[kBlock];
    double local = 0.0;
    for (int b = threadIdx.x; b < n_blocks; b += kBlock) {
        local += partials[static_cast<std::int64_t>(c) * n_blocks + b];
    }
    return block_tree_sum(local, sdata);
}

__global__ void bcg_reduce_kernel(const double* __restrict__ partials, int n_blocks,
                                  double* __restrict__ out) {
    const int c = blockIdx.x;
    const double total = bcg_reduce_column(partials, n_blocks, c);
    if (threadIdx.x == 0) out[c] = total;
}

// A restart touches only the columns its mask marks; every other column's state stays as it was.
__device__ __forceinline__ bool marked(const double* __restrict__ mask, int c) {
    return mask == nullptr || mask[c] != 0.0;
}

__global__ void bcg_reduce_masked_kernel(const double* __restrict__ partials, int n_blocks,
                                         const double* __restrict__ mask,
                                         double* __restrict__ out) {
    const int c = blockIdx.x;
    const double total = bcg_reduce_column(partials, n_blocks, c);
    if (threadIdx.x == 0 && marked(mask, c)) out[c] = total;
}

// Deflation of the constant vector w = 1, in the A-DEF2 form of Tang, Nabben, Vuik and Erlangga
// (2009): the preconditioner becomes z = P^T M^-1 r + Q r with Q = w w^T / E, E = w^T A w and
// P^T = I - Q A.
//
// Grounding one node of a pure-Neumann operator leaves the constant as a near-null mode that the
// aggregation hierarchy does not capture. On a head mesh the preconditioned spectrum has a single
// eigenvalue near 4e-3 against a bulk above 0.05, its eigenvector is the constant everywhere but
// at the ground, and deflating it takes 21-27% of the CG iterations off.
//
// The constant is what makes it nearly free. A w is a stiffness row sum, zero everywhere but at
// the ground node's neighbours, so it is kept only on that support, and both corrections are
// scalar per column:
//     z = z~ + shift * 1,   shift = (sum(r) - Aw . z~) / E,   z~ = M^-1 r,
//     r . z = r . z~ + shift * sum(r).
// sum(r) rides on the residual-norm reduction and Aw . z~ is a handful of loads in the kernel that
// folds r . z~. Aw and 1/E are held per column, because an affine operator gives every column its
// own matrix. It never moves the fixed point, only the preconditioner. inv_e = 0 switches it off
// and leaves every shift exactly zero, which is what an operator with no positive E gets.

// Setup only: whether a row keeps its sum.
__global__ void bcg_support_flags_kernel(int n, const int* __restrict__ row_ptr,
                                         const double* __restrict__ vals, double tol,
                                         unsigned char* __restrict__ flags) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double s;
    flags[row] = keeps_sum(row_ptr, vals, row, tol, s);
}

// Aw on the support from the operator's own values, the same in every column. In the CSR's own
// order, so the projection is fixed by the matrix alone.
__global__ void bcg_defl_rows_kernel(int n_s, const int* __restrict__ rows,
                                     const int* __restrict__ row_ptr,
                                     const double* __restrict__ vals, double* __restrict__ aw) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_s) return;
    const int row = rows[t];
    double s = 0.0;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) s += vals[j];
#pragma unroll
    for (int c = 0; c < kDeflCols; ++c) aw[t * kDeflCols + c] = s;
}

// Aw on the support for an affine operator: column c's is sum_t sigma[c, t] Aw_t.
__global__ void bcg_defl_affine_kernel(int n_s, int n_terms, const double* __restrict__ aw_t,
                                       const double* __restrict__ sigma, double* __restrict__ aw) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_s) return;
    for (int c = 0; c < kDeflCols; ++c) {
        double s = 0.0;
        for (int term = 0; term < n_terms; ++term) {
            s += sigma[c * n_terms + term] * aw_t[static_cast<std::int64_t>(term) * n_s + t];
        }
        aw[t * kDeflCols + c] = s;
    }
}

// 1/E per column, E summed over the support in its own order.
__global__ void bcg_defl_energy_kernel(int n_s, const double* __restrict__ aw,
                                       double* __restrict__ inv_e) {
    const int c = threadIdx.x;
    if (c >= kDeflCols) return;
    double e = 0.0;
    for (int t = 0; t < n_s; ++t) e += aw[t * kDeflCols + c];
    inv_e[c] = (e > 0.0 && isfinite(e)) ? 1.0 / e : 0.0;
}

// The deflated start x0 <- x0 + Q r0, which leaves r0 orthogonal to w. delta = sum(r0) / E is the
// whole correction to x; r moves by delta * Aw, which is only on the support. This is that half.
template <int K>
__global__ void bcg_deflate_rows_kernel(int n_s, const int* __restrict__ rows,
                                        const double* __restrict__ rsum0,
                                        const double* __restrict__ inv_e,
                                        const double* __restrict__ aw, float* __restrict__ r) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_s) return;
    const std::int64_t row = rows[t];
#pragma unroll
    for (int c = 0; c < K; ++c) {
        r[row * K + c] = static_cast<float>(static_cast<double>(r[row * K + c]) -
                                            (rsum0[c] * inv_e[c]) * aw[t * kDeflCols + c]);
    }
}

// The dense half: x += delta everywhere. Also folds the new sum(r) into partials, which the first
// preconditioned residual needs.
template <int K>
__global__ void bcg_deflate_start_kernel(int n, const double* __restrict__ rsum0,
                                         const double* __restrict__ inv_e, double* __restrict__ x,
                                         const float* __restrict__ r,
                                         double* __restrict__ partials, int n_blocks) {
    const WidthView<double, K> xrows(x, n);
    const WidthView<const float, K> rrows(r, n);
    cuda::std::array<double, K> delta;
    cuda::std::array<float, K> sloc;
#pragma unroll
    for (int c = 0; c < K; ++c) {
        delta[c] = rsum0[c] * inv_e[c];
        sloc[c] = 0.f;
    }
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < kRowsPerThread; ++t, i += stride) {
        if (i >= n) break;
        cuda::std::array<double, K> xv;
        cuda::std::array<float, K> rv;
        load_row<K>(&xrows(i, 0), xv);
        load_row<K>(&rrows(i, 0), rv);
#pragma unroll
        for (int c = 0; c < K; ++c) {
            xv[c] += delta[c];
            sloc[c] = fold_add(rv[c], sloc[c]);
        }
        store_row<K>(&xrows(i, 0), xv);
    }
    cuda::std::array<double, K> wide = widen<K>(sloc);
    bcg_block_reduce_cols<K>(wide, partials, n_blocks);
}

// Aw . zf for column c over the support, in a fixed order. Every thread of the block must reach
// this, and it returns the total to all of them.
__device__ __forceinline__ double bcg_support_dot(int n_s, const int* __restrict__ rows,
                                                  const double* __restrict__ aw,
                                                  const float* __restrict__ zf, int k, int c) {
    __shared__ double sdot[kBlock];
    double local = 0.0;
    for (int t = threadIdx.x; t < n_s; t += kBlock) {
        local += aw[t * kDeflCols + c] *
                 static_cast<double>(zf[static_cast<std::int64_t>(rows[t]) * k + c]);
    }
    return block_tree_sum(local, sdot);
}

// A column freezes at the iteration its own residual first meets tolerance, whatever the block's
// other columns do. Both alpha and beta go to zero, which leaves x, r and rz untouched and pins p
// at the preconditioned residual, so the column contributes nothing further and its answer is the
// one it would have had solved on its own.
//
// Freezing on the device keeps the CG body replayable as a captured graph: the kernels are the
// same every iteration and only the mask's contents change.
__device__ __forceinline__ bool frozen(const double* __restrict__ converged, int c) {
    return converged[c] != 0.0;
}

__global__ void bcg_mark_converged_kernel(int k, const double* __restrict__ norm_sq,
                                          const double* __restrict__ ref_sq,
                                          const double* __restrict__ tolerance,
                                          double* __restrict__ converged) {
    const int c = threadIdx.x;
    if (c < k) {
        // The same expression the host stopping test uses. fp64 sqrt and divide are both
        // correctly rounded, so the two sides agree bit for bit on the same inputs. If they could
        // disagree by an ulp at the boundary, a column the device had frozen but the host still
        // called unconverged would stop improving and run the block to max_iters.
        converged[c] = (sqrt(norm_sq[c]) / sqrt(ref_sq[c]) <= *tolerance) ? 1.0 : 0.0;
    }
}

__global__ void bcg_alpha_kernel(int k, const double* __restrict__ rz,
                                 const double* __restrict__ pap,
                                 const double* __restrict__ converged, double* __restrict__ alpha,
                                 double* __restrict__ neg_alpha) {
    const int c = threadIdx.x;
    if (c < k) {
        // A branch, because a column frozen at an exactly zero residual has rz = pap = 0, and a
        // quotient masked by multiplication would be 0/0 * 0, a NaN in x. Rounded to fp32, the
        // precision the r update applies it in, so that x and r take the same step.
        const double a =
            frozen(converged, c) ? 0.0 : static_cast<double>(static_cast<float>(rz[c] / pap[c]));
        alpha[c] = a;
        neg_alpha[c] = -a;
    }
}

// r -= alpha Ap, with |r|^2 and sum(r) from the same pass. x is not touched here: its
// x += alpha p rides on bcg_update_p_kernel, which reads p anyway, so p is read once per
// iteration.
template <int K>
__global__ void bcg_update_r_kernel(int n, const double* __restrict__ neg_alpha,
                                    const float* __restrict__ ap, float* __restrict__ r,
                                    double* __restrict__ partials, int n_blocks) {
    const WidthView<const float, K> aprows(ap, n);
    const WidthView<float, K> rrows(r, n);
    cuda::std::array<float, K> na;
#pragma unroll
    for (int c = 0; c < K; ++c) na[c] = static_cast<float>(__ldg(neg_alpha + c));
    // [0, K) is |r|^2 and [K, 2K) is sum(r), which the deflation's Q r term needs.
    cuda::std::array<float, 2 * K> rloc;
#pragma unroll
    for (int c = 0; c < 2 * K; ++c) rloc[c] = 0.f;
    const int stride = gridDim.x * blockDim.x;
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    // Rows go in batches whose loads are all issued before any is consumed. A plain loop with an
    // early exit leaves each row's loads behind the previous row's stores, and with only two
    // streams per row that is latency-bound at 40% of DRAM. The fold into rloc keeps row order.
    constexpr int kBatch = 4;
    static_assert(kRowsPerThread % kBatch == 0, "batches must tile the rows a thread carries");
#pragma unroll
    for (int t0 = 0; t0 < kRowsPerThread; t0 += kBatch) {
        cuda::std::array<float, K> apv[kBatch], rv[kBatch];
#pragma unroll
        for (int t = 0; t < kBatch; ++t) {
            const int i = i0 + (t0 + t) * stride;
            if (i < n) {
                load_row<K>(&aprows(i, 0), apv[t]);
                load_row<K>(&rrows(i, 0), rv[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < kBatch; ++t) {
            const int i = i0 + (t0 + t) * stride;
            if (i >= n) break;
#pragma unroll
            for (int c = 0; c < K; ++c) {
                rv[t][c] = __fmaf_rn(na[c], apv[t][c], rv[t][c]);
                rloc[c] = fold_mul_add(rv[t][c], rv[t][c], rloc[c]);
                rloc[K + c] = fold_add(rv[t][c], rloc[K + c]);
            }
            store_row<K>(&rrows(i, 0), rv[t]);
        }
    }
    cuda::std::array<double, 2 * K> wide = widen<2 * K>(rloc);
    bcg_block_reduce_cols<2 * K>(wide, partials, n_blocks);
}

// Folds column c of r . zf, adds the support's Aw . zf, and turns them into the deflated r . z
// and the shift that takes zf to z. Every thread of the block must reach this.
__device__ __forceinline__ void bcg_deflated_rz(const double* __restrict__ partials, int n_blocks,
                                                int c, Deflation defl, const float* __restrict__ zf,
                                                int k, const double* __restrict__ rsum,
                                                double& rz, double& shift) {
    const double rzt = bcg_reduce_column(partials, n_blocks, c);
    const double awz = bcg_support_dot(defl.n_rows, defl.rows, defl.aw, zf, k, c);
    const double s = rsum[c];
    shift = (s - awz) * defl.inv_e[c];
    rz = rzt + shift * s;
}

__global__ void bcg_reduce_init_kernel(const double* __restrict__ partials, int n_blocks,
                                       Deflation defl, const float* __restrict__ zf,
                                       const double* __restrict__ rsum,
                                       const double* __restrict__ mask, double* __restrict__ rz,
                                       double* __restrict__ shift) {
    const int c = blockIdx.x;
    double total, sh;
    bcg_deflated_rz(partials, n_blocks, c, defl, zf, gridDim.x, rsum, total, sh);
    if (threadIdx.x == 0 && marked(mask, c)) {
        rz[c] = total;
        shift[c] = sh;
    }
}

__global__ void bcg_reduce_beta_kernel(const double* __restrict__ partials, int n_blocks,
                                       const double* __restrict__ converged, Deflation defl,
                                       const float* __restrict__ zf,
                                       const double* __restrict__ rsum, double* __restrict__ rz,
                                       double* __restrict__ beta, double* __restrict__ shift) {
    const int c = blockIdx.x;
    double total, sh;
    bcg_deflated_rz(partials, n_blocks, c, defl, zf, gridDim.x, rsum, total, sh);
    if (threadIdx.x == 0) {
        // beta = 0 on a frozen column pins p at z. Left at total/rz it would be 1, since r has
        // not moved, and p would grow by z every remaining iteration for no purpose.
        beta[c] = frozen(converged, c) ? 0.0 : total / rz[c];
        rz[c] = total;
        shift[c] = sh;
    }
}

// x += alpha p, then p = beta p + z with z = zf + shift rebuilt on the fly as everywhere else.
// alpha is still this iteration's and p still the one the SpMV read, so x takes exactly the
// update it would have taken beside r. x is the one fp64 vector: a grounded potential is large
// against its change across an element, and every field is a difference of it.
template <int K>
__global__ void bcg_update_p_kernel(int n, const double* __restrict__ alpha,
                                    const double* __restrict__ beta,
                                    const double* __restrict__ shift,
                                    const float* __restrict__ zf, float* __restrict__ p,
                                    double* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const WidthView<float, K> prows(p, n);
        const WidthView<double, K> xrows(x, n);
        const WidthView<const float, K> zfv(zf, n);
        cuda::std::array<float, K> pv, zv;
        cuda::std::array<double, K> xv;
        load_row<K>(&prows(i, 0), pv);
        load_row<K>(&xrows(i, 0), xv);
        load_row<K>(&zfv(i, 0), zv);
#pragma unroll
        for (int c = 0; c < K; ++c) {
            xv[c] += __ldg(alpha + c) * static_cast<double>(pv[c]);
            pv[c] = __fmaf_rn(static_cast<float>(__ldg(beta + c)), pv[c],
                              __fadd_rn(zv[c], static_cast<float>(__ldg(shift + c))));
        }
        store_row<K>(&xrows(i, 0), xv);
        store_row<K>(&prows(i, 0), pv);
    }
}

// p0 = z0 = zf + shift.
template <int K>
__global__ void bcg_init_p_kernel(int n, const double* __restrict__ shift,
                                  const float* __restrict__ zf, const double* __restrict__ mask,
                                  float* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const WidthView<float, K> pv(p, n);
        const WidthView<const float, K> zfv(zf, n);
#pragma unroll
        for (int c = 0; c < K; ++c) {
            if (marked(mask, c)) {
                pv(i, c) = __fadd_rn(zfv(i, c), static_cast<float>(__ldg(shift + c)));
            }
        }
    }
}

// The fp64 residual b - Ap of the fp64 operator, which is what convergence is measured on: |r|^2
// per column into partials, and r rounded into the fp32 CG residual for the columns `write`
// marks (all of them when it is null). A null ap is a zero x.
template <int K>
__global__ void bcg_residual_kernel(int n, const double* __restrict__ b,
                                    const double* __restrict__ ap,
                                    const double* __restrict__ write, float* __restrict__ r,
                                    double* __restrict__ partials, int n_blocks) {
    const WidthView<const double, K> bv(b, n);
    const WidthView<const double, K> apv(ap, n);
    const WidthView<float, K> rv(r, n);
    cuda::std::array<bool, K> keep;
#pragma unroll
    for (int c = 0; c < K; ++c) keep[c] = write == nullptr || write[c] != 0.0;
    cuda::std::array<double, K> local;
#pragma unroll
    for (int c = 0; c < K; ++c) local[c] = 0.0;
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < kRowsPerThread; ++t, i += stride) {
        if (i >= n) break;
        cuda::std::array<double, K> bb, aa{};
        load_row<K>(&bv(i, 0), bb);
        if (ap != nullptr) load_row<K>(&apv(i, 0), aa);
#pragma unroll
        for (int c = 0; c < K; ++c) {
            const double d = bb[c] - aa[c];
            local[c] += d * d;
            if (keep[c]) rv(i, c) = static_cast<float>(d);
        }
    }
    bcg_block_reduce_cols<K>(local, partials, n_blocks);
}

constexpr const char* kBadK = "block CG supports k in {1, 2, 4, 8}";

}  // namespace

int bcg_partials_blocks(int n) { return static_cast<int>(grid_for(n, kBlock * kRowsPerThread)); }

void launch_bcsrmv_f64_block(int n, int k, const int* row_ptr, const std::int16_t* cols,
                             const double* vals, const double* x, double* y,
                             cudaStream_t stream) {
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        constexpr int kThreadsPerRow = (K >= 4) ? K : kTpr;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * kThreadsPerRow)) {
            if constexpr (K >= 4) {
                bcsrmv_f64_collane_kernel<K>
                    <<<blocks, kBlock, 0, stream>>>(n, row_ptr, cols, vals, x, y);
            } else {
                bcsrmv_f64_kernel<K>
                    <<<blocks, kBlock, 0, stream>>>(n, row_ptr, cols, vals, x, y);
            }
        }
    });
}

void launch_bcsrmv_diff_block(int n, int k, const int* row_ptr, const std::int16_t* cols,
                              const float* vals, const float* x, float* y, cudaStream_t stream) {
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * kTpr)) {
            bcsrmv_diff_kernel<K><<<blocks, kBlock, 0, stream>>>(n, row_ptr, cols, vals, x, y);
        }
    });
}

void launch_affine_diff_block(int n, int k, const AffineOperator& op, const float* vals,
                              const float* sigma, const float* x, float* y, cudaStream_t stream) {
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * K)) {
            affine_diff_kernel<K><<<blocks, kBlock, 0, stream>>>(
                n, op.order, op.seg_ptr, op.seg_start, op.seg_term, op.cols, vals, sigma,
                op.n_terms, x, y);
        }
    });
}

void launch_diff_values(int n_segs, const int* bounds, const std::int16_t* cols,
                        const double* vals, double tol, float* out, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_segs)) {
        diff_values_kernel<<<blocks, kBlock, 0, stream>>>(n_segs, bounds, cols, vals, tol, out);
    }
}

void launch_affine_spmv_block(int n, int k, const AffineOperator& op, const double* sigma,
                              const double* x, double* y, cudaStream_t stream) {
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * K)) {
            affine_spmv_kernel<K><<<blocks, kBlock, 0, stream>>>(
                n, op.order, op.seg_ptr, op.seg_start, op.seg_term, op.cols, op.vals, sigma,
                op.n_terms, x, y);
        }
    });
}

void launch_bcg_dot(int n, int k, const float* x, const float* y, double* partials, double* out,
                    cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, Fold::kDot, float>
            <<<n_blocks, kBlock, 0, stream>>>(n, x, y, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, out);
}

void launch_bcg_norm2(int n, int k, const double* x, double* partials, double* out,
                      cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, Fold::kNorm>
            <<<n_blocks, kBlock, 0, stream>>>(n, x, static_cast<const double*>(nullptr),
                                              partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, out);
}

namespace {

void launch_bcg_sum(int n, int k, const float* x, double* partials, double* out,
                    cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, Fold::kSum, float>
            <<<n_blocks, kBlock, 0, stream>>>(n, x, static_cast<const float*>(nullptr),
                                              partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, out);
}

}  // namespace

int select_deflation_rows(int n, const int* row_ptr, const double* vals, double tol, int* rows,
                          cudaStream_t stream) {
    const ScratchBuffer<unsigned char> flags =
        scratch_alloc<unsigned char>(static_cast<std::size_t>(n), stream, "solver", "support");
    const ScratchBuffer<int> count = scratch_alloc<int>(1, stream, "solver", "support count");
    bcg_support_flags_kernel<<<grid_for(n), kBlock, 0, stream>>>(n, row_ptr, vals, tol,
                                                                 flags.get());
    const cuda::counting_iterator<int> row_ids(0);
    std::size_t bytes = 0;
    check_cuda(cub::DeviceSelect::Flagged(nullptr, bytes, row_ids, flags.get(), rows, count.get(),
                                          n, stream),
               "solver", "support select size");
    const ScratchBuffer<std::byte> temp =
        scratch_alloc<std::byte>(bytes, stream, "solver", "support select");
    check_cuda(cub::DeviceSelect::Flagged(temp.get(), bytes, row_ids, flags.get(), rows,
                                          count.get(), n, stream),
               "solver", "support select");
    int n_rows = 0;
    check_cuda(cudaMemcpyAsync(&n_rows, count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
               "solver", "support count");
    check_cuda(cudaStreamSynchronize(stream), "solver", "support count");
    return n_rows;
}

void launch_bcg_defl_from_values(int n_s, const int* rows, const int* row_ptr,
                                 const double* vals, double* aw, double* inv_e,
                                 cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_s)) {
        bcg_defl_rows_kernel<<<blocks, kBlock, 0, stream>>>(n_s, rows, row_ptr, vals, aw);
    }
    bcg_defl_energy_kernel<<<1, kDeflCols, 0, stream>>>(n_s, aw, inv_e);
}

void launch_bcg_defl_affine(int n_s, int n_terms, const double* aw_t, const double* sigma,
                            double* aw, double* inv_e, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_s)) {
        bcg_defl_affine_kernel<<<blocks, kBlock, 0, stream>>>(n_s, n_terms, aw_t, sigma, aw);
    }
    bcg_defl_energy_kernel<<<1, kDeflCols, 0, stream>>>(n_s, aw, inv_e);
}

void launch_bcg_deflate_start(int n, int k, Deflation defl, double* x, float* r,
                              double* partials, double* rsum, cudaStream_t stream) {
    launch_bcg_sum(n, k, r, partials, rsum, stream);
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(defl.n_rows)) {
            bcg_deflate_rows_kernel<K><<<blocks, kBlock, 0, stream>>>(defl.n_rows, defl.rows,
                                                                      rsum, defl.inv_e, defl.aw,
                                                                      r);
        }
        bcg_deflate_start_kernel<K><<<n_blocks, kBlock, 0, stream>>>(n, rsum, defl.inv_e, x, r,
                                                                     partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, rsum);
}

void launch_bcg_alpha(int k, const double* rz, const double* pap, const double* converged,
                      double* alpha, double* neg_alpha, cudaStream_t stream) {
    bcg_alpha_kernel<<<1, k, 0, stream>>>(k, rz, pap, converged, alpha, neg_alpha);
}

void launch_bcg_mark_converged(int k, const double* norm_sq, const double* ref_sq,
                               const double* tolerance, double* converged, cudaStream_t stream) {
    bcg_mark_converged_kernel<<<1, k, 0, stream>>>(k, norm_sq, ref_sq, tolerance, converged);
}

void launch_bcg_update_r_norm(int n, int k, const double* neg_alpha, const float* ap, float* r,
                              double* partials, double* norms_sums, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_update_r_kernel<K>
            <<<n_blocks, kBlock, 0, stream>>>(n, neg_alpha, ap, r, partials, n_blocks);
    });
    bcg_reduce_kernel<<<2 * k, kBlock, 0, stream>>>(partials, n_blocks, norms_sums);
}

void launch_bcg_cast_dot_beta(int n, int k, const float* zf, const float* r, Deflation defl,
                              const double* converged, const double* rsum, double* partials,
                              double* rz, double* beta, double* shift, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, Fold::kDot, float>
            <<<n_blocks, kBlock, 0, stream>>>(n, r, zf, partials, n_blocks);
    });
    bcg_reduce_beta_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, converged, defl, zf,
                                                     rsum, rz, beta, shift);
}

void launch_bcg_cast_dot_init(int n, int k, const float* zf, const float* r, Deflation defl,
                              const double* rsum, double* partials, double* rz, double* shift,
                              float* p, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, Fold::kDot, float>
            <<<n_blocks, kBlock, 0, stream>>>(n, r, zf, partials, n_blocks);
        bcg_reduce_init_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, defl, zf, rsum,
                                                         nullptr, rz, shift);
        if (const unsigned blocks = grid_for(n)) {
            bcg_init_p_kernel<K><<<blocks, kBlock, 0, stream>>>(n, shift, zf, nullptr, p);
        }
    });
}

void launch_bcg_restart(int n, int k, const float* zf, const float* r, Deflation defl,
                        const double* mask, double* partials, double* rsum, double* rz,
                        double* shift, float* p, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_partials_kernel<K, Fold::kSum, float>
            <<<n_blocks, kBlock, 0, stream>>>(n, r, static_cast<const float*>(nullptr), partials,
                                              n_blocks);
        bcg_reduce_masked_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, mask, rsum);
        bcg_partials_kernel<K, Fold::kDot, float>
            <<<n_blocks, kBlock, 0, stream>>>(n, r, zf, partials, n_blocks);
        bcg_reduce_init_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, defl, zf, rsum,
                                                         mask, rz, shift);
        if (const unsigned blocks = grid_for(n)) {
            bcg_init_p_kernel<K><<<blocks, kBlock, 0, stream>>>(n, shift, zf, mask, p);
        }
    });
}


void launch_bcg_update_p(int n, int k, const double* alpha, const double* beta,
                         const double* shift, const float* zf, float* p, double* x,
                         cudaStream_t stream) {
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        if (const unsigned blocks = grid_for(n)) {
            bcg_update_p_kernel<K>
                <<<blocks, kBlock, 0, stream>>>(n, alpha, beta, shift, zf, p, x);
        }
    });
}

void launch_bcg_residual(int n, int k, const double* b, const double* ap, const double* write,
                         float* r, double* partials, double* norms, cudaStream_t stream) {
    const int n_blocks = bcg_partials_blocks(n);
    dispatch_k<1, 2, 4, 8>(k, kBadK, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        bcg_residual_kernel<K>
            <<<n_blocks, kBlock, 0, stream>>>(n, b, ap, write, r, partials, n_blocks);
    });
    bcg_reduce_kernel<<<k, kBlock, 0, stream>>>(partials, n_blocks, norms);
}
