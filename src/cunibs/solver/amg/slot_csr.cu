#include "amg/slot_csr.hpp"
#include "core/scan.cuh"
#include "core/slot_csr.cuh"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

__device__ __forceinline__ bool slot_is_near(int row, int col) {
    const int d = col - row;
    return d >= kSlotNearMin && d <= kSlotNearMax;
}

__global__ void slot_count_kernel(int n, const int* __restrict__ row_ptr,
                                  const int* __restrict__ col_idx, int* __restrict__ counts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int slots = 0;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        slots += slot_is_near(row, col_idx[p]) ? 1 : 2;
    }
    counts[row] = slots;
}

__global__ void slot_fill_kernel(int n, const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_idx, const int* __restrict__ slot_ptr,
                                 std::int16_t* __restrict__ cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int s = slot_ptr[row];
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const int col = col_idx[p];
        if (slot_is_near(row, col)) {
            cols[s++] = static_cast<std::int16_t>(col - row);
        } else {
            cols[s++] = static_cast<std::int16_t>(0x8000 | (col >> kSlotHalfBits));
            cols[s++] = static_cast<std::int16_t>(0x4000 | (col & kSlotHalfMask));
        }
    }
}

__global__ void slot_scatter_kernel(int n, const int* __restrict__ row_ptr,
                                    const int* __restrict__ slot_ptr,
                                    const std::int16_t* __restrict__ cols,
                                    const double* __restrict__ src, double* __restrict__ dst) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int p = row_ptr[row];
    for (int s = slot_ptr[row]; s < slot_ptr[row + 1]; ++s) {
        dst[s] = (cols[s] > kSlotNearMax) ? 0.0 : src[p++];
    }
}

// Walks row's slots alongside its entries: a low-half slot belongs to the far column before it and
// has no entry of its own. f(entry, slot, far) is called once per entry.
template <typename F>
__device__ __forceinline__ void for_each_entry(int row, const int* __restrict__ row_ptr,
                                               const int* __restrict__ slot_ptr,
                                               const std::int16_t* __restrict__ slot_cols, F&& f) {
    int p = row_ptr[row];
    for (int s = slot_ptr[row]; s < slot_ptr[row + 1]; ++s) {
        const int c = slot_cols[s];
        if (c > kSlotNearMax) continue;
        f(p++, s, c < kSlotNearMin);
    }
}

// The inverse of the fill and the scatter: a row's entries back in entry order, its columns
// decoded and its values copied, either of which may be skipped with a null output.
__global__ void slot_gather_kernel(int n, const int* __restrict__ row_ptr,
                                   const int* __restrict__ slot_ptr,
                                   const std::int16_t* __restrict__ cols,
                                   const double* __restrict__ src, int* __restrict__ col_idx,
                                   double* __restrict__ values) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for_each_entry(row, row_ptr, slot_ptr, cols, [&](int p, int s, bool) {
        if (col_idx != nullptr) col_idx[p] = slot_column(cols, s, row);
        if (values != nullptr) values[p] = src[s];
    });
}

// out[p] = sum_t sigma[t] A_t[p] over a row's entries. A segment's entries are a subset of the
// row's, in the same column order, so one forward walk over the row's slots places each of them.
__global__ void affine_values_kernel(int n, const int* __restrict__ row_ptr,
                                     const int* __restrict__ slot_ptr,
                                     const std::int16_t* __restrict__ slot_cols,
                                     const int* __restrict__ seg_ptr,
                                     const int* __restrict__ seg_start,
                                     const std::uint8_t* __restrict__ seg_term,
                                     const std::int16_t* __restrict__ cols,
                                     const double* __restrict__ vals,
                                     const double* __restrict__ sigma, double* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) out[p] = 0.0;
    for (int g = seg_ptr[row]; g < seg_ptr[row + 1]; ++g) {
        const double coef = sigma[seg_term[g]];
        int s = slot_ptr[row];
        int p = row_ptr[row];
        for (int j = seg_start[g]; j < seg_start[g + 1]; ++j) {
            if (cols[j] > kSlotNearMax) continue;
            const int col = slot_column(cols, j, row);
            while (slot_cols[s] > kSlotNearMax || slot_column(slot_cols, s, row) != col) {
                if (slot_cols[s] <= kSlotNearMax) ++p;
                ++s;
            }
            out[p] += coef * vals[j];
            ++s;
            ++p;
        }
    }
}

// y (n, m) = A_t x for row-major x, one thread per output entry. A row's entries of part t are
// summed in column order with one fused multiply-add each, skipping the zero low halves of far
// columns, which is the CSR product over the part's own nonzeros.
__global__ void affine_term_matmul_kernel(int n, int m, int term, const int* __restrict__ seg_ptr,
                                          const int* __restrict__ seg_start,
                                          const std::uint8_t* __restrict__ seg_term,
                                          const std::int16_t* __restrict__ cols,
                                          const double* __restrict__ vals,
                                          const double* __restrict__ x, double* __restrict__ y) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(n) * m) return;
    const int row = static_cast<int>(i / m);
    const int c = static_cast<int>(i % m);
    double acc = 0.0;
    for (int g = seg_ptr[row]; g < seg_ptr[row + 1]; ++g) {
        if (seg_term[g] != term) continue;
        for (int j = seg_start[g]; j < seg_start[g + 1]; ++j) {
            if (cols[j] > kSlotNearMax) continue;
            const std::int64_t col = slot_column(cols, j, row);
            acc = fma(vals[j], x[col * m + c], acc);
        }
        break;
    }
    y[i] = acc;
}

__global__ void affine_count_kernel(int n, int nnz, const int* __restrict__ row_ptr,
                                    const int* __restrict__ slot_ptr,
                                    const std::int16_t* __restrict__ slot_cols,
                                    const double* __restrict__ term_vals, int n_terms,
                                    int* __restrict__ segs, int* __restrict__ slots) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int n_seg = 0, n_slot = 0;
    for (int t = 0; t < n_terms; ++t) {
        const double* v = term_vals + static_cast<std::int64_t>(t) * nnz;
        int cnt = 0;
        for_each_entry(row, row_ptr, slot_ptr, slot_cols, [&](int p, int, bool is_far) {
            if (v[p] != 0.0) cnt += is_far ? 2 : 1;
        });
        n_seg += cnt > 0;
        n_slot += cnt;
    }
    segs[row] = n_seg;
    slots[row] = n_slot;
}

__global__ void affine_fill_kernel(int n, int nnz, const int* __restrict__ row_ptr,
                                   const int* __restrict__ slot_ptr,
                                   const std::int16_t* __restrict__ slot_cols,
                                   const double* __restrict__ term_vals, int n_terms,
                                   const int* __restrict__ seg_ptr,
                                   const int* __restrict__ slot_base, int* __restrict__ seg_start,
                                   std::uint8_t* __restrict__ seg_term,
                                   std::int16_t* __restrict__ cols, double* __restrict__ vals) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int g = seg_ptr[row];
    int o = slot_base[row];
    for (int t = 0; t < n_terms; ++t) {
        const double* v = term_vals + static_cast<std::int64_t>(t) * nnz;
        const int begin = o;
        for_each_entry(row, row_ptr, slot_ptr, slot_cols, [&](int p, int s, bool is_far) {
            if (v[p] == 0.0) return;
            cols[o] = slot_cols[s];
            vals[o++] = v[p];
            if (is_far) {
                cols[o] = slot_cols[s + 1];
                vals[o++] = 0.0;
            }
        });
        if (o > begin) {
            seg_term[g] = static_cast<std::uint8_t>(t);
            seg_start[g++] = begin;
        }
    }
}

__global__ void term_row_sums_kernel(int n_s, const int* __restrict__ rows,
                                     const int* __restrict__ row_ptr,
                                     const double* __restrict__ term_vals, int nnz, int n_terms,
                                     double* __restrict__ aw_t) {
    const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::int64_t>(n_s) * n_terms) return;
    const int term = static_cast<int>(t / n_s);
    const int i = static_cast<int>(t % n_s);
    const double* v = term_vals + static_cast<std::int64_t>(term) * nnz;
    double s = 0.0;
    for (int p = row_ptr[rows[i]]; p < row_ptr[rows[i] + 1]; ++p) s += v[p];
    aw_t[t] = s;
}

}  // namespace

SlotCsr build_slot_csr(int n, const int* row_ptr, const int* col_idx, cudaStream_t stream,
                       const char* origin) {
    if (static_cast<std::int64_t>(n) > kSlotMaxColumns) {
        throw std::invalid_argument("slot CSR: too many columns for a far pair to address");
    }
    SlotCsr out;
    out.ptr = device_alloc<int>(static_cast<std::size_t>(n) + 1, origin, "slot ptr");
    const ScratchBuffer<int> counts =
        scratch_alloc<int>(static_cast<std::size_t>(n) + 1, stream, origin, "slot counts");
    if (const unsigned blocks = grid_for(n)) {
        slot_count_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, counts.get());
    }
    out.n_slots = scan_counts(n, counts.get(), out.ptr.get(), stream, origin);
    out.cols = device_alloc<std::int16_t>(static_cast<std::size_t>(out.n_slots), origin,
                                          "slot cols");
    if (const unsigned blocks = grid_for(n)) {
        slot_fill_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, col_idx, out.ptr.get(),
                                                        out.cols.get());
    }
    check_cuda(cudaGetLastError(), origin, "slot fill");
    check_cuda(cudaStreamSynchronize(stream), origin, "slot fill");
    return out;
}

void launch_slot_scatter(int n, const int* row_ptr, const int* slot_ptr,
                         const std::int16_t* cols, const double* src, double* dst,
                         cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n)) {
        slot_scatter_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, slot_ptr, cols, src, dst);
    }
}

AffineSlots build_affine_slots(int n, int nnz, const int* row_ptr, const int* slot_ptr,
                               const std::int16_t* slot_cols, const double* term_vals,
                               int n_terms, cudaStream_t stream, const char* origin) {
    if (n_terms < 1 || n_terms > 255) {
        throw std::invalid_argument("affine operator: between 1 and 255 parts");
    }
    AffineSlots out;
    out.n_terms = n_terms;
    const std::size_t rows1 = static_cast<std::size_t>(n) + 1;
    const ScratchBuffer<int> segs = scratch_alloc<int>(rows1, stream, origin, "affine segs");
    const ScratchBuffer<int> slots = scratch_alloc<int>(rows1, stream, origin, "affine slots");
    if (const unsigned blocks = grid_for(n)) {
        affine_count_kernel<<<blocks, kBlock, 0, stream>>>(n, nnz, row_ptr, slot_ptr, slot_cols,
                                                           term_vals, n_terms, segs.get(),
                                                           slots.get());
    }
    out.seg_ptr = device_alloc<int>(rows1, origin, "affine seg_ptr");
    const ScratchBuffer<int> slot_base =
        scratch_alloc<int>(rows1, stream, origin, "affine slot base");
    out.n_segs = scan_counts(n, segs.get(), out.seg_ptr.get(), stream, origin);
    out.n_slots = scan_counts(n, slots.get(), slot_base.get(), stream, origin);
    out.seg_start =
        device_alloc<int>(static_cast<std::size_t>(out.n_segs) + 1, origin, "affine seg_start");
    out.seg_term =
        device_alloc<std::uint8_t>(static_cast<std::size_t>(out.n_segs), origin, "affine terms");
    out.cols = device_alloc<std::int16_t>(static_cast<std::size_t>(out.n_slots), origin,
                                          "affine cols");
    out.vals = device_alloc<double>(static_cast<std::size_t>(out.n_slots), origin, "affine vals");
    if (const unsigned blocks = grid_for(n)) {
        affine_fill_kernel<<<blocks, kBlock, 0, stream>>>(
            n, nnz, row_ptr, slot_ptr, slot_cols, term_vals, n_terms, out.seg_ptr.get(),
            slot_base.get(), out.seg_start.get(), out.seg_term.get(), out.cols.get(),
            out.vals.get());
    }
    // The last segment ends where the slots do.
    check_cuda(cudaMemcpyAsync(out.seg_start.get() + out.n_segs, &out.n_slots, sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               origin, "affine seg end");
    std::vector<int> h_slots(n), h_segs(n);
    check_cuda(cudaMemcpyAsync(h_slots.data(), slots.get(), n * sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               origin, "affine lengths");
    check_cuda(cudaMemcpyAsync(h_segs.data(), segs.get(), n * sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               origin, "affine segments");
    check_cuda(cudaStreamSynchronize(stream), origin, "affine lengths");
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    for (int lo = 0; lo < n; lo += kAffineWindow) {
        const int hi = std::min(n, lo + kAffineWindow);
        std::stable_sort(order.begin() + lo, order.begin() + hi, [&](int a, int b) {
            if (h_segs[a] != h_segs[b]) return h_segs[a] > h_segs[b];
            return h_slots[a] > h_slots[b];
        });
    }
    out.order = device_alloc<int>(static_cast<std::size_t>(n), origin, "affine order");
    check_cuda(cudaMemcpyAsync(out.order.get(), order.data(), n * sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               origin, "affine order");
    check_cuda(cudaGetLastError(), origin, "affine fill");
    check_cuda(cudaStreamSynchronize(stream), origin, "affine fill");
    return out;
}

void launch_slot_gather(int n, const int* row_ptr, const int* slot_ptr, const std::int16_t* cols,
                        const double* src, int* col_idx, double* values, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n)) {
        slot_gather_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, slot_ptr, cols, src,
                                                          col_idx, values);
    }
}

void launch_affine_values(int n, const int* row_ptr, const int* slot_ptr,
                          const std::int16_t* slot_cols, const AffineSlots& terms,
                          const double* sigma, double* out, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n)) {
        affine_values_kernel<<<blocks, kBlock, 0, stream>>>(
            n, row_ptr, slot_ptr, slot_cols, terms.seg_ptr.get(), terms.seg_start.get(),
            terms.seg_term.get(), terms.cols.get(), terms.vals.get(), sigma, out);
    }
}

void launch_affine_term_matmul(int n, const AffineSlots& terms, int term, const double* x,
                               double* y, int m, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n) * m)) {
        affine_term_matmul_kernel<<<blocks, kBlock, 0, stream>>>(
            n, m, term, terms.seg_ptr.get(), terms.seg_start.get(), terms.seg_term.get(),
            terms.cols.get(), terms.vals.get(), x, y);
    }
}

void launch_term_row_sums(int n_s, const int* rows, const int* row_ptr, const double* term_vals,
                          int nnz, int n_terms, double* aw_t, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n_s) * n_terms)) {
        term_row_sums_kernel<<<blocks, kBlock, 0, stream>>>(n_s, rows, row_ptr, term_vals, nnz,
                                                            n_terms, aw_t);
    }
}
