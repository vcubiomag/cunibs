#pragma once
#include <cstdint>

#include "core/common.hpp"

// A CSR pattern re-encoded into slots (see core/slot_csr.cuh). ptr indexes slots, not entries:
// a row's slot count is its nonzero count plus one per far column.
struct SlotCsr {
    DeviceBuffer<int> ptr;
    DeviceBuffer<std::int16_t> cols;
    int n_slots = 0;
};

// Setup only: runs on `stream` and synchronises it, so the caller may drop its arrays.
SlotCsr build_slot_csr(int n, const int* row_ptr, const int* col_idx, cudaStream_t stream,
                       const char* origin);

// dst (n_slots) = src (row_ptr's entries) laid out in slots, with a zero in every low-half slot.
void launch_slot_scatter(int n, const int* row_ptr, const int* slot_ptr,
                         const std::int16_t* cols, const double* src, double* dst,
                         cudaStream_t stream);

// col_idx and values (row_ptr's entries) from the slot form, either of which may be null: the
// inverse of the fill and of launch_slot_scatter.
void launch_slot_gather(int n, const int* row_ptr, const int* slot_ptr, const std::int16_t* cols,
                        const double* src, int* col_idx, double* values, cudaStream_t stream);

// An operator affine in per-column coefficients, laid out for the affine SpMV (cg_kernels.hpp):
// every row's entries regrouped into one segment per part that has a nonzero there, each in
// column order and in slots.
struct AffineSlots {
    // Rows in the order the SpMV takes them: each window of kAffineWindow consecutive rows grouped
    // by segment count, then longest first, so rows that share a warp walk near-equal lengths.
    DeviceBuffer<int> order;
    DeviceBuffer<int> seg_ptr;
    DeviceBuffer<int> seg_start;
    DeviceBuffer<std::uint8_t> seg_term;
    DeviceBuffer<std::int16_t> cols;
    DeviceBuffer<double> vals;
    int n_terms = 0;
    int n_segs = 0;
    int n_slots = 0;
};

constexpr int kAffineWindow = 32;

// Setup only. term_vals is (n_terms, nnz) over the entry CSR row_ptr, whose slot form is
// slot_ptr/slot_cols. Runs on `stream` and synchronises it.
AffineSlots build_affine_slots(int n, int nnz, const int* row_ptr, const int* slot_ptr,
                               const std::int16_t* slot_cols, const double* term_vals,
                               int n_terms, cudaStream_t stream, const char* origin);

// aw_t[t * n_s + s] = the sum of part t's entries in row rows[s], in entry order.
void launch_term_row_sums(int n_s, const int* rows, const int* row_ptr, const double* term_vals,
                          int nnz, int n_terms, double* aw_t, cudaStream_t stream);

// out (row_ptr's entries) = sum_t sigma[t] A_t, sigma a device array of terms.n_terms.
void launch_affine_values(int n, const int* row_ptr, const int* slot_ptr,
                          const std::int16_t* slot_cols, const AffineSlots& terms,
                          const double* sigma, double* out, cudaStream_t stream);

// y (n, m) = A_t x for one part t and row-major (n, m) x.
void launch_affine_term_matmul(int n, const AffineSlots& terms, int term, const double* x,
                               double* y, int m, cudaStream_t stream);
