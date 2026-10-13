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

// Setup only: synchronises the legacy stream it runs on, so the caller may drop its arrays.
SlotCsr build_slot_csr(int n, const int* row_ptr, const int* col_idx, const char* origin);

// dst (n_slots) = src (row_ptr's entries) laid out in slots, with a zero in every low-half slot.
void launch_slot_scatter(int n, const int* row_ptr, const int* slot_ptr,
                         const std::int16_t* cols, const double* src, double* dst,
                         cudaStream_t stream);

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
// slot_ptr/slot_cols. Synchronises the legacy stream it runs on.
AffineSlots build_affine_slots(int n, int nnz, const int* row_ptr, const int* slot_ptr,
                               const std::int16_t* slot_cols, const double* term_vals,
                               int n_terms, const char* origin);

// aw_t[t * n_s + s] = the sum of part t's entries in row rows[s], in entry order.
void launch_term_row_sums(int n_s, const int* rows, const int* row_ptr, const double* term_vals,
                          int nnz, int n_terms, double* aw_t, cudaStream_t stream);
