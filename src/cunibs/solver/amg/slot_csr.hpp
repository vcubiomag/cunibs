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

