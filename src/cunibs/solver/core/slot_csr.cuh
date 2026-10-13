#pragma once
#include "core/common.hpp"

#include <cstdint>

// Column encoding for the CSRs the solve streams every iteration: the fp64 CG operator and every
// V-cycle level. A node's neighbours sit close to it in the spatially ordered numbering, so a
// column is stored as a 15-bit signed offset from its row, which covers 96% of a head mesh's
// nonzeros and makes the column stream half the bytes of an int32 index. At k = 1 the column
// index was half of each SpMV's DRAM traffic.
//
// The rest are the jumps a space-filling curve makes across its own cell boundaries. Such a
// column takes two consecutive slots, high half then low half, 14 bits each, and the second slot
// carries a zero value. The top two bits tag the three kinds, so every slot decodes on its own:
//     00, 11   near: the offset itself, in [-16384, 16383]
//     10       far, high half
//     01       far, low half
// A low-half slot decodes to its own row. Its value is zero, so the term it adds is a signed zero
// and no sum changes; what matters is that every kernel walks the same slots in the same lanes,
// which keeps the CG kernels' association, and so block-width invariance, intact.

constexpr int kSlotNearMin = -16384;
constexpr int kSlotNearMax = 16383;
constexpr int kSlotHalfBits = 14;
constexpr int kSlotHalfMask = (1 << kSlotHalfBits) - 1;
// Columns a far pair can address.
constexpr std::int64_t kSlotMaxColumns = std::int64_t{1} << (2 * kSlotHalfBits);

__device__ __forceinline__ int slot_column(const std::int16_t* __restrict__ cols, int j, int row) {
    const int c = cols[j];
    if (c > kSlotNearMax) return row;
    if (c >= kSlotNearMin) return row + c;
    return ((c & kSlotHalfMask) << kSlotHalfBits) | (cols[j + 1] & kSlotHalfMask);
}
