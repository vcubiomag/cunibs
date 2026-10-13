#include "amg/slot_csr.hpp"
#include "core/slot_csr.cuh"

#include <cub/device/device_scan.cuh>

#include <stdexcept>

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

}  // namespace

SlotCsr build_slot_csr(int n, const int* row_ptr, const int* col_idx, const char* origin) {
    if (static_cast<std::int64_t>(n) > kSlotMaxColumns) {
        throw std::invalid_argument("slot CSR: too many columns for a far pair to address");
    }
    SlotCsr out;
    out.ptr = device_alloc<int>(static_cast<std::size_t>(n) + 1, origin, "slot ptr");
    DeviceBuffer<int> counts =
        device_alloc<int>(static_cast<std::size_t>(n) + 1, origin, "slot counts");
    check_cuda(cudaMemsetAsync(counts.get() + n, 0, sizeof(int), nullptr), origin, "slot counts");
    if (const unsigned blocks = grid_for(n)) {
        slot_count_kernel<<<blocks, kBlock>>>(n, row_ptr, col_idx, counts.get());
    }
    std::size_t bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts.get(), out.ptr.get(), n + 1),
               origin, "slot scan size");
    DeviceBuffer<unsigned char> temp = device_alloc<unsigned char>(bytes, origin, "slot scan");
    check_cuda(
        cub::DeviceScan::ExclusiveSum(temp.get(), bytes, counts.get(), out.ptr.get(), n + 1),
        origin, "slot scan");
    check_cuda(cudaMemcpy(&out.n_slots, out.ptr.get() + n, sizeof(int), cudaMemcpyDeviceToHost),
               origin, "slot count");
    out.cols = device_alloc<std::int16_t>(static_cast<std::size_t>(out.n_slots), origin,
                                          "slot cols");
    if (const unsigned blocks = grid_for(n)) {
        slot_fill_kernel<<<blocks, kBlock>>>(n, row_ptr, col_idx, out.ptr.get(), out.cols.get());
    }
    check_cuda(cudaGetLastError(), origin, "slot fill");
    check_cuda(cudaStreamSynchronize(nullptr), origin, "slot fill");
    return out;
}

void launch_slot_scatter(int n, const int* row_ptr, const int* slot_ptr,
                         const std::int16_t* cols, const double* src, double* dst,
                         cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n)) {
        slot_scatter_kernel<<<blocks, kBlock, 0, stream>>>(n, row_ptr, slot_ptr, cols, src, dst);
    }
}

