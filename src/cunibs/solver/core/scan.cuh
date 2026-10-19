#pragma once
#include "core/common.hpp"

#include <cub/device/device_scan.cuh>

#include <cstddef>

// Exclusive scan of counts[0, n) into the (n + 1)-entry ptr, returning the total, which is the
// length the caller allocates the next array at. counts holds n + 1 entries, the last of which
// this zeroes. Synchronises the stream to read the total.
inline int scan_counts(int n, int* counts, int* ptr, cudaStream_t stream, const char* origin) {
    check_cuda(cudaMemsetAsync(counts + n, 0, sizeof(int), stream), origin, "scan tail");
    std::size_t bytes = 0;
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts, ptr, n + 1, stream), origin,
               "scan size");
    const ScratchBuffer<std::byte> temp = scratch_alloc<std::byte>(bytes, stream, origin, "scan");
    check_cuda(cub::DeviceScan::ExclusiveSum(temp.get(), bytes, counts, ptr, n + 1, stream),
               origin, "scan");
    int total = 0;
    check_cuda(cudaMemcpyAsync(&total, ptr + n, sizeof(int), cudaMemcpyDeviceToHost, stream),
               origin, "scan total");
    check_cuda(cudaStreamSynchronize(stream), origin, "scan total");
    return total;
}
