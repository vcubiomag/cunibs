#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// Host-side pieces every translation unit in this directory needs. Safe to include from
// .cpp as well as .cu; anything with __device__ code belongs in device_math.cuh instead.

constexpr int kBlock = 256;
constexpr int kWarp = 32;

static_assert(kBlock % kWarp == 0, "a block must be a whole number of warps");

// Block (k <= 8 placements per chunk) variants of the per-placement stages: the shared
// mesh arrays (tet_nodes, wg, node2corner, g) are read once per chunk rather than once
// per placement. Per-placement arrays stay separate contiguous buffers, passed as
// pointer packs, so the per-placement result layout is the same as the serial path's.
constexpr int kMaxStageBlock = 8;

inline void check_cuda(cudaError_t err, const char* origin, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(origin) + " CUDA error (" + what +
                                 "): " + cudaGetErrorString(err));
    }
}

// Rows times threads-per-row overflows int before gridDim.x runs out, so size the grid in
// 64 bits. An empty launch is a grid of zero blocks, which is a launch error, so callers
// test the result and skip.
inline unsigned grid_for(std::int64_t total, int per = kBlock) {
    if (total <= 0) return 0;
    return static_cast<unsigned>((total + per - 1) / per);
}

// Runtime k to a compile-time K. The allowed widths are spelled as template arguments so
// each caller's set is visible at the call site; `what` carries that caller's message.
template <int... Ks, typename F>
void dispatch_k(int k, const char* what, F&& f) {
    const bool matched = ((k == Ks ? (f(std::integral_constant<int, Ks>{}), true) : false) || ...);
    if (!matched) throw std::invalid_argument(what);
}
