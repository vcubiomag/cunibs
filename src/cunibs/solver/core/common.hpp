#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// Host-side pieces every translation unit in this directory needs. Safe to include from
// .cpp as well as .cu; anything with __device__ code belongs in device_math.cuh instead.

// A reduction whose association order is unspecified cannot stand in for one of these kernels:
// results are required to be bit-reproducible, so every summation tree is written out here.
// Library calls are confined to setup (cub::DeviceScan, cub::DeviceSegmentedSort) and to
// place.cu's minimum over a total order, which no order of combination can change.

constexpr int kBlock = 256;
constexpr int kWarp = 32;

// Every lane of a warp. No shuffle or vote in this directory is reached by a divergent subset of
// one, so this is the mask all of them carry.
constexpr unsigned kFullWarp = 0xffffffffu;

static_assert(kBlock % kWarp == 0, "a block must be a whole number of warps");

// Block (k <= 8 placements per chunk) variants of the per-placement stages: the shared
// mesh arrays (tet_nodes, wg, node2corner, g) are read once per chunk rather than once
// per placement. Per-placement arrays stay separate contiguous buffers, passed as
// pointer packs, so the per-placement result layout is the same as the serial path's.
constexpr int kMaxStageBlock = 8;

// Compiled widths of the k-RHS CG and V-cycle kernels, increasing. The dispatch_k call sites
// in block_cg.cu and vcycle.cu enumerate the same set as template arguments; this is what the
// bindings export, so the Python layer derives its BLOCK_SIZES rather than restating it.
inline constexpr int kBlockWidths[] = {1, 2, 4, 8};
inline constexpr int kNumBlockWidths = 4;

static_assert(kBlockWidths[kNumBlockWidths - 1] <= kMaxStageBlock,
              "a solve width wider than a staging chunk could never be fed");

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

// Multiprocessor count of the current device, cached because nothing here changes it. Callers use
// it only to decide how far to split work for occupancy, so a failed query falls back to one SM
// and costs throughput, nothing more.
inline int sm_count() {
    static const int count = [] {
        int device = 0, sms = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return 1;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            return 1;
        }
        return sms > 0 ? sms : 1;
    }();
    return count;
}

// How many blocks to split one query's brute-force scan of a triangle surface across. Such a scan
// runs a block per (query, chunk) and reduces the chunks afterwards under an order-independent
// rule, so this trades occupancy against reduction overhead and never changes what it returns.
//
// Split only while the queries alone leave SMs idle, and never past a block per
// kMinTrisPerChunk triangles, below which the block reduction outweighs the scan it covers. A
// surface with no triangles still gets one chunk, so that every query is written its miss.
inline int scan_chunks(int n_queries, int n_tri) {
    constexpr int kMinTrisPerChunk = 1024;
    if (n_queries <= 0 || n_tri <= 0) return 1;
    // Both round up without forming `a + b - 1`, whose overflow on a count near INT_MAX would be
    // undefined. cuda::ceil_div avoids it by dividing unsigned; spelling it out instead keeps this
    // header to the CUDA runtime, which is what lets the .cpp translation units include it.
    const int sms = sm_count();
    const int for_occupancy = sms / n_queries + (sms % n_queries != 0);
    const int for_size = n_tri / kMinTrisPerChunk + (n_tri % kMinTrisPerChunk != 0);
    return std::max(1, std::min(for_occupancy, for_size));
}

// Runtime k to a compile-time K. The allowed widths are spelled as template arguments so
// each caller's set is visible at the call site; `what` carries that caller's message.
template <int... Ks, typename F>
void dispatch_k(int k, const char* what, F&& f) {
    const bool matched = ((k == Ks ? (f(std::integral_constant<int, Ks>{}), true) : false) || ...);
    if (!matched) throw std::invalid_argument(what);
}

// Move-only owners for the four CUDA resources this directory allocates. Setup takes a dozen
// of them in a row, so wrapping them keeps a failure part way through from leaking the
// earlier ones and keeps every destructor implicit.
//
// All four release without checking: a free during interpreter teardown, after the context is
// gone, reports an error there is nothing useful to do with.
template <typename T, typename Deleter>
class Owner {
public:
    Owner() = default;
    explicit Owner(T handle) noexcept : handle_(handle) {}
    ~Owner() { reset(); }

    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    Owner(Owner&& other) noexcept : handle_(other.release()) {}
    Owner& operator=(Owner&& other) noexcept {
        if (this != &other) reset(other.release());
        return *this;
    }

    T get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != T{}; }
    T release() noexcept { return std::exchange(handle_, T{}); }

    void reset(T handle = T{}) noexcept {
        if (handle_ != T{}) Deleter{}(handle_);
        handle_ = handle;
    }

private:
    T handle_{};
};

namespace detail {
struct FreeDevice {
    void operator()(void* p) const noexcept { cudaFree(p); }
};
struct FreeHost {
    void operator()(void* p) const noexcept { cudaFreeHost(p); }
};
struct DestroyStream {
    void operator()(cudaStream_t s) const noexcept { cudaStreamDestroy(s); }
};
struct DestroyEvent {
    void operator()(cudaEvent_t e) const noexcept { cudaEventDestroy(e); }
};
}  // namespace detail

template <typename T>
using DeviceBuffer = Owner<T*, detail::FreeDevice>;
template <typename T>
using PinnedBuffer = Owner<T*, detail::FreeHost>;
using CudaStream = Owner<cudaStream_t, detail::DestroyStream>;
using CudaEvent = Owner<cudaEvent_t, detail::DestroyEvent>;

template <typename T>
DeviceBuffer<T> device_alloc(std::size_t count, const char* origin, const char* what) {
    T* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, count * sizeof(T)), origin, what);
    return DeviceBuffer<T>(ptr);
}

// Ordered on `stream`; the caller synchronises it before releasing src.
template <typename T>
DeviceBuffer<T> device_clone(const T* src, std::size_t count, cudaStream_t stream,
                             const char* origin, const char* what) {
    DeviceBuffer<T> buf = device_alloc<T>(count, origin, what);
    check_cuda(cudaMemcpyAsync(buf.get(), src, count * sizeof(T), cudaMemcpyDeviceToDevice,
                               stream),
               origin, what);
    return buf;
}

template <typename T>
PinnedBuffer<T> pinned_alloc(std::size_t count, const char* origin, const char* what) {
    T* ptr = nullptr;
    check_cuda(cudaMallocHost(&ptr, count * sizeof(T)), origin, what);
    return PinnedBuffer<T>(ptr);
}
