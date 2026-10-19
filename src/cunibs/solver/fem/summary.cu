// Volume-weighted summaries of |E| over one region, for up to kMaxStageBlock fields per call.
//
// A field's numbers come from the same arithmetic whatever else shares its call: the reductions
// are a fixed tree over a row mapping set by the region size alone, the sort is a stable radix
// sort, and the weight prefix sum is a fixed-tile scan. Decoupled-lookback scans and atomics are
// avoided because both associate float adds in an order that follows scheduling.
#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cuda/cmath>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kRowsPerThread = 8;
constexpr int kPerThread = 4;
constexpr int kTile = kBlock * kPerThread;
constexpr int kWarps = kBlock / kWarp;
// Sums carried by the first pass: weight, weight x field, and weight x field x barycentre.
constexpr int kSums = 5;

template <typename T>
struct Fields {
    const T* p[kMaxStageBlock];
};

struct Peak {
    double value;
    int index;
};

// Larger value wins and ties go to the lower index, so the result is the first maximum in region
// order however the partial results are combined. NaN beats every number, as in np.max, so a field
// carrying one reports the NaN.
__device__ __forceinline__ Peak better(Peak a, Peak b) {
    const bool a_nan = isnan(a.value);
    const bool b_nan = isnan(b.value);
    if (a_nan != b_nan) return b_nan ? b : a;
    const bool b_wins = a_nan || b.value == a.value ? b.index < a.index : b.value > a.value;
    return b_wins ? b : a;
}

__device__ __forceinline__ Peak shfl_down(Peak p, int off) {
    return Peak{__shfl_down_sync(kFullWarp, p.value, off),
                __shfl_down_sync(kFullWarp, p.index, off)};
}

int moment_blocks(int n) { return static_cast<int>(grid_for(n, kBlock * kRowsPerThread)); }
int tiles(int n) { return static_cast<int>(grid_for(n, kTile)); }

// Partials per (field, block): kSums sums, then the block's peak.
template <typename T>
__global__ void __launch_bounds__(kBlock)
    summary_moments_kernel(Fields<T> fields, int n, const int* __restrict__ index,
                           const double* __restrict__ w, const double* __restrict__ bary,
                           int n_blocks, double* __restrict__ sums, Peak* __restrict__ peaks) {
    const int c = blockIdx.y;
    const T* __restrict__ m = fields.p[c];
    double s[kSums] = {};
    Peak best{-CUDART_INF, n};
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < kRowsPerThread; ++t, i += stride) {
        if (i >= n) break;
        const double mi = static_cast<double>(m[index[i]]);
        const double wi = w[i];
        const double wm = wi * mi;
        s[0] += wi;
        s[1] += wm;
        s[2] += wm * bary[3 * static_cast<std::int64_t>(i)];
        s[3] += wm * bary[3 * static_cast<std::int64_t>(i) + 1];
        s[4] += wm * bary[3 * static_cast<std::int64_t>(i) + 2];
        best = better(best, Peak{mi, i});
    }
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int k = 0; k < kSums; ++k) s[k] += __shfl_down_sync(kFullWarp, s[k], off);
        best = better(best, shfl_down(best, off));
    }
    __shared__ double ws[kWarps][kSums];
    __shared__ Peak wp[kWarps];
    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < kSums; ++k) ws[warp][k] = s[k];
        wp[warp] = best;
    }
    __syncthreads();
    if (threadIdx.x < kSums) {
        double v = 0.0;
        for (int q = 0; q < kWarps; ++q) v += ws[q][threadIdx.x];
        sums[(static_cast<std::int64_t>(c) * kSums + threadIdx.x) * n_blocks + blockIdx.x] = v;
    }
    if (threadIdx.x == 0) {
        Peak b = wp[0];
        for (int q = 1; q < kWarps; ++q) b = better(b, wp[q]);
        peaks[static_cast<std::int64_t>(c) * n_blocks + blockIdx.x] = b;
    }
}

// Every thread of the block must reach this; the total comes back to all of them.
__device__ double block_total(const double* __restrict__ part, int n_blocks) {
    __shared__ double sdata[kBlock];
    double v = 0.0;
    for (int b = threadIdx.x; b < n_blocks; b += kBlock) v += part[b];
    const double total = block_tree_sum(v, sdata);
    // Callers sum again into the same array.
    __syncthreads();
    return total;
}

// One block per field: peak and its location, volume, mean and centre of gravity.
__global__ void __launch_bounds__(kBlock)
    summary_moments_finish_kernel(const double* __restrict__ sums, const Peak* __restrict__ peaks,
                                  int n_blocks, const double* __restrict__ bary, int n_out,
                                  double* __restrict__ out, double* __restrict__ scratch) {
    const int c = blockIdx.x;
    double t[kSums];
    for (int k = 0; k < kSums; ++k) {
        t[k] = block_total(sums + (static_cast<std::int64_t>(c) * kSums + k) * n_blocks, n_blocks);
    }
    if (threadIdx.x != 0) return;
    Peak best = peaks[static_cast<std::int64_t>(c) * n_blocks];
    for (int b = 1; b < n_blocks; ++b) {
        best = better(best, peaks[static_cast<std::int64_t>(c) * n_blocks + b]);
    }
    double* o = out + static_cast<std::int64_t>(c) * n_out;
    o[kSummaryPeak] = best.value;
    for (int a = 0; a < 3; ++a) {
        o[kSummaryPeakLocation + a] = bary[3 * static_cast<std::int64_t>(best.index) + a];
        o[kSummaryCenterOfGravity + a] = t[2 + a] / t[1];
    }
    const double mean = t[1] / t[0];
    o[kSummaryMean] = mean;
    scratch[2 * c] = mean;
    scratch[2 * c + 1] = t[0];
}

template <typename T>
__global__ void __launch_bounds__(kBlock)
    summary_spread_kernel(Fields<T> fields, int n, const int* __restrict__ index,
                          const double* __restrict__ w, const double* __restrict__ scratch,
                          double* __restrict__ part, int n_blocks) {
    const int c = blockIdx.y;
    const T* __restrict__ m = fields.p[c];
    const double mean = scratch[2 * c];
    double s = 0.0;
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < kRowsPerThread; ++t, i += stride) {
        if (i >= n) break;
        const double d = static_cast<double>(m[index[i]]) - mean;
        s += w[i] * (d * d);
    }
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1) s += __shfl_down_sync(kFullWarp, s, off);
    __shared__ double ws[kWarps];
    if (threadIdx.x % kWarp == 0) ws[threadIdx.x / kWarp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        double v = 0.0;
        for (int q = 0; q < kWarps; ++q) v += ws[q];
        part[static_cast<std::int64_t>(c) * n_blocks + blockIdx.x] = v;
    }
}

__global__ void __launch_bounds__(kBlock)
    summary_spread_finish_kernel(const double* __restrict__ part, int n_blocks,
                                 const double* __restrict__ scratch, int n_out,
                                 double* __restrict__ out) {
    const int c = blockIdx.x;
    const double total = block_total(part + static_cast<std::int64_t>(c) * n_blocks, n_blocks);
    if (threadIdx.x == 0) {
        out[static_cast<std::int64_t>(c) * n_out + kSummaryStd] =
            sqrt(total / scratch[2 * c + 1]);
    }
}

template <typename T>
__global__ void summary_gather_kernel(const T* __restrict__ m, const int* __restrict__ index,
                                      int n, T* __restrict__ keys) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) keys[i] = m[index[i]];
}

// Hillis-Steele inclusive scan of one value per thread across the block, whose data flow does not
// depend on scheduling. Every thread of the block must reach it.
__device__ double block_inclusive_scan(double v) {
    __shared__ double buf[2][kBlock];
    int cur = 0;
    buf[cur][threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < kBlock; off <<= 1) {
        const double t = buf[cur][threadIdx.x] +
                         (threadIdx.x >= off ? buf[cur][threadIdx.x - off] : 0.0);
        buf[cur ^ 1][threadIdx.x] = t;
        cur ^= 1;
        __syncthreads();
    }
    return buf[cur][threadIdx.x];
}

// Inclusive scan of one tile in a fixed order: kPerThread consecutive entries per thread in
// sequence, then a block scan of the thread totals. within[i] is the scan inside the tile;
// tile_total[t] the tile's sum.
__global__ void __launch_bounds__(kBlock)
    summary_tile_scan_kernel(const double* __restrict__ w, int n, double* __restrict__ within,
                             double* __restrict__ tile_total) {
    const std::int64_t base =
        static_cast<std::int64_t>(blockIdx.x) * kTile + static_cast<std::int64_t>(threadIdx.x) *
                                                            kPerThread;
    double local[kPerThread];
    double run = 0.0;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const std::int64_t i = base + j;
        run += (i < n) ? w[i] : 0.0;
        local[j] = run;
    }
    const double inclusive = block_inclusive_scan(run);
    const double offset = inclusive - run;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const std::int64_t i = base + j;
        if (i < n) within[i] = offset + local[j];
    }
    if (threadIdx.x == kBlock - 1) tile_total[blockIdx.x] = inclusive;
}

// Exclusive scan of the tile totals in one block: a fixed chunk per thread, then the same block
// scan over the chunk sums.
__global__ void __launch_bounds__(kBlock)
    summary_tile_offsets_kernel(const double* __restrict__ tile_total, int n_tiles,
                                double* __restrict__ tile_offset) {
    const int chunk = cuda::ceil_div(n_tiles, kBlock);
    const int lo = threadIdx.x * chunk;
    const int hi = min(n_tiles, lo + chunk);
    double run = 0.0;
    for (int t = lo; t < hi; ++t) run += tile_total[t];
    double acc = block_inclusive_scan(run) - run;
    for (int t = lo; t < hi; ++t) {
        tile_offset[t] = acc;
        acc += tile_total[t];
    }
}

struct Probes {
    double q[kMaxSummaryQuantiles];
    double frac[kMaxSummaryQuantiles];
    int n_q;
    int n_f;
    int anchor;
};

// The largest j in [-1, n) with pred(j), for pred true on a prefix of [0, n). The warp probes 32
// evenly spaced indices per step; all lanes hold the same bounds throughout.
template <typename Pred>
__device__ int warp_last_true(int n, Pred pred) {
    int lo = -1;
    int hi = n;
    const int lane = threadIdx.x % kWarp;
    while (hi - lo > 1) {
        const std::int64_t span = hi - lo - 1;
        const int probe = lo + 1 + static_cast<int>((span * lane) / kWarp);
        const unsigned truth = __ballot_sync(kFullWarp, pred(probe));
        const int n_true = __popc(truth);
        const int new_lo = (n_true > 0) ? __shfl_sync(kFullWarp, probe, n_true - 1) : lo;
        const int new_hi = (n_true < kWarp) ? __shfl_sync(kFullWarp, probe, n_true) : hi;
        lo = new_lo;
        hi = new_hi;
    }
    return lo;
}

// Weighted quantiles over midpoint positions pos_i = (cw_i - w_i / 2) / W of the sorted field,
// interpolated as numpy.interp does, then the volume at or above frac x the anchor quantile. One
// warp per probe.
template <typename T>
__global__ void summary_quantiles_kernel(const T* __restrict__ keys, const double* __restrict__ w,
                                         const double* __restrict__ within,
                                         const double* __restrict__ tile_offset, int n,
                                         Probes probes, double* __restrict__ o) {
    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const auto cw = [&](int i) { return tile_offset[i / kTile] + within[i]; };
    const double total = cw(n - 1);
    const auto pos = [&](int i) { return (cw(i) - 0.5 * w[i]) / total; };
    __shared__ double anchor;
    if (warp < probes.n_q) {
        const double x = probes.q[warp];
        const int j = warp_last_true(n, [&](int i) { return pos(i) <= x; });
        double v;
        if (j < 0) {
            v = static_cast<double>(keys[0]);
        } else if (j == n - 1) {
            v = static_cast<double>(keys[n - 1]);
        } else {
            const double pj = pos(j);
            const double kj = static_cast<double>(keys[j]);
            if (pj == x) {
                v = kj;
            } else {
                const double slope = (static_cast<double>(keys[j + 1]) - kj) / (pos(j + 1) - pj);
                v = slope * (x - pj) + kj;
            }
        }
        if (lane == 0) {
            o[kSummaryQuantiles + warp] = v;
            if (warp == probes.anchor) anchor = v;
        }
    }
    __syncthreads();
    if (warp < probes.n_f) {
        // The threshold is narrowed to the field's precision before the comparison, which is what
        // comparing a T array against a Python float does.
        const T threshold = static_cast<T>(probes.frac[warp] * anchor);
        const int j = warp_last_true(n, [&](int i) { return keys[i] < threshold; });
        if (lane == 0) o[kSummaryQuantiles + probes.n_q + warp] = (j < 0) ? total : total - cw(j);
    }
}

std::size_t aligned(std::size_t bytes) { return cuda::round_up(bytes, std::size_t{256}); }

template <typename T>
std::size_t sort_bytes(int n) {
    std::size_t bytes = 0;
    check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const T*>(nullptr),
                                               static_cast<T*>(nullptr),
                                               static_cast<const double*>(nullptr),
                                               static_cast<double*>(nullptr), n),
               "summary", "sort size");
    return bytes;
}

template <typename T>
std::size_t work_size(int n, int k) {
    const std::size_t nb = moment_blocks(n);
    const std::size_t nt = tiles(n);
    return 2 * aligned(n * sizeof(T)) + 2 * aligned(n * sizeof(double)) +
           2 * aligned(nt * sizeof(double)) + aligned(k * kSums * nb * sizeof(double)) +
           aligned(k * nb * sizeof(Peak)) + aligned(k * nb * sizeof(double)) +
           aligned(2 * k * sizeof(double)) + aligned(sort_bytes<T>(n));
}

template <typename T>
void summarize(const void* const* fields, int k, int n, const int* index, const double* w,
               const double* bary, const Probes& probes, void* work, std::size_t work_bytes,
               double* out, int n_out, cudaStream_t stream) {
    if (work_bytes < work_size<T>(n, k)) {
        throw std::invalid_argument("region_summary: workspace too small");
    }
    Fields<T> f{};
    for (int c = 0; c < k; ++c) f.p[c] = static_cast<const T*>(fields[c]);
    const int nb = moment_blocks(n);
    const int nt = tiles(n);
    auto* cursor = static_cast<unsigned char*>(work);
    const auto take = [&](std::size_t bytes) {
        void* p = cursor;
        cursor += aligned(bytes);
        return p;
    };
    T* keys = static_cast<T*>(take(n * sizeof(T)));
    T* keys_sorted = static_cast<T*>(take(n * sizeof(T)));
    double* w_sorted = static_cast<double*>(take(n * sizeof(double)));
    double* within = static_cast<double*>(take(n * sizeof(double)));
    double* tile_total = static_cast<double*>(take(nt * sizeof(double)));
    double* tile_offset = static_cast<double*>(take(nt * sizeof(double)));
    double* sums = static_cast<double*>(take(static_cast<std::size_t>(k) * kSums * nb * 8));
    Peak* peaks = static_cast<Peak*>(take(static_cast<std::size_t>(k) * nb * sizeof(Peak)));
    double* spread = static_cast<double*>(take(static_cast<std::size_t>(k) * nb * 8));
    double* scratch = static_cast<double*>(take(2 * static_cast<std::size_t>(k) * 8));
    std::size_t sort_temp_bytes = sort_bytes<T>(n);
    void* sort_temp = take(sort_temp_bytes);

    const dim3 grid(nb, k);
    summary_moments_kernel<T><<<grid, kBlock, 0, stream>>>(f, n, index, w, bary, nb, sums, peaks);
    summary_moments_finish_kernel<<<k, kBlock, 0, stream>>>(sums, peaks, nb, bary, n_out, out,
                                                            scratch);
    summary_spread_kernel<T><<<grid, kBlock, 0, stream>>>(f, n, index, w, scratch, spread, nb);
    summary_spread_finish_kernel<<<k, kBlock, 0, stream>>>(spread, nb, scratch, n_out, out);
    for (int c = 0; c < k; ++c) {
        summary_gather_kernel<T><<<grid_for(n), kBlock, 0, stream>>>(f.p[c], index, n, keys);
        check_cuda(cub::DeviceRadixSort::SortPairs(sort_temp, sort_temp_bytes, keys, keys_sorted,
                                                   w, w_sorted, n, 0, sizeof(T) * 8, stream),
                   "summary", "sort");
        summary_tile_scan_kernel<<<nt, kBlock, 0, stream>>>(w_sorted, n, within, tile_total);
        summary_tile_offsets_kernel<<<1, kBlock, 0, stream>>>(tile_total, nt, tile_offset);
        const int warps = probes.n_q > probes.n_f ? probes.n_q : probes.n_f;
        summary_quantiles_kernel<T><<<1, warps * kWarp, 0, stream>>>(
            keys_sorted, w_sorted, within, tile_offset, n, probes,
            out + static_cast<std::int64_t>(c) * n_out);
    }
    check_cuda(cudaGetLastError(), "summary", "launch");
}

}  // namespace

std::size_t region_summary_work_size(int n, int k, bool f64) {
    return f64 ? work_size<double>(n, k) : work_size<float>(n, k);
}

void launch_region_summary(const void* const* fields, bool f64, int k, int n, const int* index,
                           const double* w, const double* bary, const double* qs, int n_q,
                           int anchor, const double* fracs, int n_f, void* work,
                           std::size_t work_bytes, double* out, cudaStream_t stream) {
    if (k < 1 || k > kMaxStageBlock) throw std::invalid_argument("region_summary: bad k");
    if (n < 1) throw std::invalid_argument("region_summary: the region holds no elements");
    if (n_q < 1 || n_q > kMaxSummaryQuantiles || n_f < 0 || n_f > kMaxSummaryQuantiles ||
        anchor < 0 || anchor >= n_q) {
        throw std::invalid_argument("region_summary: bad quantiles");
    }
    Probes probes{};
    for (int i = 0; i < n_q; ++i) probes.q[i] = qs[i];
    for (int i = 0; i < n_f; ++i) probes.frac[i] = fracs[i];
    probes.n_q = n_q;
    probes.n_f = n_f;
    probes.anchor = anchor;
    const int n_out = kSummaryQuantiles + n_q + n_f;
    if (f64) {
        summarize<double>(fields, k, n, index, w, bary, probes, work, work_bytes, out, n_out,
                          stream);
    } else {
        summarize<float>(fields, k, n, index, w, bary, probes, work, work_bytes, out, n_out,
                         stream);
    }
}
