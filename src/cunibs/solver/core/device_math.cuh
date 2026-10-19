#pragma once
#include "core/common.hpp"

#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/mdspan>

// __device__ helpers shared across kernels. .cu only. Keep this out of dadt.cu: that
// translation unit alone compiles with --use_fast_math, which would give these fp64
// bodies different contraction rules than every other caller gets.

// The K-wide accumulators these helpers move are cuda::std::array. Index them only with
// compile-time constants from unrolled loops; a runtime index spills one to local memory.

// Views over the flat arrays this directory passes between kernels. Trailing extents are static
// because they are properties of the mesh rather than of the problem size: a tetrahedron has four
// corners and a field vector three components.
//
// Index is int by default. Spell it std::int64_t wherever the row count times the trailing
// extents can outgrow an int, which on a head mesh happens well before the counts themselves do.
template <typename T, typename Index = int>
using Vec3View = cuda::std::mdspan<T, cuda::std::extents<Index, cuda::std::dynamic_extent, 3>>;
template <typename T, typename Index = int>
using Tet4View = cuda::std::mdspan<T, cuda::std::extents<Index, cuda::std::dynamic_extent, 4>>;
// (n_tri, 3) surface connectivity. Same shape as a Vec3View, spelled apart from it because its
// entries are node ids rather than components of one vector.
template <typename T, typename Index = int>
using Tri3View = cuda::std::mdspan<T, cuda::std::extents<Index, cuda::std::dynamic_extent, 3>>;
// (n_tri, 3, 3): the three corner coordinates of each triangle of a surface.
template <typename T, typename Index = int>
using TriVertView =
    cuda::std::mdspan<T, cuda::std::extents<Index, cuda::std::dynamic_extent, 3, 3>>;
// (n_tet, 4, 3): the P1 basis gradient of corner i of tetrahedron e. Also addressable per
// corner c = 4e + i through a Vec3View, which is how the corner-centric kernels read it.
template <typename T, typename Index = int>
using TetGradView =
    cuda::std::mdspan<T, cuda::std::extents<Index, cuda::std::dynamic_extent, 4, 3>>;
// Dense row-major with both extents known only at run time; the (n, k) block operands are these.
template <typename T>
using Mat2View = cuda::std::mdspan<T, cuda::std::dextents<std::int64_t, 2>>;
// (n, K) row-major with K a compiled width. The CG and V-cycle operands are these.
template <typename T, int K>
using WidthView =
    cuda::std::mdspan<T, cuda::std::extents<std::int64_t, cuda::std::dynamic_extent, K>>;

// A view for a kernel that was never handed its row count: it reaches rows through an incidence
// list that only names valid ones. The leading extent is the largest the index type can describe
// beside the trailing extents, so every row such a list can name is in range.
template <typename View>
__host__ __device__ constexpr View unsized(typename View::data_handle_type p) {
    using Index = typename View::index_type;
    using Extents = typename View::extents_type;
    Index inner = 1;
    for (cuda::std::size_t r = 1; r < Extents::rank(); ++r) inner *= Extents::static_extent(r);
    return View(p, cuda::std::numeric_limits<Index>::max() / inner);
}

// Three components passed around by value; the geometry kernels build and fold these, the helpers
// below take plain pointers.
using Vec3 = cuda::std::array<double, 3>;

__device__ __forceinline__ double dot3(const double* u, const double* v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

// out = u x v. `out` must not alias either input: every component is written before the last is
// read, which is why the callers that overwrite an operand stage the result first.
__device__ __forceinline__ void cross3(const double* u, const double* v, double* out) {
    out[0] = u[1] * v[2] - u[2] * v[1];
    out[1] = u[2] * v[0] - u[0] * v[2];
    out[2] = u[0] * v[1] - u[1] * v[0];
}

// Per-placement pointers for the block stages, passed by value as a kernel argument.
using ConstPtrPack = cuda::std::array<const float*, kMaxStageBlock>;
using PtrPack = cuda::std::array<float*, kMaxStageBlock>;

// Host side of the same: a launcher fills the first k slots from its caller's array and leaves the
// rest null, since a kernel only ever reads the columns it was given a k for.
template <typename T>
cuda::std::array<T*, kMaxStageBlock> ptr_pack(T* const* src, int k) {
    cuda::std::array<T*, kMaxStageBlock> out{};
    for (int c = 0; c < k; ++c) out[c] = src[c];
    return out;
}

// Fixed-order shuffle reduction of K per-lane accumulators into lane 0 of each WIDTH-wide group.
//
// The pairing is load-bearing: block_cg.cu's two SpMV kernels land on the same association, and a
// V-cycle level takes its threads-per-row from the operator alone, so a placement's field is the
// same whatever block width it was solved at.
//
// The mask is the full warp, not the group: threads whose row is past the end of the matrix still
// have to reach the shuffle, carrying zero accumulators.
template <int WIDTH, typename T, cuda::std::size_t K>
__device__ __forceinline__ void warp_reduce_sum(cuda::std::array<T, K>& sum) {
    static_assert(WIDTH > 0 && WIDTH <= kWarp && kWarp % WIDTH == 0,
                  "a shuffle group must be a power-of-two fraction of the warp");
#pragma unroll
    for (int off = WIDTH / 2; off > 0; off >>= 1) {
#pragma unroll
        for (cuda::std::size_t c = 0; c < K; ++c) {
            sum[c] += __shfl_down_sync(kFullWarp, sum[c], off, WIDTH);
        }
    }
}

// Move the K contiguous values of one row of a (n, K) row-major operand as vector accesses.
//
// This is what keeps a wide block off the L1 request limit: scalar accesses ask for K separate
// slices per thread, so a warp generates up to 32 distinct wavefronts for data occupying a
// handful of sectors and the kernel stalls on request throughput long before DRAM. K is a power
// of two and every operand starts at an allocation boundary, so a row's base is aligned to the
// width being moved. The values and their order are unaffected.
//
// store_row deliberately takes a plain pointer: the V-cycle's smoother writes a buffer it also
// reads at the row being written, and __restrict__ there would be a promise that call pattern
// breaks.

template <int K>
__device__ __forceinline__ void load_row(const float* __restrict__ p,
                                         cuda::std::array<float, K>& v) {
    if constexpr (K % 4 == 0) {
#pragma unroll
        for (int b = 0; b < K / 4; ++b) {
            const float4 t = *reinterpret_cast<const float4*>(p + b * 4);
            v[b * 4 + 0] = t.x;
            v[b * 4 + 1] = t.y;
            v[b * 4 + 2] = t.z;
            v[b * 4 + 3] = t.w;
        }
    } else if constexpr (K % 2 == 0) {
#pragma unroll
        for (int b = 0; b < K / 2; ++b) {
            const float2 t = *reinterpret_cast<const float2*>(p + b * 2);
            v[b * 2 + 0] = t.x;
            v[b * 2 + 1] = t.y;
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) v[c] = __ldg(p + c);
    }
}

template <int K>
__device__ __forceinline__ void load_row(const double* __restrict__ p,
                                         cuda::std::array<double, K>& v) {
    if constexpr (K % 2 == 0) {
#pragma unroll
        for (int b = 0; b < K / 2; ++b) {
            const double2 t = *reinterpret_cast<const double2*>(p + b * 2);
            v[b * 2 + 0] = t.x;
            v[b * 2 + 1] = t.y;
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) v[c] = __ldg(p + c);
    }
}

template <int K>
__device__ __forceinline__ void store_row(float* p, const cuda::std::array<float, K>& v) {
    if constexpr (K % 4 == 0) {
#pragma unroll
        for (int b = 0; b < K / 4; ++b) {
            *reinterpret_cast<float4*>(p + b * 4) =
                make_float4(v[b * 4 + 0], v[b * 4 + 1], v[b * 4 + 2], v[b * 4 + 3]);
        }
    } else if constexpr (K % 2 == 0) {
#pragma unroll
        for (int b = 0; b < K / 2; ++b) {
            *reinterpret_cast<float2*>(p + b * 2) = make_float2(v[b * 2 + 0], v[b * 2 + 1]);
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) p[c] = v[c];
    }
}

template <int K>
__device__ __forceinline__ void store_row(double* p, const cuda::std::array<double, K>& v) {
    if constexpr (K % 2 == 0) {
#pragma unroll
        for (int b = 0; b < K / 2; ++b) {
            *reinterpret_cast<double2*>(p + b * 2) = make_double2(v[b * 2 + 0], v[b * 2 + 1]);
        }
    } else {
#pragma unroll
        for (int c = 0; c < K; ++c) p[c] = v[c];
    }
}

// A fixed-order tree over one value per thread, returned to every thread of the block. Every
// thread must reach it. A kernel that sums twice passes each call its own array, or synchronises
// in between, since the first total is still being read when the second tree starts writing.
__device__ __forceinline__ double block_tree_sum(double v, double (&sh)[kBlock]) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    return sh[0];
}
