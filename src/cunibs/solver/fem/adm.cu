#include "core/device_math.cuh"
#include "fem/fem.hpp"

#include <cuda/std/algorithm>

#include <climits>
#include <cstdint>
#include <stdexcept>

namespace {

// grad_v = Σ_i v[tet_nodes[e,i]] · g[e,i,:] for one P1 tetrahedron, taken as
// Σ_{i≥1} (v_i − v_0) g[e,i,:] since P1 gradients sum to zero over the corners. The direct sum
// carries v's common level through fp32 g that do not sum to zero to the last bit, which leaks
// |v| |g| ε into a gradient that is small beside it. Accumulated in fp64 because callers subtract
// a comparable term from it and lose the leading digits. g[e,0,:] is never read.
__device__ __forceinline__ Vec3 tet_grad4(const double* __restrict__ v,
                                          Tet4View<const int> tet_nodes,
                                          TetGradView<const float> g, int e) {
    const double v0 = v[tet_nodes(e, 0)];
    double gx = 0.0, gy = 0.0, gz = 0.0;
#pragma unroll
    for (int i = 1; i < 4; ++i) {
        const double di = v[tet_nodes(e, i)] - v0;
        gx = fma(di, static_cast<double>(g(e, i, 0)), gx);
        gy = fma(di, static_cast<double>(g(e, i, 1)), gy);
        gz = fma(di, static_cast<double>(g(e, i, 2)), gz);
    }
    return {gx, gy, gz};
}

//   out[n] = Σ_{c ∋ n} corner[c]   (node2corner stores corner ids c = 4e + i)
// One thread per node over the incidence CSR, so the per-node order is the one every other
// assembly uses and the sum is reproducible; a scatter would need atomics and sum each node's
// corners in whatever order the blocks retired.
__global__ void node_gather_kernel(const double* __restrict__ corner, const int* __restrict__ ptr,
                                   const int* __restrict__ idx, double* __restrict__ out,
                                   int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const int end = ptr[node + 1];
    double acc = 0.0;
    for (int p = ptr[node]; p < end; ++p) acc += corner[idx[p]];
    out[node] = acc;
}

// The position of e in the ascending roi_elems, or -1.
__device__ __forceinline__ int roi_find(const int* __restrict__ roi_elems, int n_roi, int e) {
    if (n_roi == 0 || e < roi_elems[0] || e > roi_elems[n_roi - 1]) return -1;
    const int* const at = cuda::std::lower_bound(roi_elems, roi_elems + n_roi, e);
    return *at == e ? static_cast<int>(at - roi_elems) : -1;
}

//   node_w[n,k] = ¼ Σ_{c ∋ n} w_e[c>>2, k],   w_e = (−neg_vc[e]) · (G_e λ) − α_e ê
// with α_e the ROI weight of e, or no term outside the ROI. Each corner recomputes its element's
// weight, so no per-element array is stored. Every product and sum is rounded on its own: the
// weight is a difference of comparable terms, and the result must not depend on what the compiler
// chooses to fuse.
__global__ void node_weight_kernel(const double* __restrict__ lam,
                                   const int* __restrict__ tet_nodes, const float* __restrict__ g,
                                   const float* __restrict__ neg_vc, int n_tet,
                                   const int* __restrict__ ptr, const int* __restrict__ idx,
                                   const int* __restrict__ roi_elems,
                                   const double* __restrict__ roi_alpha, int n_roi, double dir_x,
                                   double dir_y, double dir_z, double* __restrict__ node_w,
                                   int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const Tet4View<const int> tets(tet_nodes, n_tet);
    const TetGradView<const float> grads(g, n_tet);
    const Vec3View<double> out(node_w, n_nodes);

    const int begin = ptr[node];
    const int end = ptr[node + 1];
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int p = begin; p < end; ++p) {
        const int e = idx[p] >> 2;
        const Vec3 grad = tet_grad4(lam, tets, grads, e);
        const double s = -static_cast<double>(neg_vc[e]);
        double wx = __dmul_rn(s, grad[0]);
        double wy = __dmul_rn(s, grad[1]);
        double wz = __dmul_rn(s, grad[2]);
        if (const int r = roi_find(roi_elems, n_roi, e); r >= 0) {
            const double a = roi_alpha[r];
            wx = __dsub_rn(wx, __dmul_rn(a, dir_x));
            wy = __dsub_rn(wy, __dmul_rn(a, dir_y));
            wz = __dsub_rn(wz, __dmul_rn(a, dir_z));
        }
        ax = __dadd_rn(ax, wx);
        ay = __dadd_rn(ay, wy);
        az = __dadd_rn(az, wz);
    }
    out(node, 0) = __dmul_rn(0.25, ax);
    out(node, 1) = __dmul_rn(0.25, ay);
    out(node, 2) = __dmul_rn(0.25, az);
}

// A regular grid, as the ADM kernels see it: index (i, j, k) sits at origin + (i, j, k) * spacing,
// and a point's values are row-major over (i, j, k).
struct GridGeom {
    double origin[3];
    double spacing[3];
    int n[3];
};

// A candidate's coil frame mapped straight into fractional grid indices. The translation is taken
// against the grid origin in fp64 once per candidate, so the per-dipole arithmetic is fp32 on
// quantities the size of the coil: a fractional index then carries ~1e-5 of a cell of rounding.
struct CandidateMap {
    float a[3];       // the coil origin, in grid index units
    float rs[3][3];   // rotation over spacing: coil metres to grid index units
    float rot[3][3];  // rotation, for the moments
};

__device__ __forceinline__ CandidateMap candidate_map(const double* __restrict__ frame,
                                                     const GridGeom& g) {
    CandidateMap c;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        c.a[i] = static_cast<float>((frame[i * 4 + 3] * 1e-3 - g.origin[i]) / g.spacing[i]);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            c.rot[i][j] = static_cast<float>(frame[i * 4 + j]);
            c.rs[i][j] = static_cast<float>(frame[i * 4 + j] / g.spacing[i]);
        }
    }
    return c;
}

// Slack on the fractional index bounds, 1e-6 of a cell, the same as evaluate.py's _INDEX_TOL so the
// host's off-grid message agrees with the kernel's verdict.
constexpr float kIndexTol = 1e-6f;

// Slots of the fault array: the lowest candidate with a dipole off the grid, and the lowest with
// one in an unsampled cell.
enum AdmFault : int { kAdmOffGrid = 0, kAdmUnsampled = 1 };

// The lowest corner of a dipole's cell, or false when the dipole is off the grid. A coordinate on
// the last sample belongs to the last cell, so every in-grid point has a full cell around it.
__device__ __forceinline__ bool locate(const CandidateMap& c, const GridGeom& g, float px, float py,
                                       float pz, int (&i0)[3], float (&f)[3]) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float x = fmaf(c.rs[i][0], px, fmaf(c.rs[i][1], py, fmaf(c.rs[i][2], pz, c.a[i])));
        // >= / <= rather than a negated comparison, so a non-finite coordinate fails closed.
        if (!(x >= -kIndexTol && x <= static_cast<float>(g.n[i] - 1) + kIndexTol)) return false;
        const int lo = min(max(static_cast<int>(floorf(x)), 0), g.n[i] - 2);
        i0[i] = lo;
        f[i] = fminf(fmaxf(x - static_cast<float>(lo), 0.0f), 1.0f);
    }
    return true;
}

// A minimum, so the candidate reported does not depend on which warp got there first.
__device__ __forceinline__ void report_fault(int* fault, AdmFault kind, int candidate) {
    atomicMin(&fault[kind], candidate);
}

// out += g W x d for every moment set, g = |d|^-3. Coordinates are grid index units, so the
// caller scales by mu0_4pi / spacing^2. d = target - source is formed directly, since a node a
// fraction of a millimetre from the target leaves no digits in |r|^2 + |s|^2 - 2 r.s.
template <int D>
__device__ __forceinline__ void add_source(float (&out)[D * 3], float dx, float dy, float dz,
                                           const float* __restrict__ w) {
    const float d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
    const float inv = rsqrtf(d2);
    const float g = inv * inv * inv;
#pragma unroll
    for (int d = 0; d < D; ++d) {
        const float wx = w[3 * d], wy = w[3 * d + 1], wz = w[3 * d + 2];
        out[3 * d + 0] = fmaf(g, wy * dz - wz * dy, out[3 * d + 0]);
        out[3 * d + 1] = fmaf(g, wz * dx - wx * dz, out[3 * d + 1]);
        out[3 * d + 2] = fmaf(g, wx * dy - wy * dx, out[3 * d + 2]);
    }
}

// The reciprocity field is singular at every mesh node, and a 3 mm cell can straddle the scalp,
// so trilinear interpolation alone smears a corner's near-node spike into a dipole a millimetre
// away. The grid therefore stores, at each point, the field of every source outside the eight
// cells around it. A cell whose 27 neighbours hold a source gets a slot: its own eight corner
// values with all of those sources removed, and the list of them, which the evaluator sums at the
// dipole directly. Adjacency is decided by the source's cell, floor(s), so every side agrees on
// every source. Extended cells -1 .. n-1 per axis give each grid point all eight of its cells.
__device__ __forceinline__ bool adjacent(float p, float s) {
    const float o = p - floorf(s);
    return o >= 0.0f && o <= 1.0f;
}

// A slot's sources are packed one entry each: the position, then the D weight vectors, padded to
// whole float4s so an entry is a few vector loads.
template <int D>
__host__ __device__ constexpr int entry_floats() {
    return (3 + 3 * D + 3) / 4 * 4;
}

template <int D>
__device__ __forceinline__ void load_entry(float (&e)[entry_floats<D>()],
                                           const float* __restrict__ entries, int k) {
    const float4* v = reinterpret_cast<const float4*>(entries) +
                      static_cast<std::int64_t>(k) * (entry_floats<D>() / 4);
#pragma unroll
    for (int i = 0; i < entry_floats<D>() / 4; ++i) {
        const float4 x = __ldg(v + i);
        e[4 * i] = x.x;
        e[4 * i + 1] = x.y;
        e[4 * i + 2] = x.z;
        e[4 * i + 3] = x.w;
    }
}

// E_{p,d} = didt * sum_j m_{p,j} . Q_d(s_{p,j}), one warp per candidate. Q is trilinearly
// interpolated from the grid's (nx, ny, nz, D, 3) samples, so a corner is one contiguous read of
// every direction, or from the cell's slot, whose sources are then added directly. The per-dipole
// products are fp32, like the samples they read; the sum over the coil is fp64, because a
// figure-of-eight coil's two wings cancel in it.
template <int D>
__global__ void __launch_bounds__(kBlock)
    adm_evaluate_kernel(const double* __restrict__ frames, int n_frames,
                        const float* __restrict__ coil_pos, const float* __restrict__ coil_mom,
                        int n_dip, const float* __restrict__ q,
                        const unsigned char* __restrict__ cell_ok,
                        const int* __restrict__ cell_slot, const float* __restrict__ q_near,
                        const int* __restrict__ list_ptr, const float* __restrict__ entries,
                        float near_scale, GridGeom g, double didt, double* __restrict__ out,
                        int* __restrict__ fault) {
    const int cand = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / kWarp);
    const int lane = static_cast<int>(threadIdx.x % kWarp);
    cuda::std::array<double, D> e;
#pragma unroll
    for (int d = 0; d < D; ++d) e[d] = 0.0;
    if (cand < n_frames) {
        const CandidateMap c = candidate_map(frames + static_cast<std::int64_t>(cand) * 16, g);
        const std::int64_t sy = static_cast<std::int64_t>(g.n[2]) * (3 * D);
        const std::int64_t sx = static_cast<std::int64_t>(g.n[1]) * sy;
        for (int j = lane; j < n_dip; j += kWarp) {
            const float px = coil_pos[3 * j], py = coil_pos[3 * j + 1], pz = coil_pos[3 * j + 2];
            int i0[3];
            float f[3];
            if (!locate(c, g, px, py, pz, i0, f)) {
                report_fault(fault, kAdmOffGrid, cand);
                continue;
            }
            const std::int64_t cell =
                (static_cast<std::int64_t>(i0[0]) * (g.n[1] - 1) + i0[1]) * (g.n[2] - 1) + i0[2];
            if (!cell_ok[cell]) {
                report_fault(fault, kAdmUnsampled, cand);
                continue;
            }
            float qv[D * 3];
#pragma unroll
            for (int t = 0; t < D * 3; ++t) qv[t] = 0.0f;
            const int slot = __ldg(cell_slot + cell);
            // A slot's eight corners are contiguous, (x, y, z) bit order.
            const float* base =
                slot < 0 ? q + i0[0] * sx + i0[1] * sy + static_cast<std::int64_t>(i0[2]) * (3 * D)
                         : q_near + static_cast<std::int64_t>(slot) * (8 * 3 * D);
            const std::int64_t cx = slot < 0 ? sx : 4 * 3 * D;
            const std::int64_t cy = slot < 0 ? sy : 2 * 3 * D;
#pragma unroll
            for (int corner = 0; corner < 8; ++corner) {
                const int bx = corner >> 2, by = (corner >> 1) & 1, bz = corner & 1;
                const float w = (bx ? f[0] : 1.0f - f[0]) * (by ? f[1] : 1.0f - f[1]) *
                                (bz ? f[2] : 1.0f - f[2]);
                const float* v = base + bx * cx + by * cy + bz * (3 * D);
#pragma unroll
                for (int t = 0; t < D * 3; ++t) qv[t] = fmaf(w, __ldg(v + t), qv[t]);
            }
            if (slot >= 0) {
                const float x = static_cast<float>(i0[0]) + f[0];
                const float y = static_cast<float>(i0[1]) + f[1];
                const float z = static_cast<float>(i0[2]) + f[2];
                float near_q[D * 3];
#pragma unroll
                for (int t = 0; t < D * 3; ++t) near_q[t] = 0.0f;
                const int end = __ldg(list_ptr + slot + 1);
                for (int k = __ldg(list_ptr + slot); k < end; ++k) {
                    float entry[entry_floats<D>()];
                    load_entry<D>(entry, entries, k);
                    add_source<D>(near_q, x - entry[0], y - entry[1], z - entry[2], entry + 3);
                }
#pragma unroll
                for (int t = 0; t < D * 3; ++t) qv[t] = fmaf(near_scale, near_q[t], qv[t]);
            }
            const float mx = coil_mom[3 * j], my = coil_mom[3 * j + 1], mz = coil_mom[3 * j + 2];
            float m[3];
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                m[i] = c.rot[i][0] * mx + c.rot[i][1] * my + c.rot[i][2] * mz;
            }
#pragma unroll
            for (int d = 0; d < D; ++d) {
                e[d] += static_cast<double>(m[0] * qv[3 * d] + m[1] * qv[3 * d + 1] +
                                            m[2] * qv[3 * d + 2]);
            }
        }
    }
    warp_reduce_sum<kWarp>(e);
    if (cand < n_frames && lane == 0) {
#pragma unroll
        for (int d = 0; d < D; ++d) out[static_cast<std::int64_t>(cand) * D + d] = didt * e[d];
    }
}

// q_near[slot, corner] = q at the corner minus every source in the slot's entries that the
// corner's own sample kept, one thread per (slot, corner) from first_slot on. What is left is the
// field of the sources outside the cell's 27 neighbours.
template <int D>
__global__ void __launch_bounds__(kBlock)
    adm_slot_corners_kernel(const std::int64_t* __restrict__ slot_cells, int first_slot,
                            int n_slots, const int* __restrict__ list_ptr,
                            const float* __restrict__ entries, const float* __restrict__ q,
                            GridGeom g, float scale, float* __restrict__ q_near) {
    const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n = static_cast<std::int64_t>(n_slots - first_slot) * 8;
    if (t >= n) return;
    const int slot = first_slot + static_cast<int>(t / 8);
    const int corner = static_cast<int>(t % 8);
    const std::int64_t cell = slot_cells[slot];
    const std::int64_t ncz = g.n[2] - 1, ncy = g.n[1] - 1;
    const int ix = static_cast<int>(cell / (ncy * ncz)) + (corner >> 2);
    const int iy = static_cast<int>((cell / ncz) % ncy) + ((corner >> 1) & 1);
    const int iz = static_cast<int>(cell % ncz) + (corner & 1);
    const float px = static_cast<float>(ix), py = static_cast<float>(iy),
                pz = static_cast<float>(iz);
    float acc[D * 3];
#pragma unroll
    for (int c = 0; c < D * 3; ++c) acc[c] = 0.0f;
    const int end = list_ptr[slot + 1];
    for (int k = list_ptr[slot]; k < end; ++k) {
        float entry[entry_floats<D>()];
        load_entry<D>(entry, entries, k);
        if (adjacent(px, entry[0]) && adjacent(py, entry[1]) && adjacent(pz, entry[2])) continue;
        add_source<D>(acc, px - entry[0], py - entry[1], pz - entry[2], entry + 3);
    }
    const float* v = q + ((static_cast<std::int64_t>(ix) * g.n[1] + iy) * g.n[2] + iz) * (3 * D);
    float* dst = q_near + (static_cast<std::int64_t>(slot) * 8 + corner) * (3 * D);
#pragma unroll
    for (int c = 0; c < D * 3; ++c) dst[c] = fmaf(-scale, acc[c], v[c]);
}

// Marks every grid point some dipole's interpolation would read, for the points a frame needs
// that the eager sampling did not cover. Writing 1 is idempotent, so no atomics are needed.
__global__ void __launch_bounds__(kBlock)
    adm_mark_kernel(const double* __restrict__ frames, int n_frames,
                    const float* __restrict__ coil_pos, int n_dip, GridGeom g,
                    unsigned char* __restrict__ need) {
    const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::int64_t>(n_frames) * n_dip) return;
    const int cand = static_cast<int>(t / n_dip);
    const int j = static_cast<int>(t % n_dip);
    const CandidateMap c = candidate_map(frames + static_cast<std::int64_t>(cand) * 16, g);
    int i0[3];
    float f[3];
    if (!locate(c, g, coil_pos[3 * j], coil_pos[3 * j + 1], coil_pos[3 * j + 2], i0, f)) return;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        const std::int64_t ix = i0[0] + (corner >> 2);
        const std::int64_t iy = i0[1] + ((corner >> 1) & 1);
        const std::int64_t p = (ix * g.n[1] + iy) * g.n[2] + i0[2] + (corner & 1);
        need[p] = 1;
    }
}

// A grid point is reachable when, about some coverage centre's coil axis, its cylindrical
// coordinates (rho, h) fall in the coil's swept profile: an in-plane rotation carries each dipole
// round a circle of fixed (rho, h), so the set every rotation reaches is the profile's solid of
// revolution. profile is a raster over (rho, h) that already includes the margin.
__global__ void __launch_bounds__(kBlock)
    adm_reach_kernel(GridGeom g, const double* __restrict__ origins,
                     const double* __restrict__ axes, int n_centers,
                     const unsigned char* __restrict__ profile, int n_rho, int n_h, double h0,
                     double cell, unsigned char* __restrict__ mask) {
    const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n_pts = static_cast<std::int64_t>(g.n[0]) * g.n[1] * g.n[2];
    if (p >= n_pts) return;
    const int iz = static_cast<int>(p % g.n[2]);
    const int iy = static_cast<int>((p / g.n[2]) % g.n[1]);
    const int ix = static_cast<int>(p / (static_cast<std::int64_t>(g.n[1]) * g.n[2]));
    const double x = g.origin[0] + ix * g.spacing[0];
    const double y = g.origin[1] + iy * g.spacing[1];
    const double z = g.origin[2] + iz * g.spacing[2];
    unsigned char hit = 0;
    for (int c = 0; c < n_centers && !hit; ++c) {
        const double dx = x - origins[3 * c];
        const double dy = y - origins[3 * c + 1];
        const double dz = z - origins[3 * c + 2];
        const double h = dx * axes[3 * c] + dy * axes[3 * c + 1] + dz * axes[3 * c + 2];
        const double rho = sqrt(fmax(dx * dx + dy * dy + dz * dz - h * h, 0.0));
        const int ir = static_cast<int>(rho / cell);
        const int ih = static_cast<int>(floor((h - h0) / cell));
        if (ir < n_rho && ih >= 0 && ih < n_h) {
            hit = profile[static_cast<std::int64_t>(ir) * n_h + ih];
        }
    }
    mask[p] = hit;
}

constexpr int kSampleBlock = 128;
// Tiles summed in fp32 before each fp64 flush. The field sums every mesh node, and a single fp32
// running sum over ~10^6 terms carries ~1e-3 of rounding.
constexpr int kSampleFlush = 16;
// Grid points each thread carries, which amortises one source tile over K points.
constexpr int kSamplePoints = 2;

// One block's sampled points: kSampleBlock * K consecutive entries of points, which lie close
// together, and their bounding box in absolute indices.
struct SampleBlock {
    int lo[3];
    int hi[3];
};

__device__ __forceinline__ void decode_point(std::int64_t p, const GridGeom& g, int (&i)[3]) {
    i[2] = static_cast<int>(p % g.n[2]);
    i[1] = static_cast<int>((p / g.n[2]) % g.n[1]);
    i[0] = static_cast<int>(p / (static_cast<std::int64_t>(g.n[1]) * g.n[2]));
}

// A tile source's W then W x (s - o), padded to whole float4s so it is a few vector loads.
template <int D>
__host__ __device__ constexpr int wm_vectors() {
    return (6 * D + 3) / 4;
}

// The cross product is accumulated as (sum g W) x (p - o) - sum g W x (s - o): six FMAs per set
// where W x (p - s) takes nine. o is the grid's middle index, so the two terms cancel only by
// |p - o| / |p - s|, under half the grid in cells since every source within a cell of p is
// excluded, and a point's value does not depend on which other points share its block. CHECK
// tests that exclusion, which only tiles near the block need.
template <int D, int K, bool CHECK>
__device__ __forceinline__ void sample_tile(int tile, const float4* s_pos, const float4* s_cell,
                                            const float4 (*s_wm)[wm_vectors<D>()],
                                            const float (&px)[K], const float (&py)[K],
                                            const float (&pz)[K], float (&aw)[K][D * 3],
                                            float (&am)[K][D * 3]) {
#pragma unroll 2
    for (int j = 0; j < tile; ++j) {
        const float4 sp = s_pos[j];
        float4 sc;
        if constexpr (CHECK) sc = s_cell[j];
        float wm[4 * wm_vectors<D>()];
#pragma unroll
        for (int v = 0; v < wm_vectors<D>(); ++v) {
            const float4 x = s_wm[j][v];
            wm[4 * v] = x.x;
            wm[4 * v + 1] = x.y;
            wm[4 * v + 2] = x.z;
            wm[4 * v + 3] = x.w;
        }
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float dx = px[k] - sp.x, dy = py[k] - sp.y, dz = pz[k] - sp.z;
            const float d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
            const float inv = rsqrtf(d2);
            float gw = inv * inv * inv;
            if constexpr (CHECK) {
                // An excluded source can sit on the point, where gw is infinite: a select.
                const bool skip = fmaxf(fabsf(px[k] - sc.x),
                                        fmaxf(fabsf(py[k] - sc.y), fabsf(pz[k] - sc.z))) <= 0.5f;
                gw = skip ? 0.0f : gw;
            }
#pragma unroll
            for (int c = 0; c < D * 3; ++c) {
                aw[k][c] = fmaf(gw, wm[c], aw[k][c]);
                am[k][c] = fmaf(gw, wm[3 * D + c], am[k][c]);
            }
        }
    }
}

// q[point] = scale * sum over sources not in the point's eight cells of W x d / |d|^3. src and w
// are the sources in grid index units sorted by cell, and tile_box each kSampleBlock-source tile's
// cell bounding box (lo xyz, hi xyz), so a tile that cannot reach any of the block's points skips
// the exclusion test. points are flat C-order grid indices.
template <int D, int K>
__global__ void __launch_bounds__(kSampleBlock)
    adm_sample_kernel(const float* __restrict__ src, const float* __restrict__ w, int n_src,
                      const int* __restrict__ tile_box, const std::int64_t* __restrict__ points,
                      int n_points, GridGeom g, double scale, float* __restrict__ q) {
    __shared__ float4 s_pos[kSampleBlock];   // s - o
    __shared__ float4 s_cell[kSampleBlock];  // floor(s) + 0.5 - o
    __shared__ float4 s_wm[kSampleBlock][wm_vectors<D>()];
    __shared__ SampleBlock blk;

    const float o[3] = {static_cast<float>((g.n[0] - 1) / 2), static_cast<float>((g.n[1] - 1) / 2),
                        static_cast<float>((g.n[2] - 1) / 2)};
    const int first = blockIdx.x * (kSampleBlock * K);
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) {
            blk.lo[i] = INT_MAX;
            blk.hi[i] = INT_MIN;
        }
    }
    __syncthreads();

    float px[K], py[K], pz[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = first + k * kSampleBlock + threadIdx.x;
        int ip[3];
        decode_point(points[min(i, n_points - 1)], g, ip);
        if (i < n_points) {
            for (int a = 0; a < 3; ++a) {
                atomicMin(&blk.lo[a], ip[a]);
                atomicMax(&blk.hi[a], ip[a]);
            }
        }
        px[k] = static_cast<float>(ip[0]) - o[0];
        py[k] = static_cast<float>(ip[1]) - o[1];
        pz[k] = static_cast<float>(ip[2]) - o[2];
    }
    __syncthreads();

    float aw[K][D * 3], am[K][D * 3];
    double sum[K][D * 3];
#pragma unroll
    for (int k = 0; k < K; ++k) {
#pragma unroll
        for (int c = 0; c < D * 3; ++c) {
            aw[k][c] = am[k][c] = 0.0f;
            sum[k][c] = 0.0;
        }
    }

    int tiles = 0;
    for (int base = 0; base < n_src; base += kSampleBlock) {
        const int tile = min(kSampleBlock, n_src - base);
        if (const int i = threadIdx.x; i < tile) {
            const int j = base + i;
            const float sx = src[3 * j], sy = src[3 * j + 1], sz = src[3 * j + 2];
            const float rx = sx - o[0], ry = sy - o[1], rz = sz - o[2];
            s_pos[i] = make_float4(rx, ry, rz, 0.0f);
            s_cell[i] = make_float4(floorf(sx) - o[0] + 0.5f, floorf(sy) - o[1] + 0.5f,
                                    floorf(sz) - o[2] + 0.5f, 0.0f);
            float wm[4 * wm_vectors<D>()] = {};
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const float* wd = w + (static_cast<std::int64_t>(d) * n_src + j) * 3;
                const float wx = wd[0], wy = wd[1], wz = wd[2];
                wm[3 * d] = wx;
                wm[3 * d + 1] = wy;
                wm[3 * d + 2] = wz;
                wm[3 * D + 3 * d] = wy * rz - wz * ry;
                wm[3 * D + 3 * d + 1] = wz * rx - wx * rz;
                wm[3 * D + 3 * d + 2] = wx * ry - wy * rx;
            }
#pragma unroll
            for (int v = 0; v < wm_vectors<D>(); ++v) {
                s_wm[i][v] = make_float4(wm[4 * v], wm[4 * v + 1], wm[4 * v + 2], wm[4 * v + 3]);
            }
        }
        __syncthreads();

        // A source is in one of p's eight cells when floor(s) is p - 1 or p on every axis.
        const int* box = tile_box + 6 * (base / kSampleBlock);
        const bool check = box[0] <= blk.hi[0] && box[3] >= blk.lo[0] - 1 &&
                           box[1] <= blk.hi[1] && box[4] >= blk.lo[1] - 1 &&
                           box[2] <= blk.hi[2] && box[5] >= blk.lo[2] - 1;
        if (check) {
            sample_tile<D, K, true>(tile, s_pos, s_cell, s_wm, px, py, pz, aw, am);
        } else {
            sample_tile<D, K, false>(tile, s_pos, s_cell, s_wm, px, py, pz, aw, am);
        }
        __syncthreads();

        if (++tiles == kSampleFlush || base + kSampleBlock >= n_src) {
            tiles = 0;
#pragma unroll
            for (int k = 0; k < K; ++k) {
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    const float* a = aw[k] + 3 * d;
                    const float* m = am[k] + 3 * d;
                    sum[k][3 * d] += static_cast<double>(a[1] * pz[k] - a[2] * py[k] - m[0]);
                    sum[k][3 * d + 1] += static_cast<double>(a[2] * px[k] - a[0] * pz[k] - m[1]);
                    sum[k][3 * d + 2] += static_cast<double>(a[0] * py[k] - a[1] * px[k] - m[2]);
                }
#pragma unroll
                for (int c = 0; c < D * 3; ++c) aw[k][c] = am[k][c] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int i = first + k * kSampleBlock + threadIdx.x;
        if (i < n_points) {
            float* const dst = q + points[i] * (D * 3);
#pragma unroll
            for (int c = 0; c < D * 3; ++c) dst[c] = static_cast<float>(scale * sum[k][c]);
        }
    }
}

// out[i] = scale * sum over every source of W x (p_i - s) / |p_i - s|^3 at arbitrary points, the
// unapproximated field the grid stands in for. One point per thread against shared source tiles,
// summed in fp32 per tile run and fp64 across runs, as adm_sample_kernel does.
template <int D>
__global__ void __launch_bounds__(kSampleBlock)
    adm_q_points_kernel(const float* __restrict__ src, const float* __restrict__ w, int n_src,
                        const float* __restrict__ points, int n_points, double scale,
                        float* __restrict__ out) {
    __shared__ float4 s_pos[kSampleBlock];
    __shared__ float s_w[kSampleBlock][D * 3];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = min(i, n_points - 1);
    const float px = points[3 * p], py = points[3 * p + 1], pz = points[3 * p + 2];
    float acc[D * 3];
    double sum[D * 3];
#pragma unroll
    for (int c = 0; c < D * 3; ++c) {
        acc[c] = 0.0f;
        sum[c] = 0.0;
    }

    int tiles = 0;
    for (int base = 0; base < n_src; base += kSampleBlock) {
        const int tile = min(kSampleBlock, n_src - base);
        if (const int t = threadIdx.x; t < tile) {
            const int j = base + t;
            s_pos[t] = make_float4(src[3 * j], src[3 * j + 1], src[3 * j + 2], 0.0f);
#pragma unroll
            for (int d = 0; d < D; ++d) {
#pragma unroll
                for (int c = 0; c < 3; ++c) {
                    s_w[t][3 * d + c] = w[(static_cast<std::int64_t>(d) * n_src + j) * 3 + c];
                }
            }
        }
        __syncthreads();
        for (int j = 0; j < tile; ++j) {
            const float4 sp = s_pos[j];
            add_source<D>(acc, px - sp.x, py - sp.y, pz - sp.z, s_w[j]);
        }
        __syncthreads();
        if (++tiles == kSampleFlush || base + kSampleBlock >= n_src) {
            tiles = 0;
#pragma unroll
            for (int c = 0; c < D * 3; ++c) {
                sum[c] += static_cast<double>(acc[c]);
                acc[c] = 0.0f;
            }
        }
    }
    if (i < n_points) {
#pragma unroll
        for (int c = 0; c < D * 3; ++c) {
            out[static_cast<std::int64_t>(i) * (D * 3) + c] = static_cast<float>(scale * sum[c]);
        }
    }
}

}  // namespace

void launch_node_gather(const double* corner, const int* ptr, const int* idx, double* out,
                        int n_nodes, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_nodes)) {
        node_gather_kernel<<<blocks, kBlock, 0, stream>>>(corner, ptr, idx, out, n_nodes);
    }
}

void launch_node_weight(const double* lam, const int* tet_nodes, const float* g,
                        const float* neg_vc, int n_tet, const int* ptr, const int* idx,
                        const int* roi_elems, const double* roi_alpha, int n_roi,
                        const double* direction, double* node_w, int n_nodes,
                        cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_nodes)) {
        node_weight_kernel<<<blocks, kBlock, 0, stream>>>(
            lam, tet_nodes, g, neg_vc, n_tet, ptr, idx, roi_elems, roi_alpha, n_roi, direction[0],
            direction[1], direction[2], node_w, n_nodes);
    }
}

namespace {

GridGeom grid_geom(const double* origin, const double* spacing, const int* shape) {
    GridGeom g;
    for (int i = 0; i < 3; ++i) {
        g.origin[i] = origin[i];
        g.spacing[i] = spacing[i];
        g.n[i] = shape[i];
    }
    return g;
}

}  // namespace

void launch_adm_evaluate(const double* frames, int n_frames, const float* coil_pos,
                         const float* coil_mom, int n_dip, const float* q,
                         const unsigned char* cell_ok, const int* cell_slot, const float* q_near,
                         const int* list_ptr, const float* entries, double mu0_4pi,
                         const double* origin, const double* spacing, const int* shape, int n_dir,
                         double didt, double* out, int* fault, cudaStream_t stream) {
    const GridGeom g = grid_geom(origin, spacing, shape);
    const unsigned blocks = grid_for(static_cast<std::int64_t>(n_frames) * kWarp);
    if (blocks == 0) return;
    const auto near_scale = static_cast<float>(mu0_4pi / (spacing[0] * spacing[0]));
    if (n_dir == 1) {
        adm_evaluate_kernel<1><<<blocks, kBlock, 0, stream>>>(
            frames, n_frames, coil_pos, coil_mom, n_dip, q, cell_ok, cell_slot, q_near, list_ptr,
            entries, near_scale, g, didt, out, fault);
    } else if (n_dir == 3) {
        adm_evaluate_kernel<3><<<blocks, kBlock, 0, stream>>>(
            frames, n_frames, coil_pos, coil_mom, n_dip, q, cell_ok, cell_slot, q_near, list_ptr,
            entries, near_scale, g, didt, out, fault);
    } else {
        throw std::invalid_argument("adm_evaluate: directions must be 1 or 3");
    }
}

void launch_adm_slot_corners(const std::int64_t* slot_cells, int first_slot, int n_slots,
                             const int* list_ptr, const float* entries, int n_dir, const float* q,
                             double mu0_4pi, const double* origin, const double* spacing,
                             const int* shape, float* q_near, cudaStream_t stream) {
    const GridGeom g = grid_geom(origin, spacing, shape);
    const auto scale = static_cast<float>(mu0_4pi / (spacing[0] * spacing[0]));
    const unsigned blocks = grid_for(static_cast<std::int64_t>(n_slots - first_slot) * 8);
    if (blocks == 0) return;
    if (n_dir == 1) {
        adm_slot_corners_kernel<1><<<blocks, kBlock, 0, stream>>>(
            slot_cells, first_slot, n_slots, list_ptr, entries, q, g, scale, q_near);
    } else if (n_dir == 3) {
        adm_slot_corners_kernel<3><<<blocks, kBlock, 0, stream>>>(
            slot_cells, first_slot, n_slots, list_ptr, entries, q, g, scale, q_near);
    } else {
        throw std::invalid_argument("adm_slot_corners: directions must be 1 or 3");
    }
}

int adm_sample_tile() { return kSampleBlock; }

void launch_adm_q_points(const float* src, const float* src_w, int n_src, int n_dir,
                         const float* points, int n_points, double scale, float* out,
                         cudaStream_t stream) {
    const unsigned blocks = grid_for(n_points, kSampleBlock);
    if (blocks == 0) return;
    if (n_dir == 1) {
        adm_q_points_kernel<1><<<blocks, kSampleBlock, 0, stream>>>(src, src_w, n_src, points,
                                                                     n_points, scale, out);
    } else if (n_dir == 3) {
        adm_q_points_kernel<3><<<blocks, kSampleBlock, 0, stream>>>(src, src_w, n_src, points,
                                                                     n_points, scale, out);
    } else {
        throw std::invalid_argument("adm_q_points: directions must be 1 or 3");
    }
}

int adm_entry_floats(int n_dir) {
    if (n_dir == 1) return entry_floats<1>();
    if (n_dir == 3) return entry_floats<3>();
    throw std::invalid_argument("adm: directions must be 1 or 3");
}

void launch_adm_sample(const float* src, const float* src_w, int n_src, int n_dir,
                       const int* tile_box, const std::int64_t* points, int n_points,
                       double mu0_4pi,
                       const double* origin, const double* spacing, const int* shape, float* q,
                       cudaStream_t stream) {
    const GridGeom g = grid_geom(origin, spacing, shape);
    const double scale = mu0_4pi / (spacing[0] * spacing[0]);
    const unsigned blocks = grid_for(n_points, kSampleBlock * kSamplePoints);
    if (blocks == 0) return;
    if (n_dir == 1) {
        adm_sample_kernel<1, kSamplePoints><<<blocks, kSampleBlock, 0, stream>>>(
            src, src_w, n_src, tile_box, points, n_points, g, scale, q);
    } else if (n_dir == 3) {
        adm_sample_kernel<3, kSamplePoints><<<blocks, kSampleBlock, 0, stream>>>(
            src, src_w, n_src, tile_box, points, n_points, g, scale, q);
    } else {
        throw std::invalid_argument("adm_sample: directions must be 1 or 3");
    }
}

void launch_adm_mark(const double* frames, int n_frames, const float* coil_pos, int n_dip,
                     const double* origin, const double* spacing, const int* shape,
                     unsigned char* need, cudaStream_t stream) {
    const GridGeom g = grid_geom(origin, spacing, shape);
    if (const unsigned blocks = grid_for(static_cast<std::int64_t>(n_frames) * n_dip)) {
        adm_mark_kernel<<<blocks, kBlock, 0, stream>>>(frames, n_frames, coil_pos, n_dip, g,
                                                        need);
    }
}

void launch_adm_reach(const double* origin, const double* spacing, const int* shape,
                      const double* origins, const double* axes, int n_centers,
                      const unsigned char* profile, int n_rho, int n_h, double h0, double cell,
                      unsigned char* mask, cudaStream_t stream) {
    const GridGeom g = grid_geom(origin, spacing, shape);
    const std::int64_t n_pts = static_cast<std::int64_t>(shape[0]) * shape[1] * shape[2];
    if (const unsigned blocks = grid_for(n_pts)) {
        adm_reach_kernel<<<blocks, kBlock, 0, stream>>>(g, origins, axes, n_centers, profile,
                                                         n_rho, n_h, h0, cell, mask);
    }
}
