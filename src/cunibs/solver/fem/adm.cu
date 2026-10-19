#include "core/device_math.cuh"
#include "fem/fem.hpp"

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
        gx += di * static_cast<double>(g(e, i, 0));
        gy += di * static_cast<double>(g(e, i, 1));
        gz += di * static_cast<double>(g(e, i, 2));
    }
    return {gx, gy, gz};
}

//   w_e[e,k] = (−neg_vc[e]) · Σ_i values[tet_nodes[e,i]] · g[e,i,k]   (= vol_e·σ_e·(G_e λ))
// Accumulate in float64: the weight feeds a difference against the direct ROI term.
__global__ void element_weight_kernel(const double* __restrict__ values,
                                      const int* __restrict__ tet_nodes,
                                      const float* __restrict__ g,
                                      const float* __restrict__ neg_vc,
                                      double* __restrict__ w_e, int n_tet) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_tet) return;

    const Vec3 grad = tet_grad4(values, Tet4View<const int>(tet_nodes, n_tet),
                                TetGradView<const float>(g, n_tet), e);
    const Vec3View<double> w(w_e, n_tet);
    const double s = -static_cast<double>(neg_vc[e]);
    for (int k = 0; k < 3; ++k) w(e, k) = s * grad[k];
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

//   node_w[n,k] = ¼ Σ_{c ∋ n} w_e[c>>2, k]
__global__ void node_gather3_kernel(const double* __restrict__ w_e, const int* __restrict__ ptr,
                                     const int* __restrict__ idx, double* __restrict__ node_w,
                                     int n_nodes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;

    const auto w = unsized<Vec3View<const double>>(w_e);
    const Vec3View<double> out(node_w, n_nodes);

    const int begin = ptr[node];
    const int end = ptr[node + 1];
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int p = begin; p < end; ++p) {
        const int e = idx[p] >> 2;
        ax += w(e, 0);
        ay += w(e, 1);
        az += w(e, 2);
    }
    out(node, 0) = 0.25 * ax;
    out(node, 1) = 0.25 * ay;
    out(node, 2) = 0.25 * az;
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

// Slack on the fractional index bounds, the same 1e-6 of a cell the host-side check allowed.
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

// E_{p,d} = didt * sum_j m_{p,j} . Q_d(s_{p,j}), one warp per candidate. Q is trilinearly
// interpolated from the grid's (nx, ny, nz, D, 3) samples, so a corner is one contiguous read of
// every direction. The per-dipole products are fp32, like the samples they read; the sum over the
// coil is fp64, because a figure-of-eight coil's two wings cancel in it.
template <int D>
__global__ void __launch_bounds__(kBlock)
    adm_evaluate_kernel(const double* __restrict__ frames, int n_frames,
                        const float* __restrict__ coil_pos, const float* __restrict__ coil_mom,
                        int n_dip, const float* __restrict__ q,
                        const unsigned char* __restrict__ cell_ok, GridGeom g, double didt,
                        double* __restrict__ out, int* __restrict__ fault) {
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
            const float* base =
                q + i0[0] * sx + i0[1] * sy + static_cast<std::int64_t>(i0[2]) * (3 * D);
#pragma unroll
            for (int corner = 0; corner < 8; ++corner) {
                const int bx = corner >> 2, by = (corner >> 1) & 1, bz = corner & 1;
                const float w = (bx ? f[0] : 1.0f - f[0]) * (by ? f[1] : 1.0f - f[1]) *
                                (bz ? f[2] : 1.0f - f[2]);
                const float* v = base + bx * sx + by * sy + bz * (3 * D);
#pragma unroll
                for (int t = 0; t < D * 3; ++t) qv[t] = fmaf(w, __ldg(v + t), qv[t]);
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

}  // namespace

void launch_element_weight(const double* values, const int* tet_nodes, const float* g,
                           const float* neg_vc, double* w_e, int n_tet, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_tet)) {
        element_weight_kernel<<<blocks, kBlock, 0, stream>>>(values, tet_nodes, g, neg_vc, w_e,
                                                             n_tet);
    }
}

void launch_node_gather(const double* corner, const int* ptr, const int* idx, double* out,
                        int n_nodes, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_nodes)) {
        node_gather_kernel<<<blocks, kBlock, 0, stream>>>(corner, ptr, idx, out, n_nodes);
    }
}

void launch_node_gather3(const double* w_e, const int* ptr, const int* idx, double* node_w,
                          int n_nodes, cudaStream_t stream) {
    if (const unsigned blocks = grid_for(n_nodes)) {
        node_gather3_kernel<<<blocks, kBlock, 0, stream>>>(w_e, ptr, idx, node_w, n_nodes);
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
                         const unsigned char* cell_ok, const double* origin,
                         const double* spacing, const int* shape, int n_dir, double didt,
                         double* out, int* fault, cudaStream_t stream) {
    const GridGeom g = grid_geom(origin, spacing, shape);
    const unsigned blocks = grid_for(static_cast<std::int64_t>(n_frames) * kWarp);
    if (blocks == 0) return;
    if (n_dir == 1) {
        adm_evaluate_kernel<1><<<blocks, kBlock, 0, stream>>>(
            frames, n_frames, coil_pos, coil_mom, n_dip, q, cell_ok, g, didt, out, fault);
    } else if (n_dir == 3) {
        adm_evaluate_kernel<3><<<blocks, kBlock, 0, stream>>>(
            frames, n_frames, coil_pos, coil_mom, n_dip, q, cell_ok, g, didt, out, fault);
    } else {
        throw std::invalid_argument("adm_evaluate: directions must be 1 or 3");
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
