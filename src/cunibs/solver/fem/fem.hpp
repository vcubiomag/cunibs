#pragma once
#include "core/common.hpp"

// Launchers for the mesh-side stages: the P1 operator, the assembled system, and the
// post-processing that turns a solved potential back into a field. The AMG solver's own
// launchers live in amg/, the coil-side ones in coil/.

// --- gradient.cu: P1 basis-function gradients and element volumes --------------------------
// nodes_mm is (n_nodes, 3) in millimetres; g is (n_tet, 4, 3) in 1/m and vols is (n_tet,) in m^3.
void launch_p1_gradients(const double* nodes_mm, const int* tet_nodes, double* g, double* vols,
                         int n_tet, cudaStream_t stream);

// --- rhs.cu: FEM right-hand-side assembly --------------------------------------------------
// The per-element RHS gathered at each node; the UQ draws use it, since their conductivities
// change per draw.
void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream);

// The forward path's RHS as an assembled operator on the nodal dA/dt (see rhs.cu). op is
// (nnz, 4) float32 over the stiffness pattern indptr/indices, one packed (Bx, By, Bz, column)
// per nonzero; scale[e] = neg_vc[e] / 4 in float64. a is (n_nodes, k, 3) and b is (n_nodes, k).
void launch_rhs_operator_fill(const double* g, const double* scale, const int* tet_nodes,
                              const int* ptr, const int* idx, const int* indptr,
                              const int* indices, float* op, int n_rows, cudaStream_t stream);
void launch_rhs_operator(const int* indptr, const float* op, const float* a, float* b,
                         int n_nodes, int k, cudaStream_t stream);

// --- reorder.cu: keys for the Morton layout -------------------------------------------------
// inverse maps a mesh node to its position in the node permutation; lowest[e] is the smallest such
// position over tetrahedron e's four nodes, which is the key its own permutation sorts by.
void launch_tet_lowest_node(const int* inverse, const int* tet_nodes, int* lowest, int n_tet,
                            cudaStream_t stream);

// --- pattern.cu: segment-wise CSR construction ----------------------------------------------
// Each builder comes in two halves, because how many distinct values a segment holds is not known
// until it has been sorted. `count` fills out_ptr with the exclusive scan of the row lengths and
// returns nnz, which is the length the caller then allocates out_idx at; `fill` writes the rows
// out_ptr describes. A segment's candidates stay in shared memory unless it is far wider than a
// head mesh's, which is the only case that materialises them. `count` synchronises the stream to
// return nnz; `fill` is stream-ordered.

// Distinct nodes of the tetrahedra in each segment of a corner CSR (c = 4e + i), every corner of
// a segment being one of the same node's. ptr/idx is rhs.cu's node2corner map for the stiffness
// pattern, or a per-(node, tissue) corner CSR for the recovery slots.
int count_incident_node_csr(const int* tet_nodes, const int* ptr, const int* idx, int* out_ptr,
                            int n_seg, cudaStream_t stream);
void fill_incident_node_csr(const int* tet_nodes, const int* ptr, const int* idx,
                            const int* out_ptr, int* out_idx, int n_seg, cudaStream_t stream);

// Recovery patches over the first-ring CSR built above. neighbour[j] is the slot of the same
// tissue centred on r1_idx[j]. A slot reaching fewer than min_nodes grows to the union of its
// neighbours' rings.
int count_patch_csr(const int* r1_ptr, const int* r1_idx, const int* neighbour, int min_nodes,
                    int* out_ptr, int n_slots, cudaStream_t stream);
void fill_patch_csr(const int* r1_ptr, const int* r1_idx, const int* neighbour, int min_nodes,
                    const int* out_ptr, int* out_idx, int n_slots, cudaStream_t stream);

// --- stiffness.cu: conductivity stiffness values over a prebuilt CSR pattern ----------------
// indptr/indices must have sorted column indices per row and cover every (i, j) the tets touch;
// nothing checks it, and a missing column corrupts a neighbouring entry.
// data is overwritten. scale[e] = vols[e] * cond[e]. ptr/idx are rhs.cu's node2corner map.
void launch_stiffness_rows(const double* g, const double* scale, const int* tet_nodes,
                           const int* ptr, const int* idx, const int* indptr, const int* indices,
                           double* data, int n_rows, cudaStream_t stream);

// --- reconstruct.cu: E = -grad(v) - dA/dt --------------------------------------------------
// g must be P1 basis gradients, whose four corners sum to zero: the kernels difference the
// potential against corner 0 and never read g[e, 0].
void launch_reconstruct(const double* v, const int* tet_nodes, const float* g,
                        const float* dadt_elm, float* e_out, float* magn_out, int n_tet,
                        cudaStream_t stream);

// v_block is row-major (n_nodes, k) float64.
void launch_reconstruct_block(const double* v_block, const int* tet_nodes, const float* g,
                              const float* const* dadt_elm, float* const* e_out,
                              float* const* magn_out, int n_tet, int k, cudaStream_t stream);

// --- recovery.cu: patch-recovery post-processing --------------------------------------------
// A "slot" is what a patch is fitted around: a node in the global mode, a (node, tissue) pair in
// the tissue-restricted ones. The SPR entry points take a corner CSR over slots, in the same
// c = 4e + i encoding rhs.cu uses; the harmonic ones take a CSR of patch nodes. Either way the
// per-slot reduction order is fixed by the CSR's build.

// Sets is_boundary[n] = 1 for every node of a face that only one tet owns; entries for other nodes
// are left alone, so is_boundary arrives zeroed. ptr/idx is rhs.cu's node2corner map, over which
// a face's owners are counted from the incidence list of one of its nodes.
void launch_mark_outer_boundary(const int* tet_nodes, const int* ptr, const int* idx,
                                int* is_boundary, int n_tet, cudaStream_t stream);

// One scalar weight per corner, from a linear least-squares fit over each slot's patch.
// slot_node may be null when a slot is a node. is_boundary is indexed by node, not by slot, and
// is read through slot_node; it may be null to fit every slot. n_fallback may be null; when given
// it counts slots that took the volume-weighted average.
void launch_spr_weights(const double* nodes_mm, const int* tet_nodes, const float* vols,
                        const int* ptr, const int* idx, const int* slot_node,
                        const int* is_boundary, float* w, int* n_fallback, int n_slots,
                        cudaStream_t stream);

// e_slots[c] is (n_slots, 3); e_in[c] is (n_tet, 3). k <= kMaxStageBlock, every width compiled.
void launch_recover_nodes(const float* const* e_in, const float* w, const int* ptr, const int* idx,
                          float* const* e_slots, int n_slots, int k, cudaStream_t stream);

// Smoothed outward normals for the skin surface, one per triangle. node_ptr/node_idx is a CSR
// over mesh nodes whose entries are triangle corner ids c = 3t + i, and whose order fixes the
// summation order every node's smoothing runs in.
void launch_skin_normals(const double* nodes_mm, const int* tris, const int* node_ptr,
                         const int* node_idx, double* out_tri_normals, int n_tri, int n_nodes,
                         cudaStream_t stream);

// Harmonic-constrained potential recovery. pptr/pidx is a CSR of patch NODES per slot, not
// corners, and w carries a 3-vector per entry: grad_v[s] = sum_m w[s,m] * v[m].
//
// The fit takes the widest basis its patch can carry and drops a rung where it cannot. status[s]
// records which one it took, counted down from the widest: 0 harmonic cubic, 1 harmonic quadratic,
// 2 linear, 3 no gradient determined.
// The fitted slots are active[0, n), or every slot when active is null and n is the slot count.
void launch_hpr_weights(const double* nodes_mm, const int* pptr, const int* pidx,
                        const int* slot_node, float* w, int* status, const int* active, int n,
                        cudaStream_t stream);

// v_block is row-major (n_nodes, stride) float64; column c of each slot uses dadt_nodes[c].
// stride is the block width the potential was solved at, which is k except on the serial path.
// Only the slots active[0, n) are written, or all n_slots when active is null.
void launch_hpr_grad(const double* v_block, const float* w, const int* pptr, const int* pidx,
                     const int* slot_node, const float* const* dadt_nodes, float* const* e_slots,
                     int n_slots, const int* active, int n, int k, int stride, int n_nodes,
                     cudaStream_t stream);

// slot_of_corner is (4 * n_tet,): the slot each corner reads back from.
// Only the elements active[0, n) are written, or all n_tet when active is null.
void launch_recover_elements(const float* const* e_slots, const int* slot_of_corner,
                             float* const* e_out, float* const* magn_out, int n_tet,
                             const int* active, int n, int k, cudaStream_t stream);

// --- raycast.cu: first hit of a ray against a tissue surface --------------------------------
// verts is (n_tri, 3, 3), the three corner coordinates of each triangle; out_t is one distance
// per ray, in the units origins and verts carry, and is +inf where the ray misses. Hits nearer
// than min_t are skipped, so a ray leaving one surface does not hit the one it started on.
void launch_ray_first_hit(const double* origins, const double* directions, const double* verts,
                          double* out_t, int n_rays, int n_tri, double min_t,
                          cudaStream_t stream);

// --- adm.cu: ADM reciprocity weights -------------------------------------------------------
void launch_element_weight(const double* values, const int* tet_nodes, const float* g,
                           const float* neg_vc, double* w_e, int n_tet, cudaStream_t stream);

// corner is indexed by corner id c = 4e + i, not by element.
void launch_node_gather(const double* corner, const int* ptr, const int* idx, double* out,
                        int n_nodes, cudaStream_t stream);

void launch_node_gather3(const double* w_e, const int* ptr, const int* idx, double* node_w,
                          int n_nodes, cudaStream_t stream);

// Candidate evaluation against a sampled reciprocity field. The grid is origin/spacing (metres)
// and shape (3 ints, host arrays); q is (nx, ny, nz, n_dir, 3) and cell_ok one byte per cell, set
// when all eight of its corners were sampled. frames are (n_frames, 4, 4) coil-to-head affines
// with the translation in mm; coil_pos/coil_mom are the coil-frame dipoles in metres. out is
// (n_frames, n_dir). fault is two device ints the caller sets to n_frames: they receive the lowest
// candidate with a dipole off the grid, and the lowest with one in an unsampled cell.
void launch_adm_evaluate(const double* frames, int n_frames, const float* coil_pos,
                         const float* coil_mom, int n_dip, const float* q,
                         const unsigned char* cell_ok, const double* origin,
                         const double* spacing, const int* shape, int n_dir, double didt,
                         double* out, int* fault, cudaStream_t stream);
// Sets need[p] = 1 for every grid point some dipole's interpolation reads.
void launch_adm_mark(const double* frames, int n_frames, const float* coil_pos, int n_dip,
                     const double* origin, const double* spacing, const int* shape,
                     unsigned char* need, cudaStream_t stream);
// mask[p] = 1 where a grid point falls in the coil's swept (rho, h) profile about any centre's
// axis. profile is (n_rho, n_h) over rho = ir * cell, h = h0 + ih * cell.
void launch_adm_reach(const double* origin, const double* spacing, const int* shape,
                      const double* origins, const double* axes, int n_centers,
                      const unsigned char* profile, int n_rho, int n_h, double h0, double cell,
                      unsigned char* mask, cudaStream_t stream);

// --- summary.cu: volume-weighted region metrics of |E| ---------------------------------------
// out is (k, n_out) float64 with n_out = kSummaryQuantiles + n_q + n_f: the peak, its element's
// barycentre, the centre of gravity, mean and std, then the n_q quantiles qs and the volume at or
// above fracs[f] times quantile qs[anchor]. fields are k (n_tet,) arrays, float64 when f64 and
// float32 otherwise, read at index (n,); w and bary are the region's volumes (n,) and barycentres
// (n, 3). work holds region_summary_work_size(n, k, f64) bytes.
inline constexpr int kSummaryPeak = 0;
inline constexpr int kSummaryPeakLocation = 1;
inline constexpr int kSummaryCenterOfGravity = 4;
inline constexpr int kSummaryMean = 7;
inline constexpr int kSummaryStd = 8;
inline constexpr int kSummaryQuantiles = 9;
inline constexpr int kMaxSummaryQuantiles = 16;
std::size_t region_summary_work_size(int n, int k, bool f64);
void launch_region_summary(const void* const* fields, bool f64, int k, int n, const int* index,
                           const double* w, const double* bary, const double* qs, int n_q,
                           int anchor, const double* fracs, int n_f, void* work,
                           std::size_t work_bytes, double* out, cudaStream_t stream);

// --- moments.cu: streaming UQ statistics over the |E| this directory produces ---------------
// sum_e += magn[c] and sumsq_e += magn[c]^2 for k <= kMaxStageBlock draws, folded in draw order.
void launch_accumulate_moments_block(const float* const* magn, int k, double* sum_e,
                                     double* sumsq_e, int n, cudaStream_t stream);
// out (n, k) = a (n, p) times s (k, p) transposed, each entry summed in term order.
void launch_row_dot(const double* a, const double* s, double* out, int n, int p, int k,
                    cudaStream_t stream);
// x0 (n, k) = q (q^T b) for an A-orthonormal fp32 basis q (n, m) and b (n, k), each column's
// arithmetic independent of k. work holds sweep_seed_work_size(n, m, k) doubles.
std::int64_t sweep_seed_work_size(int n, int m, int k);
void launch_sweep_seed(const float* q, const double* b, int n, int m, int k, double* work,
                       double* x0, cudaStream_t stream);
