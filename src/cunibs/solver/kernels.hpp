#pragma once
#include "common.hpp"

// Launchers for the per-placement pipeline stages, grouped by the translation unit that
// implements them. The AMG solver's own launchers live in solver.hpp and aggregate.hpp.

// --- gradient.cu: P1 basis-function gradients and element volumes --------------------------
// nodes_mm is (n_nodes, 3) in millimetres; g is (n_tet, 4, 3) in 1/m and vols is (n_tet,) in m^3.
void launch_p1_gradients(const double* nodes_mm, const int* tet_nodes, double* g, double* vols,
                         int n_tet, cudaStream_t stream);

// --- dadt.cu: Biot-Savart dA/dt from the coil's dipoles -----------------------------------
void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int n_dip, int n_nodes, float didt, float mu0_4pi, cudaStream_t stream);

// --- dadt_element.cu -----------------------------------------------------------------------
void launch_dadt_element_average(const float* dadt_nodes, const int* tet_nodes, float* dadt_elm,
                                 int n_tet, cudaStream_t stream);

// --- rhs.cu: FEM right-hand-side assembly --------------------------------------------------
// launch_rhs fuses the corner dot product into the node gather; the staged pair pays a corner
// pass first so that both halves coalesce.
void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream);

void launch_rhs_staged(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                       const int* idx, float* b, int n_nodes, int n_tet, cudaStream_t stream);

// b_block is row-major (n_nodes, k) float32 — the layout the block solver consumes.
void launch_rhs_staged_block(const float* const* dadt_elm, const float* g, const float* neg_vc,
                             const int* ptr, const int* idx, float* b_block, int n_nodes,
                             int n_tet, int k, cudaStream_t stream);

// --- pattern.cu: segment-wise CSR construction ----------------------------------------------
// Both entry points take two caller-owned int32 work buffers and leave the result CSR as out_ptr
// plus the first nnz entries of cand, which they return. Both synchronise the stream.

// Distinct nodes of the tetrahedra in each segment of a corner CSR (c = 4e + i). ptr/idx is
// rhs.cu's node2corner map for the stiffness pattern, or a per-(node, tissue) corner CSR for the
// recovery slots. cand and sorted must hold 4 * n_corner entries.
int build_incident_node_csr(const int* tet_nodes, const int* ptr, const int* idx, int* cand,
                            int* sorted, int* out_ptr, int n_seg, int n_corner,
                            cudaStream_t stream);

// Recovery patches over the first-ring CSR built above. neighbour[j] is the slot of the same
// tissue centred on r1_idx[j]. A slot reaching fewer than min_nodes grows to the union of its
// neighbours' rings. n_cand is the capacity of cand and sorted.
int build_patch_csr(const int* r1_ptr, const int* r1_idx, const int* neighbour, int min_nodes,
                    int* cand, int* sorted, int* out_ptr, int n_slots, int n_cand,
                    cudaStream_t stream);

// --- stiffness.cu: conductivity stiffness values over a prebuilt CSR pattern ----------------
// indptr/indices must have sorted column indices per row and cover every (i, j) the tets touch;
// a contribution whose column is missing lands in a neighbouring slot rather than faulting.
// data is overwritten. scale[e] = vols[e] * cond[e]. ptr/idx are rhs.cu's node2corner map.
void launch_stiffness_rows(const double* g, const double* scale, const int* tet_nodes,
                           const int* ptr, const int* idx, const int* indptr, const int* indices,
                           double* data, int n_rows, cudaStream_t stream);

// --- l1.cu: l1-Jacobi smoother scaling over a CSR operator ----------------------------------
// dinv[i] = 1 / (sign(a_ii) · Σ_j |a_ij|), the diagonal included in the row sum, and 1 where that
// sum is zero. indices need not be sorted; the row is scanned for the diagonal, and a row without
// one keeps the positive sign. dinv is overwritten and must hold n_rows entries.
void launch_l1_dinv(const int* indptr, const int* indices, const float* data, float* dinv,
                    int n_rows, cudaStream_t stream);

// --- reconstruct.cu: E = -grad(v) - dA/dt --------------------------------------------------
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

// Harmonic-constrained potential recovery. pptr/pidx is a CSR of patch NODES per slot, not
// corners, and w carries a 3-vector per entry: grad_v[s] = sum_m w[s,m] * v[m].
// status[s] records which rung the fit took: 0 harmonic, 1 linear, 2 no gradient determined.
void launch_hpr_weights(const double* nodes_mm, const int* pptr, const int* pidx,
                        const int* slot_node, float* w, int* status, int n_slots,
                        cudaStream_t stream);

// v_block is row-major (n_nodes, stride) float64; column c of each slot uses dadt_nodes[c].
// stride is the block width the potential was solved at, which is k except on the serial path.
void launch_hpr_grad(const double* v_block, const float* w, const int* pptr, const int* pidx,
                     const int* slot_node, const float* const* dadt_nodes, float* const* e_slots,
                     int n_slots, int k, int stride, cudaStream_t stream);

// slot_of_corner is (4 * n_tet,): the slot each corner reads back from.
void launch_recover_elements(const float* const* e_slots, const int* slot_of_corner,
                             float* const* e_out, float* const* magn_out, int n_tet, int k,
                             cudaStream_t stream);

// --- adm.cu: ADM reciprocity weights -------------------------------------------------------
void launch_element_weight(const double* values, const int* tet_nodes, const float* g,
                           const float* neg_vc, double* w_e, int n_tet, cudaStream_t stream);

// corner is indexed by corner id c = 4e + i, not by element.
void launch_node_gather(const double* corner, const int* ptr, const int* idx, double* out,
                        int n_nodes, cudaStream_t stream);

void launch_node_scatter3(const double* w_e, const int* ptr, const int* idx, double* node_w,
                          int n_nodes, cudaStream_t stream);

// --- moments.cu: streaming UQ statistics ---------------------------------------------------
void launch_accumulate_moments(const float* magn, double* sum_e, double* sumsq_e, int n,
                               cudaStream_t stream);

// --- place.cu: coil placement frames -------------------------------------------------------
// handles may be null, for callers that want only the scalp projection and normal; the frame
// then takes an arbitrary but deterministic in-plane axis. degenerate (n_pl) is set to 1 where
// a supplied handle left the in-plane axis undefined and that same fallback was used instead;
// pass null alongside a null handles, where the flag would carry no information. The transform
// is orthonormal either way.
void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* a, const double* b, const double* c, const double* tnorm,
                  double* out, int* degenerate, int n_pl, int n_tri, cudaStream_t stream);
