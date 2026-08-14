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

// --- reorder.cu: keys for the Morton layout -------------------------------------------------
// inverse maps a mesh node to its position in the node permutation; lowest[e] is the smallest such
// position over tetrahedron e's four nodes, which is the key its own permutation sorts by.
void launch_tet_lowest_node(const int* inverse, const int* tet_nodes, int* lowest, int n_tet,
                            cudaStream_t stream);

// --- pattern.cu: segment-wise CSR construction ----------------------------------------------
// Each builder comes in two halves, because how many distinct values a segment holds is not known
// until it has been sorted. `count` fills out_ptr with the exclusive scan of the row lengths and
// returns nnz, which is the length the caller then allocates out_idx at; `fill` writes the rows
// out_ptr describes. The candidates themselves never reach global memory. Both halves synchronise
// the stream.

// Distinct nodes of the tetrahedra in each segment of a corner CSR (c = 4e + i). ptr/idx is
// rhs.cu's node2corner map for the stiffness pattern, or a per-(node, tissue) corner CSR for the
// recovery slots.
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
// a contribution whose column is missing lands in a neighbouring slot rather than faulting.
// data is overwritten. scale[e] = vols[e] * cond[e]. ptr/idx are rhs.cu's node2corner map.
void launch_stiffness_rows(const double* g, const double* scale, const int* tet_nodes,
                           const int* ptr, const int* idx, const int* indptr, const int* indices,
                           double* data, int n_rows, cudaStream_t stream);

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
//
// The fit takes the widest basis its patch can carry and drops a rung where it cannot. status[s]
// records which one it took, counted down from the widest: 0 harmonic cubic, 1 harmonic quadratic,
// 2 linear, 3 no gradient determined.
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

// --- moments.cu: streaming UQ statistics over the |E| this directory produces ---------------
void launch_accumulate_moments(const float* magn, double* sum_e, double* sumsq_e, int n,
                               cudaStream_t stream);
