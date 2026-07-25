#pragma once
#include <cuda_runtime.h>

void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int n_dip, int n_nodes, float didt, float mu0_4pi, cudaStream_t stream);

void launch_dadt_element_average(const float* dadt_nodes, const int* tet_nodes, float* dadt_elm,
                                 int n_tet, cudaStream_t stream);

void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream);

void launch_rhs_weighted(const float* dadt_elm, const float* wg, const int* ptr, const int* idx,
                         float* b, int n_nodes, int n_tet, cudaStream_t stream);

void launch_weighted_gradient(const float* g, const float* neg_vc, float* wg, int n_tet,
                              cudaStream_t stream);

void launch_reconstruct(const double* v, const int* tet_nodes, const float* g,
                        const float* dadt_elm, float* e_out, float* magn_out, int n_tet,
                        cudaStream_t stream);

void launch_element_weight(const double* values, const int* tet_nodes, const float* g,
                           const float* neg_vc, double* w_e, int n_tet, cudaStream_t stream);

void launch_node_scatter3(const double* w_e, const int* ptr, const int* idx, double* node_w,
                          int n_nodes, cudaStream_t stream);

void launch_accumulate_moments(const float* magn, double* sum_e, double* sumsq_e, int n,
                               cudaStream_t stream);

void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* a, const double* b, const double* c, const double* tnorm,
                  double* out, int n_pl, int n_tri, cudaStream_t stream);

// Block (k <= 8 placements per chunk) variants of the per-placement stages: the shared
// mesh arrays (tet_nodes, wg, node2corner, g) are read once per chunk rather than once
// per placement. Per-placement arrays stay separate contiguous buffers, passed as
// pointer packs, so the per-placement result layout is the same as the serial path's.
constexpr int kMaxStageBlock = 8;

// b_block is row-major (n_nodes, k) float32 — the layout the block solver consumes.
void launch_rhs_weighted_block(const float* const* dadt_elm, const float* wg, const int* ptr,
                               const int* idx, float* b_block, int n_nodes, int n_tet, int k,
                               cudaStream_t stream);

// v_block is row-major (n_nodes, k) float64.
void launch_reconstruct_block(const double* v_block, const int* tet_nodes, const float* g,
                              const float* const* dadt_elm, float* const* e_out,
                              float* const* magn_out, int n_tet, int k, cudaStream_t stream);
