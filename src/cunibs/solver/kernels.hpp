#pragma once
#include "common.hpp"

// Launchers for the per-placement pipeline stages, grouped by the translation unit that
// implements them. The AMG solver's own launchers live in solver.hpp and aggregate.hpp.

// --- dadt.cu: Biot-Savart dA/dt from the coil's dipoles -----------------------------------
void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int n_dip, int n_nodes, float didt, float mu0_4pi, cudaStream_t stream);

// --- dadt_element.cu -----------------------------------------------------------------------
void launch_dadt_element_average(const float* dadt_nodes, const int* tet_nodes, float* dadt_elm,
                                 int n_tet, cudaStream_t stream);

// --- rhs.cu: FEM right-hand-side assembly --------------------------------------------------
// launch_rhs fuses the corner dot product into the node gather, for callers whose neg_vc
// changes per call; the weighted pair instead reuses a precomputed wg across placements.
void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream);

void launch_rhs_weighted(const float* dadt_elm, const float* wg, const int* ptr, const int* idx,
                         float* b, int n_nodes, int n_tet, cudaStream_t stream);

void launch_weighted_gradient(const float* g, const float* neg_vc, float* wg, int n_tet,
                              cudaStream_t stream);

// b_block is row-major (n_nodes, k) float32 — the layout the block solver consumes.
void launch_rhs_weighted_block(const float* const* dadt_elm, const float* wg, const int* ptr,
                               const int* idx, float* b_block, int n_nodes, int n_tet, int k,
                               cudaStream_t stream);

// --- reconstruct.cu: E = -grad(v) - dA/dt --------------------------------------------------
void launch_reconstruct(const double* v, const int* tet_nodes, const float* g,
                        const float* dadt_elm, float* e_out, float* magn_out, int n_tet,
                        cudaStream_t stream);

// v_block is row-major (n_nodes, k) float64.
void launch_reconstruct_block(const double* v_block, const int* tet_nodes, const float* g,
                              const float* const* dadt_elm, float* const* e_out,
                              float* const* magn_out, int n_tet, int k, cudaStream_t stream);

// --- adm.cu: ADM reciprocity weights -------------------------------------------------------
void launch_element_weight(const double* values, const int* tet_nodes, const float* g,
                           const float* neg_vc, double* w_e, int n_tet, cudaStream_t stream);

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
