#pragma once
#include "core/common.hpp"

// Launchers for the coil-side stages: where the coil sits on the scalp, and the field its
// dipoles induce.

// dA/dt at n_nodes nodes (fp64, mm) for k placements of one coil: pos_m and mom are its (n_dip, 3)
// coil-frame dipoles in fp64, transforms the k (4, 4) coil-to-head affines in mm. Column c of node
// n lands at out[n * out_stride + 3 c], so out_stride = 3k fills an interleaved (n, k, 3) block.
void launch_dadt_block(const double* pos_m, const double* mom, const double* transforms,
                       int n_dip, int k, const double* nodes_mm, int n_nodes, float* out,
                       int out_stride, float didt, float mu0_4pi, cudaStream_t stream);

void launch_dadt_element_average(const float* dadt_nodes, const int* tet_nodes, float* dadt_elm,
                                 int n_tet, cudaStream_t stream);
// dadt_nodes is (n_nodes, k, 3); dadt_elm holds k (n_tet, 3) outputs.
void launch_dadt_element_average_block(const float* dadt_nodes, const int* tet_nodes,
                                       float* const* dadt_elm, int n_tet, int k,
                                       cudaStream_t stream);

// handles may be null, for callers that want only the scalp projection and normal; the frame
// then takes an arbitrary but deterministic in-plane axis. degenerate (n_pl) is set to 1 where
// a supplied handle left the in-plane axis undefined and that same fallback was used instead;
// pass null alongside a null handles, where the flag would carry no information. The transform
// is orthonormal either way.
// verts is (n_tri, 3, 3) and tnorm (n_tri, 3); out is (n_pl, 4, 4).
void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* verts, const double* tnorm, double* out, int* degenerate,
                  int n_pl, int n_tri, cudaStream_t stream);
