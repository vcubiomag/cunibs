#pragma once
#include "core/common.hpp"

// Launchers for the coil-side stages: where the coil sits on the scalp, and the field its
// dipoles induce. Nothing here reads the stiffness or the solved potential.

// --- dadt.cu: Biot-Savart dA/dt from the coil's dipoles -----------------------------------
void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int n_dip, int n_nodes, float didt, float mu0_4pi, cudaStream_t stream);

// --- dadt_element.cu -----------------------------------------------------------------------
void launch_dadt_element_average(const float* dadt_nodes, const int* tet_nodes, float* dadt_elm,
                                 int n_tet, cudaStream_t stream);

// --- place.cu: coil placement frames -------------------------------------------------------
// handles may be null, for callers that want only the scalp projection and normal; the frame
// then takes an arbitrary but deterministic in-plane axis. degenerate (n_pl) is set to 1 where
// a supplied handle left the in-plane axis undefined and that same fallback was used instead;
// pass null alongside a null handles, where the flag would carry no information. The transform
// is orthonormal either way.
void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* a, const double* b, const double* c, const double* tnorm,
                  double* out, int* degenerate, int n_pl, int n_tri, cudaStream_t stream);
