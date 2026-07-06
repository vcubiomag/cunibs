#pragma once
#include <cuda_runtime.h>

void launch_dadt(const float* s, const float* mp, const float* sn, const float* r, float* out,
                 int n_dip, int n_nodes, float didt, float mu0_4pi, cudaStream_t stream);

void launch_rhs(const float* dadt_elm, const float* g, const float* neg_vc, const int* ptr,
                const int* idx, float* b, int n_nodes, cudaStream_t stream);

void launch_reconstruct(const double* v, const int* tet_nodes, const float* g,
                        const float* dadt_elm, float* e_out, float* magn_out, int n_tet,
                        cudaStream_t stream);

void launch_place(const double* centers, const double* handles, const double* dists,
                  const double* a, const double* b, const double* c, const double* tnorm,
                  double* out, int n_pl, int n_tri, cudaStream_t stream);
