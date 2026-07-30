#pragma once
#include <cuda_runtime.h>

// Unsmoothed pairwise aggregation, a port of the AMGx SIZE_4 selector
// (aggregation/selectors/size4_selector.cu) for the scalar, single-GPU, deterministic case.
// Two rounds of handshaking matching: round 1 forms pairs, round 2 merges pairs into quads,
// then leftovers join their strongest aggregated neighbour.
//
// Every kernel launches one thread per row and writes only its own index, so the result is
// independent of block and grid geometry and reproducible run to run. There are no atomics.
//
// The reduced stiffness is exactly symmetric, so the edge weight collapses from AMGx's
// 0.5*(|a_ij| + |a_ji|)/max(|a_ii|,|a_jj|) to |a_ij|/max(|a_ii|,|a_jj|), dropping the
// transpose search that dominates AMGx's weight kernel.

// Fills `aggregates` (length n_rows, device) with a surjective row -> aggregate map in
// [0, n_coarse) and returns n_coarse. `row_ptr`, `col_idx`, `values` are a device fp32 CSR
// with an explicit diagonal. Synchronises `stream` internally (the matching loops are
// data-dependent), so this is a setup-time call and is not graph-capturable.
int select_size4(int n_rows, int nnz, const int* row_ptr, const int* col_idx,
                 const float* values, int* aggregates, cudaStream_t stream);
