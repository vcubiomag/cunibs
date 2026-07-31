#pragma once
#include "common.hpp"

// Unsmoothed pairwise aggregation, a port of the AMGx SIZE_4 selector
// (aggregation/selectors/size4_selector.cu) for the scalar, single-GPU, deterministic case.
// Two rounds of handshaking matching: round 1 forms pairs, round 2 merges pairs into quads,
// then leftovers join their strongest aggregated neighbour.
//
// Upstream is github.com/srimanachanta/AMGX at b1e30c1, the commit this was ported from and
// the one every AMGx reference below names a file in. The tree itself is no longer vendored.
//
// Every row's entry is written by a single thread, from candidates combined under an
// associative rule, so the map is independent of block and grid geometry and reproducible run
// to run. The only atomics are the integer counters driving the convergence checks, and
// integer addition is associative, so the arrival order of the blocks does not matter either.
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
