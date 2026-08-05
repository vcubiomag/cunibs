"""Precompute the per-tissue stiffness components that make each MC sample a cheap value swap.

K is linear in the per-tissue conductivities: ``K(σ).data = K_base.data + Σ_t σ_t·Kt.data`` on a
fixed CSR sparsity pattern. Assembling each tissue's geometric contribution once turns the
per-sample matrix build into a single GEMV over the nonzeros, and lets the solver swap
coefficients without re-analysing structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp

from cunibs.fem.assembly import (
    TISSUE_CONDUCTIVITY,
    as_int32_tets,
    build_node2corner,
    conductivity_per_tet,
    fill_stiffness_values,
    gradient_operator,
    stiffness_pattern,
)
from cunibs.fem.solve import (
    DEFAULT_MAX_ITERS,
    DEFAULT_TOLERANCE,
    SolverContext,
    build_native_vcycle,
    ground_node_of,
    reduce_matrix,
    solve_ordering,
)
from cunibs.solver import NativeVCycle, PcgAmgSolver


@dataclass
class ConductivityUQPrecompute:
    """Reusable state for conductivity UQ on one mesh, amortised across placements/samples."""

    perturbed_tags: tuple[int, ...]
    idx: cp.ndarray  # grounded row/col index (drops the ground DOF, Morton-ordered)
    indptr: cp.ndarray  # reduced CSR pattern
    indices: cp.ndarray
    base_data: cp.ndarray  # (nnz,) f64 — non-perturbed tissues at nominal σ
    tissue_data: cp.ndarray  # (n_perturbed, nnz) f64 — per-tissue unit-σ contribution
    precond: NativeVCycle  # fp32 V-cycle frozen at nominal σ
    pcg: PcgAmgSolver  # fp64 outer PCG; matrix values swapped per sample
    tolerance: float
    max_iters: int
    nominal_sigma: cp.ndarray  # (n_perturbed,) f64
    nominal_data: (
        cp.ndarray
    )  # (nnz,) f64 — reduced values at nominal σ (frozen-preconditioner point)

    def combine(self, sigma: cp.ndarray) -> cp.ndarray:
        """Assemble the reduced matrix values for one conductivity sample."""
        return self.base_data + sigma @ self.tissue_data

    def preconditioner_for(self, sample_data: cp.ndarray) -> NativeVCycle:
        """Build a throwaway V-cycle matched to one draw's conductivities.

        Only for a draw the nominal-σ preconditioner fails to converge. The result is not
        cached: draws are i.i.d. around nominal, so an extreme draw's hierarchy is no better a
        default for the next draw than the nominal one.
        """
        return build_native_vcycle(
            self.indptr,
            self.indices,
            cp.ascontiguousarray(sample_data.astype(cp.float32)),
        )


def build_conductivity_uq_precompute(
    ctx: SolverContext, perturbed_tags: tuple[int, ...]
) -> ConductivityUQPrecompute:
    """Assemble the reference pattern, per-tissue components, and the nominal-σ preconditioner."""
    g64, vols = gradient_operator(ctx.nodes_mm, ctx.tet_nodes)
    ground_node = ground_node_of(ctx.nodes_mm)
    # Must match prepare_grounded_solver's ordering, or this frozen CSR pattern would not be
    # the one the forward solver's hierarchy was built on.
    idx = solve_ordering(ctx.nodes_mm, ground_node)

    tets = as_int32_tets(ctx.tet_nodes)
    n_nodes = int(ctx.n_nodes)
    a_full = stiffness_pattern(tets, n_nodes, g64.dtype)
    n2c_ptr, n2c_idx = build_node2corner(tets, n_nodes)

    # Grounding selects a submatrix of a pattern that never moves, so every reduced nonzero comes
    # from exactly one full nonzero. Reducing a matrix whose values are their own positions
    # recovers that map once, which turns every component into a refill plus a gather.
    a_full.data[:] = cp.arange(a_full.nnz, dtype=g64.dtype)
    k_ref = reduce_matrix(a_full, idx)
    gather = cp.ascontiguousarray(k_ref.data.astype(cp.int32))
    row_ptr = cp.ascontiguousarray(k_ref.indptr.astype(cp.int32))
    col_idx = cp.ascontiguousarray(k_ref.indices.astype(cp.int32))
    nnz_red = int(gather.shape[0])
    del k_ref

    def reduced_data_for(cond: cp.ndarray, out: cp.ndarray | None = None) -> cp.ndarray:
        fill_stiffness_values(a_full, g64, vols, cond, tets, n2c_ptr, n2c_idx)
        return cp.take(a_full.data, gather, out=out)

    # Every component is assembled over the whole mesh with zero conductivity outside its own
    # tets: the zeros keep the reference pattern and contribute nothing to its values.
    tissue_data = cp.empty((len(perturbed_tags), nnz_red), dtype=cp.float64)
    for i, tag in enumerate(perturbed_tags):
        reduced_data_for((ctx.tet_tags == tag).astype(cp.float64), out=tissue_data[i])

    cond_nom = conductivity_per_tet(ctx.tet_tags)
    perturbed_mask = cp.isin(ctx.tet_tags, cp.asarray(perturbed_tags))
    base_data = reduced_data_for(cp.where(perturbed_mask, 0.0, cond_nom))
    nominal_direct = reduced_data_for(cond_nom)

    nominal_sigma = cp.asarray(
        [TISSUE_CONDUCTIVITY[t] for t in perturbed_tags], dtype=cp.float64
    )

    # Correctness gate: the linear model must reproduce the nominal direct assembly exactly.
    recon = base_data + nominal_sigma @ tissue_data
    rel = float(cp.linalg.norm(recon - nominal_direct) / cp.linalg.norm(nominal_direct))
    if rel > 1e-10:
        raise RuntimeError(f"UQ stiffness decomposition mismatch (rel={rel:.2e})")
    del nominal_direct

    nominal_data = cp.ascontiguousarray(recon)

    nominal_f32 = cp.ascontiguousarray(nominal_data.astype(cp.float32))
    precond = build_native_vcycle(row_ptr, col_idx, nominal_f32)
    pcg = PcgAmgSolver(row_ptr, col_idx, nominal_data)
    tolerance = DEFAULT_TOLERANCE
    max_iters = DEFAULT_MAX_ITERS

    return ConductivityUQPrecompute(
        perturbed_tags=perturbed_tags,
        idx=idx,
        indptr=row_ptr,
        indices=col_idx,
        base_data=base_data,
        tissue_data=tissue_data,
        precond=precond,
        pcg=pcg,
        tolerance=tolerance,
        max_iters=max_iters,
        nominal_sigma=nominal_sigma,
        nominal_data=nominal_data,
    )
