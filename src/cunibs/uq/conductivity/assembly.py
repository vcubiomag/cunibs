"""Precompute the per-tissue stiffness components that make each MC sample a cheap value swap.

K is linear in the per-tissue conductivities: ``K(σ).data = K_base.data + Σ_t σ_t·Kt.data`` on a
fixed CSR sparsity pattern. Assembling each tissue's geometric contribution once turns the
per-sample matrix build into a single GEMV over the nonzeros, and lets AMGx swap coefficients
without re-analysing structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp

from cunibs.fem.assembly import (
    TISSUE_CONDUCTIVITY,
    assemble_stiffness,
    conductivity_per_tet,
    gradient_operator,
)
from cunibs.fem.solve import (
    AMGX_PRECONDITIONER_CONFIG,
    UQ_AMGX_CONFIG,
    SolverContext,
    _amgx_config_value,
    ground_node_of,
    grounded_index,
    reduce_matrix,
)
from cunibs.solver import AMGXFloatSolver, AMGXSolver, PcgAmgSolver


@dataclass
class ConductivityUQPrecompute:
    """Reusable state for conductivity UQ on one mesh, amortised across placements/samples."""

    perturbed_tags: tuple[int, ...]
    idx: cp.ndarray  # grounded row/col index (drops the ground DOF)
    n_nodes: int
    indptr: cp.ndarray  # reduced CSR pattern (device pointers handed to AMGx once)
    indices: cp.ndarray
    base_data: cp.ndarray  # (nnz,) f64 — non-perturbed tissues at nominal σ
    tissue_data: cp.ndarray  # (n_perturbed, nnz) f64 — per-tissue unit-σ contribution
    solver: AMGXSolver  # nominal σ, structure_reuse; mixed-solve fallback
    float_preconditioner: AMGXFloatSolver  # fp32 AMG V-cycle frozen at nominal σ
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


def _reduced_data_for(
    ctx: SolverContext,
    g64: cp.ndarray,
    vols: cp.ndarray,
    cond: cp.ndarray,
    idx: cp.ndarray,
    template: "cp.ndarray | None",
) -> cp.ndarray:
    """Assemble a stiffness with conductivity ``cond``, ground it, and align to the pattern.

    ``template`` is a zero-valued CSR on the reference pattern; adding it forces the reference
    ordering so every component's ``.data`` is index-aligned with ``base_data``/AMGx.
    """
    k = reduce_matrix(assemble_stiffness(g64, vols, cond, ctx.n_nodes, ctx.tet_nodes), idx)
    if template is None:
        return k
    return (template + k).data


def build_conductivity_uq_precompute(
    ctx: SolverContext, perturbed_tags: tuple[int, ...]
) -> ConductivityUQPrecompute:
    """Assemble the reference pattern, per-tissue components, and the nominal-σ AMGx solver."""
    g64, vols = gradient_operator(ctx.nodes_mm * 1e-3, ctx.tet_nodes)
    ground_node = ground_node_of(ctx.nodes_mm)
    idx = grounded_index(ctx.n_nodes, ground_node)

    cond_nom = conductivity_per_tet(ctx.tet_tags)
    k_ref = _reduced_data_for(ctx, g64, vols, cond_nom, idx, template=None)
    zero_ref = k_ref.copy()
    zero_ref.data[:] = 0.0

    perturbed = cp.asarray(perturbed_tags)
    tissue_data = cp.empty((len(perturbed_tags), k_ref.data.shape[0]), dtype=cp.float64)
    for i, tag in enumerate(perturbed_tags):
        indicator = (ctx.tet_tags == tag).astype(cp.float64)
        tissue_data[i] = _reduced_data_for(ctx, g64, vols, indicator, idx, template=zero_ref)

    base_cond = cond_nom.copy()
    base_cond[cp.isin(ctx.tet_tags, perturbed)] = 0.0
    base_data = _reduced_data_for(ctx, g64, vols, base_cond, idx, template=zero_ref)

    nominal_sigma = cp.asarray(
        [TISSUE_CONDUCTIVITY[t] for t in perturbed_tags], dtype=cp.float64
    )

    # Correctness gate: the linear model must reproduce the nominal direct assembly exactly.
    recon = base_data + nominal_sigma @ tissue_data
    rel = float(cp.linalg.norm(recon - k_ref.data) / cp.linalg.norm(k_ref.data))
    if rel > 1e-10:
        raise RuntimeError(f"UQ stiffness decomposition mismatch (rel={rel:.2e})")

    nominal_data = cp.ascontiguousarray(recon)
    row_ptr = cp.ascontiguousarray(k_ref.indptr.astype(cp.int32))
    col_idx = cp.ascontiguousarray(k_ref.indices.astype(cp.int32))
    solver = AMGXSolver(UQ_AMGX_CONFIG)
    solver.setup(row_ptr, col_idx, nominal_data)

    float_preconditioner = AMGXFloatSolver(AMGX_PRECONDITIONER_CONFIG)
    float_preconditioner.setup(
        row_ptr, col_idx, cp.ascontiguousarray(nominal_data.astype(cp.float32))
    )
    pcg = PcgAmgSolver(row_ptr, col_idx, nominal_data)
    tolerance = float(_amgx_config_value(UQ_AMGX_CONFIG, "tolerance", "1e-6"))
    max_iters = int(_amgx_config_value(UQ_AMGX_CONFIG, "max_iters", "2000"))

    return ConductivityUQPrecompute(
        perturbed_tags=perturbed_tags,
        idx=idx,
        n_nodes=ctx.n_nodes,
        indptr=row_ptr,
        indices=col_idx,
        base_data=base_data,
        tissue_data=tissue_data,
        solver=solver,
        float_preconditioner=float_preconditioner,
        pcg=pcg,
        tolerance=tolerance,
        max_iters=max_iters,
        nominal_sigma=nominal_sigma,
        nominal_data=nominal_data,
    )
