"""Precompute the per-tissue stiffness components that make each MC sample a cheap value swap.

K is linear in the per-tissue conductivities: ``K(σ).data = K_base.data + Σ_t σ_t·Kt.data`` on a
fixed CSR sparsity pattern. Assembling each tissue's geometric contribution once turns the
per-sample matrix build into a single GEMV over the nonzeros, and lets the solver swap
coefficients without re-analysing structure.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import numpy.typing as npt

from cunibs.fem.assembly import (
    TISSUE_CONDUCTIVITY,
    as_int32_tets,
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
    reduce_matrix,
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
    affine_has_base: bool  # whether pcg's affine parts lead with the fixed base
    # The perturbed tissues' parts as their own CSR matrices with the pattern's zeros dropped, so a
    # product with one part streams only that tissue's entries. The warm start forms many.
    tissue_ops: list[csp.csr_matrix]
    base_op: csp.csr_matrix | None  # the fixed part, or None when every tissue is perturbed
    # Held for a whole placement: it resets pcg's values to nominal and a fallback draw swaps in
    # its own, so two placements sharing this state must not interleave.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def affine_coefficients(self, sigmas: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Host ``(k, n_terms)`` coefficients of the affine operator for draws ``sigmas``."""
        sig = np.asarray(cp.asnumpy(sigmas), dtype=np.float64)
        if not self.affine_has_base:
            return np.ascontiguousarray(sig)
        return np.ascontiguousarray(np.hstack([np.ones((sig.shape[0], 1)), sig]))

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
            sample_data.astype(cp.float32),
        )


def build_conductivity_uq_precompute(
    ctx: SolverContext, perturbed_tags: tuple[int, ...]
) -> ConductivityUQPrecompute:
    """Assemble the reference pattern, per-tissue components, and the nominal-σ preconditioner."""
    g64, vols = gradient_operator(ctx.nodes_mm, ctx.tet_nodes)
    # The forward solver's ordering, so the reduced pattern is the one its hierarchy was built on.
    idx = ctx.solver.idx

    tets = as_int32_tets(ctx.tet_nodes)
    n_nodes = int(ctx.n_nodes)
    n2c_ptr, n2c_idx = ctx.node2corner_ptr, ctx.node2corner_idx
    a_full = stiffness_pattern(tets, n_nodes, n2c_ptr, n2c_idx)

    # Grounding selects a submatrix of a pattern that never moves, so every reduced nonzero comes
    # from exactly one full nonzero. Reducing a matrix whose values are their own positions
    # recovers that map once, which turns every component into a refill plus a gather.
    a_full.data[:] = cp.arange(a_full.nnz, dtype=g64.dtype)
    k_ref = reduce_matrix(a_full, idx)
    gather = k_ref.data.astype(cp.int32)
    row_ptr = k_ref.indptr.astype(cp.int32)
    col_idx = k_ref.indices.astype(cp.int32)
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

    # Nominal sigma is the forward conductivity table, so the nominal operator is the forward one
    # and its hierarchy serves as is.
    precond = ctx.solver.precond
    stream = cp.cuda.get_current_stream().ptr
    pcg = PcgAmgSolver(row_ptr, col_idx, nominal_data, stream)
    # The draws' operators, A(sigma) = base + sum_t sigma_t A_t, as one affine operator: a block of
    # draws then shares every part's read and no draw's matrix is ever assembled. The base is a
    # part whose coefficient is always 1, and is left out when nothing is held fixed.
    has_base = bool(cp.any(base_data))
    terms = cp.concatenate([base_data[None], tissue_data]) if has_base else tissue_data
    pcg.set_affine_terms(cp.ascontiguousarray(terms), stream)
    del terms
    tolerance = DEFAULT_TOLERANCE
    max_iters = DEFAULT_MAX_ITERS

    shape = (int(idx.shape[0]), int(idx.shape[0]))

    rows = cp.repeat(cp.arange(shape[0], dtype=cp.int32), cp.diff(row_ptr).astype(cp.int64))

    def compressed(data: cp.ndarray) -> csp.csr_matrix:
        keep = data != 0.0
        counts = cp.bincount(rows[keep], minlength=shape[0])
        ptr = cp.concatenate([cp.zeros(1, dtype=cp.int32), cp.cumsum(counts).astype(cp.int32)])
        return csp.csr_matrix((data[keep], col_idx[keep], ptr), shape=shape)

    tissue_ops = [compressed(tissue_data[i]) for i in range(len(perturbed_tags))]
    base_op = compressed(base_data) if has_base else None

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
        affine_has_base=has_base,
        tissue_ops=tissue_ops,
        base_op=base_op,
    )
