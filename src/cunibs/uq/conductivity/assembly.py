"""Precompute the per-tissue stiffness components that make each MC sample a cheap value swap.

K is linear in the per-tissue conductivities: ``K(σ).data = K_base.data + Σ_t σ_t·Kt.data`` on a
fixed CSR sparsity pattern. Assembling each tissue's geometric contribution once lets the solver
apply every draw's operator from those parts, with no draw's matrix ever assembled.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs._memory import build_arena
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
    GroundedSolver,
    SolverContext,
    build_native_vcycle,
    reduce_matrix,
)
from cunibs.solver import AffineTerms, NativeVCycle


@dataclass
class ConductivityUQPrecompute:
    """Reusable state for conductivity UQ on one mesh, amortised across placements/samples.

    The nominal operator is the forward one, so the precompute solves on the forward solver: its
    matrix, hierarchy and buffers. What it adds is the draws' operator ``A(sigma)`` as ``terms``,
    an affine operator on that solver, so a block of draws is solved without any draw's matrix
    being assembled.
    """

    perturbed_tags: tuple[int, ...]
    solver: GroundedSolver  # the forward solver, whose ordering and hierarchy every draw uses
    terms: AffineTerms  # the parts of A(sigma) on solver.pcg, base first when there is one
    tolerance: float
    max_iters: int
    nominal_sigma: cp.ndarray  # (n_perturbed,) f64
    affine_has_base: (
        bool  # whether terms lead with the fixed base, absent when all are perturbed
    )

    @property
    def idx(self) -> cp.ndarray:
        """The grounded row/col index, which drops the ground DOF in Morton order."""
        return self.solver.idx

    def affine_coefficients(self, sigmas: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Host ``(k, n_terms)`` coefficients of the affine operator for draws ``sigmas``."""
        sig = np.asarray(cp.asnumpy(sigmas), dtype=np.float64).reshape(
            -1, len(self.perturbed_tags)
        )
        if not self.affine_has_base:
            return np.ascontiguousarray(sig)
        return np.ascontiguousarray(np.hstack([np.ones((sig.shape[0], 1)), sig]))

    def apply_part(self, x: cp.ndarray, tissue: int | None) -> cp.ndarray:
        """``A_t x`` for a dense ``x`` of one or two dimensions, over one part of the stiffness.

        The part is perturbed tissue ``tissue``, an index into ``perturbed_tags``, at unit
        conductivity, or the fixed base for ``None``. Each row is summed over the part's own
        nonzeros in column order, as :func:`~cunibs.fem.assembly.csr_matmul` sums a CSR, so a
        product with one part streams only that tissue's entries.
        """
        if tissue is None and not self.affine_has_base:
            raise ValueError("every tissue is perturbed, so there is no fixed base part")
        term = 0 if tissue is None else tissue + int(self.affine_has_base)
        x2 = cp.ascontiguousarray(x.reshape(x.shape[0], -1), dtype=cp.float64)
        y = cp.empty_like(x2)
        self.terms.matmul(term, x2, y, cp.cuda.get_current_stream().ptr)
        return y.reshape(x.shape)

    def values(self, sigma: npt.ArrayLike) -> cp.ndarray:
        """The reduced matrix values ``A(sigma)`` for one draw, in the solver's entry order."""
        out = cp.empty(self.solver.pcg.nnz, dtype=cp.float64)
        self.terms.values(
            self.affine_coefficients(sigma)[0], out, cp.cuda.get_current_stream().ptr
        )
        return out

    def preconditioner_for(self, sigma: npt.ArrayLike) -> NativeVCycle:
        """Build a throwaway V-cycle matched to one draw's conductivities.

        Only for a draw the nominal-σ preconditioner fails to converge. The result is not
        cached: draws are i.i.d. around nominal, so an extreme draw's hierarchy is no better a
        default for the next draw than the nominal one.
        """
        with build_arena():
            pattern = self.solver.matrix(values=False)
            sample = self.values(sigma).astype(cp.float32)
            return build_native_vcycle(pattern.indptr, pattern.indices, sample)


def _reduced_parts(
    ctx: SolverContext, perturbed_tags: tuple[int, ...]
) -> tuple[cp.ndarray, cp.ndarray]:
    """``(parts, nominal)``: the reduced stiffness split by tissue.

    ``parts`` is ``(1 + P, nnz)``, the base (the tissues held fixed, at nominal σ) then each
    perturbed tissue at unit σ; ``nominal`` is the nominal stiffness assembled directly, which
    the split has to reproduce. Values are in the forward solver's entry order.
    """
    g64, vols = gradient_operator(ctx.nodes_mm, ctx.tet_nodes)
    tets = as_int32_tets(ctx.tet_nodes)
    n2c_ptr, n2c_idx = ctx.node2corner_ptr, ctx.node2corner_idx
    a_full = stiffness_pattern(tets, int(ctx.n_nodes), n2c_ptr, n2c_idx)

    # Grounding selects a submatrix of a pattern that never moves, so every reduced nonzero comes
    # from exactly one full nonzero. Reducing a matrix whose values are their own positions
    # recovers that map once, which turns every component into a refill plus a gather. The
    # forward solver was reduced with the same index, so this is its pattern.
    a_full.data[:] = cp.arange(a_full.nnz, dtype=g64.dtype)
    k_ref = reduce_matrix(a_full, ctx.solver.idx)
    gather = k_ref.data.astype(cp.int32)
    del k_ref

    def reduced_data_for(cond: cp.ndarray, out: cp.ndarray | None = None) -> cp.ndarray:
        fill_stiffness_values(a_full, g64, vols, cond, tets, n2c_ptr, n2c_idx)
        return cp.take(a_full.data, gather, out=out)

    # Every component is assembled over the whole mesh with zero conductivity outside its own
    # tets: the zeros keep the reference pattern and contribute nothing to its values.
    parts = cp.empty((1 + len(perturbed_tags), int(gather.shape[0])), dtype=cp.float64)
    for i, tag in enumerate(perturbed_tags):
        reduced_data_for((ctx.tet_tags == tag).astype(cp.float64), out=parts[1 + i])
    cond_nom = conductivity_per_tet(ctx.tet_tags)
    perturbed_mask = cp.isin(ctx.tet_tags, cp.asarray(perturbed_tags))
    reduced_data_for(cp.where(perturbed_mask, 0.0, cond_nom), out=parts[0])
    return parts, reduced_data_for(cond_nom)


def build_conductivity_uq_precompute(
    ctx: SolverContext, perturbed_tags: tuple[int, ...]
) -> ConductivityUQPrecompute:
    """Assemble the per-tissue stiffness components as affine terms on the forward solver."""
    solver = ctx.solver
    stream = cp.cuda.get_current_stream().ptr
    with build_arena() as arena:
        parts, nominal_direct = _reduced_parts(ctx, perturbed_tags)
        # Row 0 is the base, so the parts handed to the solver are one contiguous block either
        # way.
        base_data, tissue_data = parts[0], parts[1:]

        with arena.resident():
            nominal_sigma = cp.asarray(
                [TISSUE_CONDUCTIVITY[t] for t in perturbed_tags], dtype=cp.float64
            )

        # Correctness gate: the linear model must reproduce the nominal direct assembly exactly.
        recon = base_data + nominal_sigma @ tissue_data
        rel = float(cp.linalg.norm(recon - nominal_direct) / cp.linalg.norm(nominal_direct))
        if rel > 1e-10:
            raise RuntimeError(f"UQ stiffness decomposition mismatch (rel={rel:.2e})")
        del recon, nominal_direct

        # The base is a part whose coefficient is always 1, and is left out when nothing is held
        # fixed.
        has_base = bool(cp.any(base_data))
        terms = AffineTerms(solver.pcg, parts if has_base else tissue_data, stream)

    return ConductivityUQPrecompute(
        perturbed_tags=perturbed_tags,
        solver=solver,
        terms=terms,
        tolerance=DEFAULT_TOLERANCE,
        max_iters=DEFAULT_MAX_ITERS,
        nominal_sigma=nominal_sigma,
        affine_has_base=has_base,
    )
