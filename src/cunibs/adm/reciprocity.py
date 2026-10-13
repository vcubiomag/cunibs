"""Adjoint (reciprocity) solves that turn the target E-field into a nodal weight field.

For a target functional ``J = Σ_e α_e ê·E_e`` (a volume-weighted ROI average of the E-field
component along ``ê``), the adjoint potential ``λ`` solving ``K λ = c`` (with ``K`` the forward
SPD stiffness and ``c[n] = Σ_e α_e (ê·∇λ_i)``) gives ``J = Σ_e w_e · dadt_elm_e``, where
``w_e = vol_e·σ_e·(G_e λ)`` and ``w_{e∈ROI} -= α_e ê``. Since ``dadt_elm_e`` is the mean of its four
nodal ``dA/dt`` values, this collapses exactly to a nodal weighting ``J = Σ_n W_n · dadt_node_n``
with ``W_n = ¼ Σ_{e∋n} w_e``, evaluable as a node-sourced N-body with no extra discretisation error.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp

from cunibs.adm.grid import Grid, build_grid
from cunibs.adm.target import ResolvedTarget, Target, resolve_target
from cunibs.fem.placement import MU0_OVER_4PI, compute_coil_transforms
from cunibs.fem.solve import (
    GroundedSolver,
    SolverContext,
    _pad_block,
    _solve_grounded_block_mat,
)
from cunibs.solver import dadt_nbody, element_weight, node_gather, node_scatter3

if TYPE_CHECKING:
    import numpy.typing as npt

    from cunibs.coil import Coil

# The adjoint RHS is a near-point source (Green's-function-like, singular at the target), so its
# residual carries far more pointwise error in the functional than the same relative residual does
# for a forward solve. The adjoint solves are one-time (3 per target), so they can afford a
# tighter tolerance than the forward solve.
ADJOINT_TOLERANCE = 1e-9
ADJOINT_MAX_ITERS = 3000


def build_adjoint_solver(
    ctx: SolverContext,
    tolerance: float = ADJOINT_TOLERANCE,
    max_iters: int = ADJOINT_MAX_ITERS,
) -> GroundedSolver:
    """A tight-tolerance grounded solver for the adjoint (reciprocity) solves.

    The adjoint system is the forward one: the context's stiffness is already assembled in
    float64 from float64 gradients and grounds the same DOF, so its operator and hierarchy are
    shared and only the stopping rule differs. Reassembling them gave a bitwise-identical matrix
    for the cost of a second assembly, AMG setup and copy of both on the device.
    """
    return dataclasses.replace(
        ctx.solver,
        tolerance=tolerance,
        max_iters=max_iters,
        last_iterations=0,
        last_relative_residual=0.0,
    )


def ensure_adjoint_solver(
    ctx: SolverContext,
    tolerance: float = ADJOINT_TOLERANCE,
    max_iters: int = ADJOINT_MAX_ITERS,
) -> GroundedSolver:
    """Return ``ctx``'s adjoint solver, building and caching it on first use.

    The solver depends on the mesh and the tolerance, never on the target, so a sweep over targets
    or a second ``optimize`` call on the same subject shares one.
    """
    key = (float(tolerance), int(max_iters))
    cached = ctx.adjoint.get(key)
    if cached is None:
        cached = build_adjoint_solver(ctx, tolerance, max_iters)
        ctx.adjoint[key] = cached
    return cached


def _adjoint_rhs(
    ctx: SolverContext, elem_idx: cp.ndarray, weights: cp.ndarray, direction: cp.ndarray
) -> cp.ndarray:
    """Assemble the full-size nodal adjoint RHS ``c`` for one direction ``ê``.

    The ROI's corner contributions ``α_e (ê·∇λ_i)`` are laid over the whole mesh so the nodal sum
    is the same fixed-order gather over the node2corner CSR that every other assembly uses. A
    scatter straight from the ROI would need atomics, and would sum a node's corners in whatever
    order the blocks retired.
    """
    corner = cp.zeros(4 * int(ctx.tet_nodes.shape[0]), dtype=cp.float64)
    corner.reshape(-1, 4)[elem_idx] = weights[:, None] * (
        ctx.g[elem_idx].astype(cp.float64) @ direction
    )
    c = cp.empty(ctx.n_nodes, dtype=cp.float64)
    node_gather(
        corner,
        ctx.node2corner_ptr,
        ctx.node2corner_idx,
        c,
        cp.cuda.get_current_stream().ptr,
    )
    return c


def _node_weights(
    ctx: SolverContext,
    lam: cp.ndarray,
    elem_idx: cp.ndarray,
    weights: cp.ndarray,
    direction: cp.ndarray,
) -> cp.ndarray:
    """Nodal reciprocity weights ``W_n`` for one adjoint solution ``λ``."""
    stream = cp.cuda.get_current_stream().ptr
    # w_e = vol_e·σ_e·(G_e λ);  ctx.neg_vc = -vol_e·σ_e.
    w_e = cp.empty((int(ctx.tet_nodes.shape[0]), 3), dtype=cp.float64)
    element_weight(
        cp.ascontiguousarray(lam, dtype=cp.float64),
        ctx.tet_nodes,
        ctx.g,
        ctx.neg_vc,
        w_e,
        stream,
    )
    # Fold the direct -ê·dA/dt_{ROI} term into the ROI element weights.
    w_e[elem_idx] -= weights[:, None] * direction[None, :]

    # W_n = ¼ Σ_{e∋n} w_e, the transpose of the node2corner reduction.
    node_w = cp.empty((ctx.n_nodes, 3), dtype=cp.float64)
    node_scatter3(w_e, ctx.node2corner_ptr, ctx.node2corner_idx, node_w, stream)
    return node_w


@dataclass
class AdjointField:
    """Nodal reciprocity weights for each adjoint direction (pre-grid)."""

    ctx: SolverContext
    target: ResolvedTarget
    node_weights: cp.ndarray  # (D, n_nodes, 3) float64


def solve_adjoint(
    ctx: SolverContext, target: ResolvedTarget, solver: GroundedSolver | None = None
) -> AdjointField:
    """Run the adjoint solves, every direction in one block, and return the nodal weights ``W_n``.

    The directions share the operator, so they go through the block solver as columns sharing each
    stiffness and hierarchy read. A column's result does not depend on the block it was solved in,
    so this is the same answer one solve per direction gives. ``solver`` defaults to the context's
    cached tight-tolerance solver (:func:`ensure_adjoint_solver`), so targets on the same mesh
    share one setup.
    """
    if solver is None:
        solver = ensure_adjoint_solver(ctx)
    directions = target.directions
    n_dir = int(directions.shape[0])
    rhs = [
        _adjoint_rhs(ctx, target.elem_idx, target.weights, directions[d]) for d in range(n_dir)
    ]
    lams = [cp.zeros(ctx.n_nodes, dtype=cp.float64) for _ in range(n_dir)]
    reduced = cp.ascontiguousarray(cp.stack(rhs, axis=1)[solver.idx])
    # The block solver refuses an all-zero column; a direction the target cannot see solves to
    # zero, which is what the single-RHS path returns for it.
    live = [d for d in range(n_dir) if bool(cp.any(reduced[:, d]))]
    if live:
        x, _ = _solve_grounded_block_mat(
            solver, _pad_block(cp.ascontiguousarray(reduced[:, live]), len(live)), len(live)
        )
        for j, d in enumerate(live):
            lams[d][solver.idx] = x[:, j]
    node_w = cp.empty((n_dir, ctx.n_nodes, 3), dtype=cp.float64)
    for d in range(n_dir):
        node_w[d] = _node_weights(ctx, lams[d], target.elem_idx, target.weights, directions[d])
    return AdjointField(ctx=ctx, target=target, node_weights=node_w)


def exact_functional(node_weights: cp.ndarray, dadt_nodes: cp.ndarray) -> cp.ndarray:
    """Exact reciprocity functional ``J_d = Σ_n W_{d,n} · dadt_node_n`` (validation path).

    ``dadt_nodes`` is the nodal ``dA/dt`` field from ``coil_dadt_at_nodes`` (already scaled by
    ``didt``), identical to the array the forward solve consumes.
    """
    dn = dadt_nodes.astype(cp.float64)
    return cp.einsum("dnk,nk->d", node_weights, dn)


@dataclass
class ReciprocityField:
    """Adjoint weights sampled as a Q-field on a regular grid, the input to the placement evaluator.

    ``q`` holds, per adjoint direction, the reciprocity field
    ``Q_d(g) = (μ0/4π) Σ_n W_{d,n} × (g - r_n) / |g - r_n|³`` so that the target E-field component
    of any placement is ``E_d = didt · Σ_j m_j · Q_d(s_j)`` (``s_j``, ``m_j`` = placed coil dipoles).
    """

    ctx: SolverContext
    target: ResolvedTarget
    grid: Grid
    q: cp.ndarray  # (D, nx, ny, nz, 3) float32
    magnitude: bool


def sample_qfield(adjoint: AdjointField, grid: Grid) -> ReciprocityField:
    """Evaluate the reciprocity Q-field on ``grid`` via the node-sourced dA/dt N-body kernel.

    ``Q_d = dadt_nbody(sources=mesh nodes, moments=W_{d,n}, targets=grid, didt=1)`` uses the same
    kernel and formula as the forward coil field, with the roles of coil and mesh swapped.
    """
    ctx = adjoint.ctx
    center = grid.center_m
    s = cp.ascontiguousarray(
        ((cp.asarray(ctx.nodes_mm) * 1e-3) - center[None, :]).astype(cp.float32)
    )
    sn = cp.ascontiguousarray((s * s).sum(1))
    r = cp.ascontiguousarray((grid.points_m() - center[None, :]).astype(cp.float32))

    n_dir = int(adjoint.node_weights.shape[0])
    q = cp.empty((n_dir, *grid.shape, 3), dtype=cp.float32)
    out = cp.empty((grid.n_points, 3), dtype=cp.float32)
    stream = cp.cuda.get_current_stream().ptr
    for d in range(n_dir):
        w = adjoint.node_weights[d].astype(cp.float32)  # (n_nodes, 3)
        mp = cp.ascontiguousarray(cp.concatenate([w, cp.cross(w, s)], axis=1))
        dadt_nbody(s, mp, sn, r, out, 1.0, MU0_OVER_4PI, stream)
        q[d] = out.reshape((*grid.shape, 3))
    return ReciprocityField(
        ctx=ctx,
        target=adjoint.target,
        grid=grid,
        q=q,
        magnitude=adjoint.target.magnitude,
    )


def _coverage_grid(
    ctx: SolverContext,
    coil: Coil,
    centers_mm: npt.ArrayLike,
    distance_mm: float,
    spacing_mm: float,
    margin_mm: float,
) -> Grid:
    """Grid covering every coil dipole for the coil placed at any rotation on each center."""
    centers = cp.ascontiguousarray(cp.asarray(centers_mm, dtype=cp.float64).reshape(-1, 3))
    # No handle: only the translation is read, and the reach dilation below is rotation-invariant.
    base = compute_coil_transforms(ctx, centers, None, cp.full(centers.shape[0], distance_mm))
    origins_m = base[:, :3, 3] * 1e-3
    # A dipole stays within max|position| of the coil origin at any rotation; dilate by that + margin.
    reach_mm = float(cp.linalg.norm(cp.asarray(coil.positions_m), axis=1).max()) * 1e3
    return build_grid(origins_m, spacing_mm=spacing_mm, margin_mm=reach_mm + margin_mm)


def build_reciprocity(
    ctx: SolverContext,
    coil: Coil,
    target: Target,
    coverage_centers_mm: npt.ArrayLike,
    *,
    distance_mm: float = 4.0,
    spacing_mm: float = 3.0,
    margin_mm: float = 8.0,
    adjoint_solver: GroundedSolver | None = None,
) -> ReciprocityField:
    """Build the reciprocity field: adjoint solves plus a Q-field grid covering the candidates.

    ``coverage_centers_mm`` are the scalp positions the coil may occupy; the grid is sized to contain
    every coil dipole over those positions at any rotation.
    """
    resolved = resolve_target(ctx, target)
    if adjoint_solver is None:
        adjoint_solver = ensure_adjoint_solver(ctx)
    adjoint = solve_adjoint(ctx, resolved, adjoint_solver)
    grid = _coverage_grid(ctx, coil, coverage_centers_mm, distance_mm, spacing_mm, margin_mm)
    return sample_qfield(adjoint, grid)
