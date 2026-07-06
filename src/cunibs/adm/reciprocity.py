"""Adjoint (reciprocity) solves that turn the target E-field into a nodal weight field.

For a target functional ``J = Σ_e α_e ê·E_e`` (a volume-weighted ROI average of the E-field
component along ``ê``), the adjoint potential ``λ`` solving ``K λ = c`` (with ``K`` the forward
SPD stiffness and ``c[n] = Σ_e α_e (ê·∇λ_i)``) gives ``J = Σ_e w_e · dadt_elm_e``, where
``w_e = vol_e·σ_e·(G_e λ)`` and ``w_{e∈ROI} -= α_e ê``. Since ``dadt_elm_e`` is the mean of its four
nodal ``dA/dt`` values, this collapses exactly to a nodal weighting ``J = Σ_n W_n · dadt_node_n``
with ``W_n = ¼ Σ_{e∋n} w_e``, evaluable as a node-sourced N-body with no extra discretisation error.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy.typing as npt

from cunibs.adm.grid import Grid, build_grid
from cunibs.adm.place import compute_coil_transforms
from cunibs.adm.target import ResolvedTarget, Target, resolve_target
from cunibs.coil import Coil
from cunibs.fem.assembly import (
    GRADIENT_TILE_TETS,
    assemble_stiffness,
    conductivity_per_tet,
    gradient_operator,
)
from cunibs.fem.placement import MU0_OVER_4PI
from cunibs.fem.solve import (
    AMGX_CONFIG,
    GroundedSolver,
    SolverContext,
    prepare_grounded_solver,
    solve_grounded,
)
from cunibs.solver import dadt_nbody

# The adjoint RHS is a near-point source (Green's-function-like, singular at the target). At the
# forward tolerance (1e-6) its residual leaves ~1% pointwise error in the functional, well above the
# field budget. The adjoint solves are one-time (3 per target), so they use a tighter tolerance.
ADJOINT_AMGX_CONFIG = AMGX_CONFIG.replace("tolerance=1e-6", "tolerance=1e-9").replace(
    "max_iters=2000", "max_iters=3000"
)


def build_adjoint_solver(
    ctx: SolverContext, config: str = ADJOINT_AMGX_CONFIG
) -> GroundedSolver:
    """Assemble a dedicated tight-tolerance grounded solver for the adjoint (reciprocity) solves.

    Reassembles the stiffness in float64 (``ctx.g`` is float32) and grounds the same DOF the forward
    solver uses, so the adjoint is consistent with the forward system.
    """
    g64, vols = gradient_operator(ctx.nodes_mm * 1e-3, ctx.tet_nodes)
    cond = conductivity_per_tet(ctx.tet_tags)
    stiffness = assemble_stiffness(g64, vols, cond, ctx.n_nodes, ctx.tet_nodes)
    ground_node = int(cp.argmin(ctx.nodes_mm[:, 2]))
    solver = prepare_grounded_solver(stiffness, ground_node, config)
    del stiffness, g64, vols, cond
    return solver


def _element_gradient(values: cp.ndarray, tet_nodes: cp.ndarray, g: cp.ndarray) -> cp.ndarray:
    """Per-element ``Σ_i values[node_i]·g[e,i]`` in float64 (``g`` may be float32)."""
    n_tet = int(tet_nodes.shape[0])
    out = cp.empty((n_tet, 3), dtype=cp.float64)
    for lo in range(0, n_tet, GRADIENT_TILE_TETS):
        hi = min(lo + GRADIENT_TILE_TETS, n_tet)
        vt = values[tet_nodes[lo:hi]]  # (t,4) float64
        gt = g[lo:hi].astype(cp.float64)  # (t,4,3)
        out[lo:hi] = cp.einsum("ti,tik->tk", vt, gt)
    return out


def _adjoint_rhs(
    ctx: SolverContext, elem_idx: cp.ndarray, weights: cp.ndarray, direction: cp.ndarray
) -> cp.ndarray:
    """Assemble the full-size nodal adjoint RHS ``c`` for one direction ``ê``."""
    g_roi = ctx.g[elem_idx].astype(cp.float64)  # (K,4,3)
    nodes = ctx.tet_nodes[elem_idx]  # (K,4)
    # α_e (ê·∇λ_i) per ROI corner.
    contrib = weights[:, None] * (g_roi @ direction)  # (K,4)
    c = cp.zeros(ctx.n_nodes, dtype=cp.float64)
    cp.add.at(c, nodes.ravel(), contrib.ravel())
    return c


def _node_weights(
    ctx: SolverContext,
    lam: cp.ndarray,
    elem_idx: cp.ndarray,
    weights: cp.ndarray,
    direction: cp.ndarray,
) -> cp.ndarray:
    """Nodal reciprocity weights ``W_n`` for one adjoint solution ``λ``."""
    grad = _element_gradient(lam, ctx.tet_nodes, ctx.g)  # (n_tet,3), = G_e λ
    # w_e = vol_e·σ_e·(G_e λ);  ctx.neg_vc = -vol_e·σ_e.
    w_e = (-ctx.neg_vc.astype(cp.float64))[:, None] * grad
    # Fold the direct -ê·dA/dt_{ROI} term into the ROI element weights.
    w_e[elem_idx] -= weights[:, None] * direction[None, :]

    node_w = cp.zeros((ctx.n_nodes, 3), dtype=cp.float64)
    rows = ctx.tet_nodes.ravel()
    for comp in range(3):
        cp.add.at(node_w[:, comp], rows, cp.repeat(w_e[:, comp], 4))
    node_w *= 0.25
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
    """Run one adjoint solve per direction and return the nodal weight fields ``W_n``.

    ``solver`` defaults to a freshly built tight-tolerance solver (:func:`build_adjoint_solver`);
    pass a cached one to amortise its setup across targets on the same mesh.
    """
    if solver is None:
        solver = build_adjoint_solver(ctx)
    directions = target.directions
    node_w = cp.empty((directions.shape[0], ctx.n_nodes, 3), dtype=cp.float64)
    for d in range(directions.shape[0]):
        ehat = directions[d]
        c = _adjoint_rhs(ctx, target.elem_idx, target.weights, ehat)
        lam = solve_grounded(solver, c)
        node_w[d] = _node_weights(ctx, lam, target.elem_idx, target.weights, ehat)
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
    center_m: cp.ndarray  # (3,) float64, translation used for the float32 N-body
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
    q = cp.empty((n_dir,) + grid.shape + (3,), dtype=cp.float32)
    out = cp.empty((grid.n_points, 3), dtype=cp.float32)
    stream = cp.cuda.get_current_stream().ptr
    for d in range(n_dir):
        w = adjoint.node_weights[d].astype(cp.float32)  # (n_nodes, 3)
        mp = cp.ascontiguousarray(cp.concatenate([w, cp.cross(w, s)], axis=1))
        dadt_nbody(s, mp, sn, r, out, 1.0, MU0_OVER_4PI, stream)
        q[d] = out.reshape(grid.shape + (3,))
    return ReciprocityField(
        ctx=ctx,
        target=adjoint.target,
        grid=grid,
        q=q,
        center_m=cp.ascontiguousarray(center),
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
    base = compute_coil_transforms(
        ctx,
        centers,
        centers + cp.asarray([1.0, 0.0, 0.0]),
        cp.full(centers.shape[0], distance_mm),
    )
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
    spacing_mm: float = 2.0,
    margin_mm: float = 8.0,
    adjoint_solver: GroundedSolver | None = None,
) -> ReciprocityField:
    """Build the reciprocity field: adjoint solves plus a Q-field grid covering the candidates.

    ``coverage_centers_mm`` are the scalp positions the coil may occupy; the grid is sized to contain
    every coil dipole over those positions at any rotation.
    """
    resolved = resolve_target(ctx, target)
    if adjoint_solver is None:
        adjoint_solver = build_adjoint_solver(ctx)
    adjoint = solve_adjoint(ctx, resolved, adjoint_solver)
    grid = _coverage_grid(ctx, coil, coverage_centers_mm, distance_mm, spacing_mm, margin_mm)
    return sample_qfield(adjoint, grid)
