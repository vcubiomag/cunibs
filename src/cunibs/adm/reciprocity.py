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
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from cunibs.adm.grid import Grid, SweptCoverage, build_grid
from cunibs.adm.target import ResolvedTarget, Target, resolve_target
from cunibs.fem.placement import MU0_OVER_4PI, compute_coil_transforms
from cunibs.fem.solve import (
    GroundedSolver,
    SolverContext,
    _pad_block,
    _solve_grounded_block_mat,
    context_of,
)
from cunibs.solver import (
    adm_reach,
    dadt_nbody_multi,
    element_weight,
    node_gather,
    node_scatter3,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from cunibs.coil import Coil
    from cunibs.simulation import Subject

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
    It is laid out ``(nx, ny, nz, D, 3)`` so that one grid point's every direction is one read.

    Only the points a coil can reach are sampled when the grid carries a coverage; ``sampled``
    marks them, and ``cell_ok`` marks the cells whose eight corners all are, which is the only
    place interpolation is allowed to read. A frame that lands elsewhere has its points sampled on
    demand from the kept sources, so the coverage decides what is paid up front and never what a
    frame may ask for.
    """

    ctx: SolverContext
    target: ResolvedTarget
    grid: Grid
    q: cp.ndarray  # (nx, ny, nz, D, 3) float32
    magnitude: bool
    sampled: cp.ndarray  # (nx * ny * nz,) uint8
    cell_ok: cp.ndarray  # ((nx-1) * (ny-1) * (nz-1),) uint8
    # The N-body sources, centred on the grid, for sampling further points on demand.
    src: cp.ndarray  # (n_nodes, 3) float32
    src_sq: cp.ndarray  # (n_nodes,) float32
    moments: cp.ndarray  # (D, n_nodes, 6) float32: W_d and W_d x s
    # The coil's offset from the scalp and the candidate centres the coverage was built for, or
    # None for a field not built by build_reciprocity. optimize_placement defaults to both.
    distance_mm: float | None = None
    centers_mm: npt.NDArray[np.float64] | None = None

    @property
    def n_directions(self) -> int:
        return int(self.q.shape[3])

    def sample_points(self, points: cp.ndarray) -> None:
        """Sample the field at the flat grid indices ``points`` and mark them sampled."""
        if points.size == 0:
            return
        grid = self.grid
        flat = grid_points_m(grid, points) - grid.center_m[None, :]
        r = cp.ascontiguousarray(flat.astype(cp.float32))
        out = cp.empty((int(points.size), self.n_directions, 3), dtype=cp.float32)
        dadt_nbody_multi(
            self.src,
            self.moments,
            self.src_sq,
            r,
            out,
            1.0,
            MU0_OVER_4PI,
            cp.cuda.get_current_stream().ptr,
        )
        self.q.reshape(-1, self.n_directions, 3)[points] = out
        self.sampled[points] = 1
        self.cell_ok = _cells_ok(self.sampled, grid.shape)


def grid_points_m(grid: Grid, points: cp.ndarray) -> cp.ndarray:
    """Coordinates ``(K, 3)`` of the flat C-order grid indices ``points``."""
    _, ny, nz = grid.shape
    ix = points // (ny * nz)
    iy = (points // nz) % ny
    iz = points % nz
    idx = cp.stack([ix, iy, iz], axis=1).astype(cp.float64)
    return grid.origin_m[None, :] + grid.spacing_m[None, :] * idx


def _cells_ok(sampled: cp.ndarray, shape: tuple[int, int, int]) -> cp.ndarray:
    """One byte per cell: whether all eight of its corners are sampled."""
    s = sampled.reshape(shape).astype(cp.bool_)
    ok = s[:-1, :-1, :-1].copy()
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ok &= s[dx : dx + shape[0] - 1, dy : dy + shape[1] - 1, dz : dz + shape[2] - 1]
    return cp.ascontiguousarray(ok.astype(cp.uint8).ravel())


def _reachable(grid: Grid, coverage: SweptCoverage) -> cp.ndarray:
    """One byte per grid point: whether the coverage's swept region reaches it.

    The coil's ``(rho, h)`` profile is rasterised once at a quarter of the grid spacing, each cell
    marked when its centre lies within the margin of some dipole plus the raster's own half
    diagonal, so the raster never misses a point the margin covers. Every grid point then looks
    its cylindrical coordinates up against each centre's axis.
    """
    cell = float(cp.min(grid.spacing_m)) / 4.0
    margin = coverage.margin_m
    prof = coverage.profile_m
    rho_hi = float(prof[:, 0].max()) + margin + cell
    h_lo = float(prof[:, 1].min()) - margin - cell
    h_hi = float(prof[:, 1].max()) + margin + cell
    n_rho = int(cp.ceil(rho_hi / cell)) + 1
    n_h = int(cp.ceil((h_hi - h_lo) / cell)) + 1
    rc = (cp.arange(n_rho, dtype=cp.float64) + 0.5) * cell
    hc = h_lo + (cp.arange(n_h, dtype=cp.float64) + 0.5) * cell
    reach_sq = (margin + cell * 0.5 * 2.0**0.5) ** 2
    profile = cp.zeros((n_rho, n_h), dtype=cp.uint8)
    # Rows at a time keeps the (rows * n_h * N) distance block to a few hundred MB.
    rows = max(1, (1 << 24) // max(1, n_h * int(prof.shape[0])))
    for lo in range(0, n_rho, rows):
        r = rc[lo : lo + rows, None, None]
        d2 = (r - prof[None, None, :, 0]) ** 2 + (
            hc[None, :, None] - prof[None, None, :, 1]
        ) ** 2
        profile[lo : lo + rows] = (d2.min(axis=2) <= reach_sq).astype(cp.uint8)

    mask = cp.empty(grid.n_points, dtype=cp.uint8)
    adm_reach(
        [float(x) for x in cp.asnumpy(grid.origin_m)],
        [float(x) for x in cp.asnumpy(grid.spacing_m)],
        list(grid.shape),
        cp.ascontiguousarray(coverage.origins_m, dtype=cp.float64),
        cp.ascontiguousarray(coverage.axes, dtype=cp.float64),
        cp.ascontiguousarray(profile),
        h_lo,
        cell,
        mask,
        cp.cuda.get_current_stream().ptr,
    )
    return mask


def sample_qfield(adjoint: AdjointField, grid: Grid) -> ReciprocityField:
    """Evaluate the reciprocity Q-field on ``grid`` via the node-sourced dA/dt N-body kernel.

    ``Q_d = dadt_nbody(sources=mesh nodes, moments=W_{d,n}, targets=grid, didt=1)`` uses the same
    kernel and formula as the forward coil field, with the roles of coil and mesh swapped, and all
    directions in one pass over the sources. With a coverage on the grid only the reachable points
    are sampled; without one, all of them.
    """
    ctx = adjoint.ctx
    center = grid.center_m
    s = cp.ascontiguousarray(
        ((cp.asarray(ctx.nodes_mm) * 1e-3) - center[None, :]).astype(cp.float32)
    )
    sn = cp.ascontiguousarray((s * s).sum(1))
    n_dir = int(adjoint.node_weights.shape[0])
    w = adjoint.node_weights.astype(cp.float32)  # (D, n_nodes, 3)
    moments = cp.ascontiguousarray(cp.concatenate([w, cp.cross(w, s[None])], axis=2))

    field = ReciprocityField(
        ctx=ctx,
        target=adjoint.target,
        grid=grid,
        q=cp.zeros((*grid.shape, n_dir, 3), dtype=cp.float32),
        magnitude=adjoint.target.magnitude,
        sampled=cp.zeros(grid.n_points, dtype=cp.uint8),
        cell_ok=cp.zeros(0, dtype=cp.uint8),
        src=s,
        src_sq=sn,
        moments=moments,
    )
    if grid.coverage is None:
        points = cp.arange(grid.n_points, dtype=cp.int64)
    else:
        points = cp.flatnonzero(_reachable(grid, grid.coverage))
    field.sample_points(points)
    return field


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
    pos = cp.asarray(coil.positions_m, dtype=cp.float64)
    reach_mm = float(cp.linalg.norm(pos, axis=1).max()) * 1e3
    grid = build_grid(origins_m, spacing_mm=spacing_mm, margin_mm=reach_mm + margin_mm)
    grid.coverage = SweptCoverage(
        origins_m=cp.ascontiguousarray(origins_m),
        axes=cp.ascontiguousarray(base[:, :3, 2]),
        profile_m=cp.ascontiguousarray(
            cp.stack([cp.linalg.norm(pos[:, :2], axis=1), pos[:, 2]], axis=1)
        ),
        margin_m=margin_mm * 1e-3,
    )
    return grid


def build_reciprocity(
    subject: Subject | SolverContext,
    coil: Coil,
    target: Target | ResolvedTarget,
    centers_mm: npt.ArrayLike,
    *,
    distance_mm: float = 4.0,
    spacing_mm: float = 3.0,
    margin_mm: float = 8.0,
    tolerance: float = ADJOINT_TOLERANCE,
) -> ReciprocityField:
    """Build the reciprocity field: adjoint solves plus a Q-field grid covering the candidates.

    ``centers_mm`` are the scalp positions the coil may occupy, at ``distance_mm`` from the scalp;
    the grid is sized to contain every coil dipole over those positions at any rotation, and the
    field remembers both so :func:`~cunibs.adm.optimize_placement` can default to them.
    ``tolerance`` is the adjoint solves' relative residual; the solver for it is cached on the
    subject and shared by every target.
    """
    if not (math.isfinite(distance_mm) and distance_mm >= 0.0):
        raise ValueError(f"distance_mm must be non-negative and finite, got {distance_mm!r}.")
    ctx = context_of(subject)
    resolved = resolve_target(ctx, target)
    adjoint = solve_adjoint(ctx, resolved, ensure_adjoint_solver(ctx, tolerance))
    centers = np.ascontiguousarray(cp.asnumpy(centers_mm), dtype=np.float64).reshape(-1, 3)
    grid = _coverage_grid(ctx, coil, centers, distance_mm, spacing_mm, margin_mm)
    field = sample_qfield(adjoint, grid)
    field.distance_mm = float(distance_mm)
    field.centers_mm = centers
    return field
