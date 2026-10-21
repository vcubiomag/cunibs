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

from cunibs._memory import build_arena
from cunibs.adm.grid import Grid, SweptCoverage, build_grid
from cunibs.adm.target import ResolvedTarget, Target, resolve_target
from cunibs.fem.assembly import csr_offsets
from cunibs.fem.placement import MU0_OVER_4PI, compute_coil_transforms
from cunibs.fem.solve import (
    GroundedSolver,
    SolverContext,
    _pad_block,
    _solve_grounded_block_mat,
    context_of,
)
from cunibs.solver import (
    adm_entry_floats,
    adm_reach,
    adm_sample,
    adm_sample_tile,
    adm_slot_corners,
    node_gather,
    node_weight,
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

# One byte per (rho, h) raster cell: whether some profile entry lies within reach of its centre.
# The squares go through pow, as CuPy's ``** 2`` does, and pow(x, 2.0) is not always x * x to the
# last bit; the sum is rounded on its own so that no contraction moves a cell on the edge.
_reach_profile = cp.ElementwiseKernel(
    "raw float64 rc, raw float64 hc, raw float64 prof, int32 n_prof, int32 n_h, float64 reach_sq",
    "uint8 hit",
    """
    const double r = rc[i / n_h];
    const double h = hc[i % n_h];
    hit = 0;
    for (int j = 0; j < n_prof; ++j) {
        const double dr = r - prof[2 * j];
        const double dh = h - prof[2 * j + 1];
        if (__dadd_rn(pow(dr, 2.0), pow(dh, 2.0)) <= reach_sq) {
            hit = 1;
            break;
        }
    }
    """,
    "cunibs_reach_profile",
)

# Each node in grid index units, as float32, and its extended cell's flat index, or n_cells when
# the node lies outside every extended cell. Each step is rounded on its own, so the coordinate is
# the correctly rounded one whatever the compiler would fuse, and a node on a cell boundary lands
# in the same cell on every architecture.
_grid_cells = cp.ElementwiseKernel(
    "raw float64 nodes, raw float64 origin, raw float64 spacing, int64 n0, int64 n1, int64 n2",
    "raw float32 src, T flat",
    """
    const long long n_ext[3] = {n0, n1, n2};
    long long ext[3];
    bool inside = true;
    for (int c = 0; c < 3; ++c) {
        const double x = __dsub_rn(__dmul_rn(nodes[3 * i + c], 1e-3), origin[c]);
        const float s = static_cast<float>(__ddiv_rn(x, spacing[c]));
        src[3 * i + c] = s;
        ext[c] = static_cast<long long>(floorf(s)) + 1;
        inside = inside && ext[c] >= 0 && ext[c] < n_ext[c];
    }
    flat = static_cast<T>(inside ? (ext[0] * n1 + ext[1]) * n2 + ext[2] : n0 * n1 * n2);
    """,
    "cunibs_grid_cells",
)

# out[d, j] = float32(w[d, order[j]]) for (D, n, 3) weights.
_gather_rows = cp.ElementwiseKernel(
    "raw float64 w, raw int64 order, int64 n",
    "float32 out",
    """
    const long long row = i / 3;
    const long long d = row / n;
    out = static_cast<float>(w[(d * n + order[row % n]) * 3 + i % 3]);
    """,
    "cunibs_gather_rows",
)

# One packed slot entry per row of out: source idx[k]'s position, then its weight per direction,
# then zero padding to the row width.
_pack_entries = cp.ElementwiseKernel(
    "raw float32 src, raw float32 w, raw int64 idx, int64 n_src, int32 n_dir, int32 width",
    "float32 out",
    """
    const long long j = idx[i / width];
    const int f = i % width;
    if (f < 3) {
        out = src[3 * j + f];
    } else if (f < 3 + 3 * n_dir) {
        out = w[((f - 3) / 3 * n_src + j) * 3 + (f - 3) % 3];
    } else {
        out = 0.0f;
    }
    """,
    "cunibs_pack_entries",
)


def build_adjoint_solver(
    ctx: SolverContext,
    tolerance: float = ADJOINT_TOLERANCE,
    max_iters: int = ADJOINT_MAX_ITERS,
) -> GroundedSolver:
    """A tight-tolerance grounded solver for the adjoint (reciprocity) solves.

    The adjoint system is the forward one: the context's stiffness is already assembled in
    float64 from float64 gradients and grounds the same DOF, so its operator and hierarchy are
    shared and only the stopping rule differs.
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


def _sorted_roi(target: ResolvedTarget) -> tuple[cp.ndarray, cp.ndarray]:
    """The ROI elements in ascending order as int32, and where each sits in ``target.elem_idx``."""
    order = cp.argsort(target.elem_idx)
    return cp.ascontiguousarray(target.elem_idx[order], dtype=cp.int32), order


def _adjoint_rhs(
    ctx: SolverContext, target: ResolvedTarget, roi: cp.ndarray, roi_order: cp.ndarray
) -> cp.ndarray:
    """Assemble the nodal adjoint RHS ``c``, ``(n_nodes, D)``, one column per direction ``ê``.

    Each node sums its corners' contributions ``α_e (ê·∇λ_i)`` over the node2corner CSR in its
    fixed order, as every other assembly does; a scatter from the ROI would need atomics and would
    sum a node's corners in whatever order the blocks retired. Only the nodes the ROI touches can
    be non-zero, so only their segments are gathered, each corner outside the ROI adding a zero
    just as it would in a sum over the whole mesh.
    """
    elem_idx = target.elem_idx
    nodes = cp.unique(ctx.tet_nodes[elem_idx].ravel())
    lo = ctx.node2corner_ptr[nodes]
    ptr = cp.zeros(int(nodes.size) + 1, dtype=cp.int32)
    cp.cumsum(ctx.node2corner_ptr[nodes + 1] - lo, out=ptr[1:])
    entry = cp.arange(int(ptr[-1]), dtype=cp.int32)
    owner = cp.searchsorted(ptr[1:], entry, side="right")
    corner = ctx.node2corner_idx[lo[owner] + (entry - ptr[owner])]
    elem = corner >> 2
    at = cp.minimum(cp.searchsorted(roi, elem), roi.size - 1)
    in_roi = roi[at] == elem
    row = roi_order[at]
    local = corner & 3

    g_roi = ctx.g[elem_idx].astype(cp.float64)
    c = cp.zeros((ctx.n_nodes, int(target.directions.shape[0])), dtype=cp.float64)
    gathered = cp.empty(int(nodes.size), dtype=cp.float64)
    stream = cp.cuda.get_current_stream().ptr
    for d in range(c.shape[1]):
        values = target.weights[:, None] * (g_roi @ target.directions[d])
        node_gather(cp.where(in_roi, values[row, local], 0.0), ptr, entry, gathered, stream)
        c[nodes, d] = gathered
    return c


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
    n_dir = int(target.directions.shape[0])
    roi, roi_order = _sorted_roi(target)
    reduced = _adjoint_rhs(ctx, target, roi, roi_order)[solver.idx]
    # A direction the target cannot see has a zero right-hand side and solves to zero.
    x, _ = _solve_grounded_block_mat(solver, _pad_block(reduced, n_dir), n_dir)
    del reduced
    lams = cp.zeros((n_dir, ctx.n_nodes), dtype=cp.float64)
    lams[:, solver.idx] = x[:, :n_dir].T
    del x

    # W_n = ¼ Σ_{e∋n} w_e with w_e = vol_e·σ_e·(G_e λ), less α_e ê on the ROI elements: the
    # direct -ê·dA/dt_{ROI} term folded into the element weights.
    alpha = cp.ascontiguousarray(target.weights[roi_order], dtype=cp.float64)
    directions = cp.asnumpy(target.directions)
    stream = cp.cuda.get_current_stream().ptr
    node_w = cp.empty((n_dir, ctx.n_nodes, 3), dtype=cp.float64)
    for d in range(n_dir):
        node_weight(
            lams[d],
            ctx.tet_nodes,
            ctx.g,
            ctx.neg_vc,
            ctx.node2corner_ptr,
            ctx.node2corner_idx,
            roi,
            alpha,
            [float(v) for v in directions[d]],
            node_w[d],
            stream,
        )
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

    Q is singular at every mesh node and the grid's cells straddle the scalp, so ``q`` leaves out,
    at each point, the nodes in the eight cells around it, and no near-node term is ever
    interpolated. A sampled cell whose 27 neighbours hold a node (``cell_near``) gets a slot: its
    own eight corners ``q_near`` without those nodes, and the nodes themselves, which the evaluator
    sums at the dipole directly. ``src`` and ``src_w`` are the nodes in grid index units and their
    weights, sorted by extended cell (-1 .. n-1 per axis, so every grid point has all eight of its
    cells), and ``cell_ptr`` is their CSR. A slot's nodes are ``entries[list_ptr[i]:list_ptr[i+1]]``,
    each its position then its weights, packed so one is a few vector loads.

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
    cell_near: cp.ndarray  # ((nx-1) * (ny-1) * (nz-1),) uint8
    src: cp.ndarray  # (n_nodes, 3) float32, grid index units
    src_w: cp.ndarray  # (D, n_nodes, 3) float32
    cell_ptr: cp.ndarray  # ((nx+1) * (ny+1) * (nz+1) + 1,) int32
    tile_box: cp.ndarray  # (ceil(n_nodes / adm_sample_tile()), 6) int32
    cell_slot: cp.ndarray  # ((nx-1) * (ny-1) * (nz-1),) int32, -1 for none
    slot_cells: cp.ndarray  # (n_slots,) int64 flat cell index
    list_ptr: cp.ndarray  # (n_slots + 1,) int32
    entries: cp.ndarray  # (n_entries, adm_entry_floats(D)) float32
    q_near: cp.ndarray  # (n_slots, 8, D, 3) float32
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
        adm_sample(
            self.src,
            self.src_w,
            self.tile_box,
            cp.ascontiguousarray(cp.sort(points), dtype=cp.int64),
            MU0_OVER_4PI,
            *_grid_args(grid),
            self.q.reshape(-1),
            cp.cuda.get_current_stream().ptr,
        )
        self.sampled[points] = 1
        self.cell_ok = _cells_ok(self.sampled, grid.shape)
        fresh = (self.cell_ok & self.cell_near).astype(cp.bool_) & (self.cell_slot < 0)
        self._add_slots(cp.flatnonzero(fresh))

    def _add_slots(self, cells: cp.ndarray) -> None:
        """Give each of the sampled near cells ``cells`` a slot: its node list and corners."""
        if cells.size == 0:
            return
        grid = self.grid
        _, ny, nz = grid.shape
        ix = cells // ((ny - 1) * (nz - 1))
        iy = (cells // (nz - 1)) % (ny - 1)
        iz = cells % (nz - 1)
        # Cell i's neighbours are extended cells i .. i + 2; each (x, y) row's three are adjacent.
        rows = cp.stack(
            [
                ((ix + a) * (ny + 1) + (iy + b)) * (nz + 1) + iz
                for a in range(3)
                for b in range(3)
            ],
            axis=1,
        )
        lo = self.cell_ptr[rows].ravel()
        seg = self.cell_ptr[rows + 3].ravel() - lo
        ends = cp.cumsum(seg)
        entry = cp.arange(int(ends[-1]), dtype=cp.int64)
        owner = cp.searchsorted(ends, entry, side="right")
        idx = lo[owner] + (entry - (ends - seg)[owner])

        first = int(self.slot_cells.size)
        counts = seg.reshape(-1, 9).sum(axis=1)
        self.list_ptr = cp.concatenate(
            [self.list_ptr, self.list_ptr[-1] + cp.cumsum(counts).astype(cp.int32)]
        )
        packed = cp.empty((int(idx.size), self.entries.shape[1]), dtype=cp.float32)
        _pack_entries(
            self.src,
            self.src_w,
            idx,
            int(self.src.shape[0]),
            self.n_directions,
            int(packed.shape[1]),
            packed,
        )
        fresh = cp.empty((int(cells.size), *self.q_near.shape[1:]), cp.float32)
        if first == 0:
            self.entries, self.q_near = packed, fresh
        else:
            self.entries = cp.concatenate([self.entries, packed])
            self.q_near = cp.concatenate([self.q_near, fresh])
        del packed, fresh
        self.slot_cells = cp.concatenate([self.slot_cells, cells.astype(cp.int64)])
        self.cell_slot[cells] = cp.arange(first, first + int(cells.size), dtype=cp.int32)
        adm_slot_corners(
            self.slot_cells,
            first,
            self.list_ptr,
            self.entries.reshape(-1),
            self.n_directions,
            self.q.reshape(-1),
            MU0_OVER_4PI,
            *_grid_args(grid),
            self.q_near.reshape(-1),
            cp.cuda.get_current_stream().ptr,
        )


def _grid_args(grid: Grid) -> tuple[list[float], list[float], list[int]]:
    return (
        [float(x) for x in cp.asnumpy(grid.origin_m)],
        [float(x) for x in cp.asnumpy(grid.spacing_m)],
        list(grid.shape),
    )


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
    # A coil's dipoles repeat (rho, h) pairs, and each cell only asks whether any one is in reach.
    # cp.unique with an axis sorts on the host, so the duplicates are dropped after a lexsort.
    ordered = cp.asarray(prof, dtype=cp.float64)[cp.lexsort(prof.T)]
    first = cp.ones(int(ordered.shape[0]), dtype=cp.bool_)
    first[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    unique = cp.ascontiguousarray(ordered[first])
    profile = cp.empty((n_rho, n_h), dtype=cp.uint8)
    _reach_profile(rc, hc, unique, int(unique.shape[0]), n_h, reach_sq, profile)

    mask = cp.empty(grid.n_points, dtype=cp.uint8)
    adm_reach(
        [float(x) for x in cp.asnumpy(grid.origin_m)],
        [float(x) for x in cp.asnumpy(grid.spacing_m)],
        list(grid.shape),
        cp.ascontiguousarray(coverage.origins_m, dtype=cp.float64),
        cp.ascontiguousarray(coverage.axes, dtype=cp.float64),
        profile,
        h_lo,
        cell,
        mask,
        cp.cuda.get_current_stream().ptr,
    )
    return mask


def _binned_sources(
    adjoint: AdjointField, grid: Grid
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """The nodes in grid index units sorted by extended cell, their weights, CSR, tile boxes and
    ``cell_near``.

    Nodes outside the extended cells are never near a grid point, so they follow the binned ones
    and only enter the far field.
    """
    nodes = cp.ascontiguousarray(cp.asarray(adjoint.ctx.nodes_mm), dtype=cp.float64)
    n_src = int(nodes.shape[0])
    n_ext = tuple(int(n) + 1 for n in grid.shape)
    n_cells = n_ext[0] * n_ext[1] * n_ext[2]
    src = cp.empty((n_src, 3), dtype=cp.float32)
    # Binned from the float32 coordinates the kernels read, so both sides agree on every cell.
    flat = cp.empty(n_src, dtype=cp.int32 if n_cells < np.iinfo(np.int32).max else cp.int64)
    _grid_cells(nodes, grid.origin_m, grid.spacing_m, *n_ext, src, flat)
    order = cp.argsort(flat, kind="stable")
    cell_ptr = csr_offsets(flat[order], n_cells)

    occupied = (cell_ptr[1:] > cell_ptr[:-1]).reshape(n_ext)
    # Cell i's 27 neighbours are extended cells i .. i + 2 on each axis.
    near = occupied[:-2] | occupied[1:-1] | occupied[2:]
    near = near[:, :-2] | near[:, 1:-1] | near[:, 2:]
    near = near[:, :, :-2] | near[:, :, 1:-1] | near[:, :, 2:]

    src = src[order]
    tile = adm_sample_tile()
    pad = (-n_src) % tile
    cells = cp.pad(cp.floor(src).astype(cp.int32), ((0, pad), (0, 0)), mode="edge").reshape(
        -1, tile, 3
    )
    tile_box = cp.concatenate([cells.min(axis=1), cells.max(axis=1)], axis=1)
    weights = cp.ascontiguousarray(adjoint.node_weights, dtype=cp.float64)
    w = cp.empty((int(weights.shape[0]), n_src, 3), dtype=cp.float32)
    _gather_rows(weights, order, n_src, w)
    return src, w, cell_ptr, tile_box, near.astype(cp.uint8).ravel()


def _reach_points(grid: Grid) -> cp.ndarray:
    """The flat indices of the grid points to sample up front: the coverage's, or all of them."""
    if grid.coverage is None:
        return cp.arange(grid.n_points, dtype=cp.int64)
    return cp.flatnonzero(_reachable(grid, grid.coverage))


def _sampled_field(
    ctx: SolverContext,
    target: ResolvedTarget,
    grid: Grid,
    binned: tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray],
    points: cp.ndarray,
) -> ReciprocityField:
    """A reciprocity field over ``binned`` sources (see :func:`_binned_sources`), sampled at
    ``points``."""
    src, src_w, cell_ptr, tile_box, cell_near = binned
    n_dir = int(src_w.shape[0])
    field = ReciprocityField(
        ctx=ctx,
        target=target,
        grid=grid,
        q=cp.zeros((*grid.shape, n_dir, 3), dtype=cp.float32),
        magnitude=target.magnitude,
        sampled=cp.zeros(grid.n_points, dtype=cp.uint8),
        cell_ok=cp.zeros(0, dtype=cp.uint8),
        cell_near=cell_near,
        src=src,
        src_w=src_w,
        cell_ptr=cell_ptr,
        tile_box=tile_box,
        cell_slot=cp.full(cell_near.size, -1, dtype=cp.int32),
        slot_cells=cp.zeros(0, dtype=cp.int64),
        list_ptr=cp.zeros(1, dtype=cp.int32),
        entries=cp.zeros((0, adm_entry_floats(n_dir)), dtype=cp.float32),
        q_near=cp.zeros((0, 8, n_dir, 3), dtype=cp.float32),
    )
    field.sample_points(points)
    return field


def sample_qfield(adjoint: AdjointField, grid: Grid) -> ReciprocityField:
    """Evaluate the reciprocity Q-field's far part on ``grid`` (see :class:`ReciprocityField`).

    All directions go in one pass over the sources. With a coverage on the grid only the reachable
    points are sampled; without one, all of them.
    """
    return _sampled_field(
        adjoint.ctx, adjoint.target, grid, _binned_sources(adjoint, grid), _reach_points(grid)
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
    centers = np.ascontiguousarray(cp.asnumpy(centers_mm), dtype=np.float64).reshape(-1, 3)
    # The field and what it holds come from the caller's allocator; the adjoint solve, the binning
    # and the reach raster are the build's own temporaries.
    with build_arena() as arena:
        with arena.resident():
            resolved = resolve_target(ctx, target)
            grid = _coverage_grid(ctx, coil, centers, distance_mm, spacing_mm, margin_mm)
        points = _reach_points(grid)
        adjoint = solve_adjoint(ctx, resolved, ensure_adjoint_solver(ctx, tolerance))
        binned = tuple(arena.keep(a) for a in _binned_sources(adjoint, grid))
        # The float64 node weights are spent once binned, so they go before q is allocated.
        del adjoint
        with arena.resident():
            field = _sampled_field(ctx, resolved, grid, binned, points)
    field.distance_mm = float(distance_mm)
    field.centers_mm = centers
    return field
