"""Solve the P1 FEM system for the TMS E-field on the GPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import numpy.typing as npt

from cunibs._memory import Arena, build_arena
from cunibs.fem.assembly import (
    assemble_stiffness,
    build_node2corner,
    conductivity_per_tet,
    gradient_operator,
)
from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transforms
from cunibs.fem.recovery import (
    RecoveredField,
    Recovery,
    RecoveryOperator,
    apply_recovery_into,
)
from cunibs.mesh import SKIN_SURFACE_TAG, HeadMesh
from cunibs.solver import (
    BLOCK_SIZES as _BLOCK_SIZES,
)
from cunibs.solver import (
    NativeVCycle,
    PcgAmgSolver,
    dadt_node_to_element,
    l1_dinv,
    reconstruct_e,
    reconstruct_e_nodal_block,
    rhs_operator_apply,
    rhs_operator_fill,
    select_size4,
    skin_normals,
    tet_lowest_node,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from cunibs.fem.sweep import SweepBasis
    from cunibs.simulation import Subject

# Stiffness assembly needs float64. Placement kernels use float32 to reduce memory use.
RESIDENT_G_DTYPE = cp.float32

# Stopping criterion of the mixed PCG: ||r||_2 / ||b||_2 <= DEFAULT_TOLERANCE.
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_ITERS = 2000


def _csr_f32(a: csp.csr_matrix) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """``(indptr, indices, data)`` in the int32/fp32 layout the native kernels take."""
    return (
        a.indptr.astype(cp.int32, copy=False),
        a.indices.astype(cp.int32, copy=False),
        a.data.astype(cp.float32, copy=False),
    )


def _l1_dinv(a: csp.csr_matrix) -> cp.ndarray:
    """l1-Jacobi smoother scaling: 1 / d, with d_i = sign(a_ii) * sum_j |a_ij|.

    The diagonal is included in the L1 norm and the sign follows the diagonal, which is what
    keeps the smoother positive definite and so the whole V-cycle a valid PCG preconditioner.

    Including the diagonal in the row sum is also what bounds the smoother's spectrum. For SPD
    ``a``, ``D - A`` is weakly diagonally dominant with a non-negative diagonal, hence PSD, so
    ``rho(D^-1 A) <= 1`` on every level with no estimate needed. Both the Chebyshev interval in
    vcycle.cu and the damping below rest on that bound.

    The row sum is a kernel so that its summation order is fixed by the CSR: dinv scales every
    entry of the smoothed prolongator and reaches every V-cycle level, so any drift in it reaches
    the whole hierarchy. See l1.cu.
    """
    dinv = cp.empty(a.shape[0], dtype=cp.float32)
    l1_dinv(*_csr_f32(a), dinv, cp.cuda.get_current_stream().ptr)
    return dinv


# Damping for the prolongator smoother: 4 / (3 rho) minimises the largest eigenvalue the
# smoothed basis leaves behind over [0, rho], and rho <= 1 by the bound in _l1_dinv.
_SA_DAMPING = 4.0 / 3.0


def _tentative_prolongator(agg: cp.ndarray, n_coarse: int) -> csp.csr_matrix:
    """Piecewise-constant P over the aggregates, columns normalised to unit 2-norm.

    The near-nullspace of a conductivity operator is the constant, so one column per aggregate
    carrying the constant over it is the coarse basis to build from. Normalising rescales the
    coarse unknowns rather than changing the space they span, and keeps the Galerkin operator's
    rows on one scale when aggregates differ in size.
    """
    n = int(agg.shape[0])
    # cp.bincount is unsafe here, as csr_offsets says; integer atomics count exactly in any order.
    counts = cp.zeros(n_coarse, dtype=cp.int32)
    cp.add.at(counts, agg, 1)
    counts = counts.astype(cp.float32)
    scale = cp.reciprocal(cp.sqrt(counts))
    return csp.csr_matrix(
        (scale[agg], agg, cp.arange(n + 1, dtype=cp.int32)), shape=(n, n_coarse)
    )


def _smooth_prolongator(
    a: csp.csr_matrix, p_tent: csp.csr_matrix, dinv: cp.ndarray
) -> csp.csr_matrix:
    """One damped Jacobi pass on the tentative prolongator: P = (I - (4/3) D^-1 A) P_tent.

    Smoothing gives each coarse basis function support that overlaps its neighbours' and a
    smooth profile across it, so the coarse space can represent the error the smoother leaves
    behind. It costs fill, in P and again in the Galerkin product built from it.
    """
    ap = a @ p_tent
    scaled = csp.csr_matrix(
        (ap.data * cp.repeat(dinv, cp.diff(ap.indptr)), ap.indices, ap.indptr), shape=ap.shape
    )
    p = (p_tent - cp.float32(_SA_DAMPING) * scaled).tocsr()
    p.sum_duplicates()
    p.sort_indices()
    return p


def _galerkin(a: csp.csr_matrix, p: csp.csr_matrix) -> csp.csr_matrix:
    """A_{l+1} = P^T A_l P."""
    coarse = (p.T.tocsr() @ a @ p).tocsr()
    coarse.sum_duplicates()
    coarse.sort_indices()
    return coarse


@dataclass(frozen=True)
class AggregationParams:
    """Hierarchy shape controls.

    ``min_coarse_rows`` bounds the dense coarse solve, whose inverse costs O(n_coarse^2)
    memory. It applies to the *next* level's size, so the coarsest level sits above it: 465
    rows on the test patch.
    """

    min_coarse_rows: int = 128
    max_levels: int = 50
    # Pairwise passes composed into one level's aggregation. See _aggregate: smoothed
    # aggregation over a single pass does not fit in memory on a head mesh.
    rounds: int = 2


@dataclass(frozen=True)
class SmootherParams:
    """Chebyshev smoother shape, over the interval ``[1 / lower_ratio, 1]``.

    The top of that interval is the analytic bound ``rho(D^-1 A) <= 1`` that the l1 diagonal
    gives (see ``_l1_dinv``), not an estimate, so no ``lower_ratio > 1`` can push the effective
    weight to ``2 / rho``: the smoother is A-convergent and the V-cycle SPD by construction, at
    every level and on every mesh.

    ``degree`` is how many terms the polynomial runs. Degree 1 is relaxed Jacobi at the
    interval's optimal weight; each further degree adds one SpMV-shaped kernel per level per
    sweep, and degree 2 buys back more than that in iterations.
    """

    degree: int = 2
    lower_ratio: float = 12.0


def _select_once(a: csp.csr_matrix, stream: int) -> tuple[cp.ndarray, int]:
    """One pairwise pass of the AMGx SIZE_4 selector over ``a``."""
    agg = cp.empty(a.shape[0], dtype=cp.int32)
    n_coarse = select_size4(*_csr_f32(a), agg, stream)
    return agg, n_coarse


def _aggregate(
    a: csp.csr_matrix, params: AggregationParams, stream: int
) -> tuple[cp.ndarray | None, int]:
    """Compose ``params.rounds`` pairwise passes into one level's aggregate map.

    SIZE_4 already pairs twice, so a single pass gives aggregates of ~4 and a coarsening ratio
    of ~4.2, which is far too fine for smoothed aggregation in 3D: a fine row's ~14 neighbours
    land in ~8 distinct aggregates, so the smoothed P carries ~8 nonzeros per row against a
    coarse space only 4.2x smaller, and the Galerkin product densifies until a head mesh runs
    out of memory one level down.

    Composing passes fixes it from both ends. Bigger aggregates mean a row's neighbours fall
    into fewer of them, so P gets *sparser* per row as the coarse space shrinks faster. The
    intermediate operators exist only to aggregate on, so they are built with the cheap
    tentative P and thrown away.

    Returns ``(None, n_rows)`` when the level cannot coarsen.
    """
    cur = a
    agg: cp.ndarray | None = None
    n_coarse = a.shape[0]
    for round_i in range(params.rounds):
        if cur.shape[0] <= params.min_coarse_rows:
            break
        step, nc = _select_once(cur, stream)
        if nc == cur.shape[0] or nc < params.min_coarse_rows:
            break
        agg = step if agg is None else step[agg]
        n_coarse = nc
        if round_i + 1 < params.rounds:
            cur = _galerkin(cur, _tentative_prolongator(step, nc))
    return agg, n_coarse


class AggregationLevel(NamedTuple):
    """One coarsening step of the hierarchy, with its smoother's l1 diagonal."""

    a: csp.csr_matrix
    aggregates: cp.ndarray
    p: csp.csr_matrix
    dinv: cp.ndarray

    @property
    def n_coarse(self) -> int:
        return int(self.p.shape[1])


def aggregation_levels(
    row_ptr: cp.ndarray,
    col_idx: cp.ndarray,
    values_f32: cp.ndarray,
    params: AggregationParams | None = None,
) -> tuple[list[AggregationLevel], csp.csr_matrix]:
    """Coarsen until the dense-solve floor; return the per-level operators and the coarsest.

    Levels come out finest first. Coarsening stops when a level cannot coarsen at all or the
    next level would undershoot ``min_coarse_rows``; that level is discarded and its matrix
    becomes the coarsest.

    Each level's prolongator is built here rather than by the caller because the next level's
    operator is the Galerkin product over it, so the transfer and the coarse operator have to
    agree.

    Each level is built in an arena of its own, since the Galerkin product's SpGEMM workspace is
    many times the operator it produces, and only the level's results are kept.
    """
    if params is None:
        params = AggregationParams()
    n = int(row_ptr.shape[0]) - 1
    a = csp.csr_matrix((values_f32, col_idx, row_ptr), shape=(n, n))
    stream = cp.cuda.get_current_stream().ptr
    levels: list[AggregationLevel] = []
    for _ in range(params.max_levels):
        n_rows = a.shape[0]
        if n_rows <= params.min_coarse_rows:
            break
        with build_arena() as arena:
            agg, n_coarse = _aggregate(a, params, stream)
            if agg is None or n_coarse == n_rows or n_coarse < params.min_coarse_rows:
                break
            dinv = _l1_dinv(a)
            p = _smooth_prolongator(a, _tentative_prolongator(agg, n_coarse), dinv)
            level = AggregationLevel(a, arena.keep(agg), arena.keep(p), arena.keep(dinv))
            a = arena.keep(_galerkin(a, p))
        levels.append(level)
    return levels, a


def build_native_vcycle(
    row_ptr: cp.ndarray,
    col_idx: cp.ndarray,
    values_f32: cp.ndarray,
    params: AggregationParams | None = None,
    smoother: SmootherParams | None = None,
) -> NativeVCycle:
    """Build the fp32 V-cycle preconditioner: aggregate, then upload each level's operators.

    P and R = P^T both go up as CSR. ``sort_indices`` on each is what fixes the summation order
    inside the prolongate and restrict kernels, which walk their rows in index order.

    The V-cycle copies every level into memory of its own, so no CuPy array outlives the build and
    all of it runs in an arena: a rebuild inside a caller's scratch allocator leaves nothing there.
    """
    if smoother is None:
        smoother = SmootherParams()
    with build_arena():
        levels, coarse = aggregation_levels(row_ptr, col_idx, values_f32, params)
        stream = cp.cuda.get_current_stream().ptr
        vc = NativeVCycle()
        vc.set_smoother(smoother.degree, smoother.lower_ratio)
        for level in levels:
            a, p = level.a, level.p
            r = p.T.tocsr()
            r.sort_indices()
            vc.add_level(*_csr_f32(a), level.dinv, *_csr_f32(p), *_csr_f32(r), stream)
        ainv = cp.linalg.inv(coarse.todense().astype(cp.float32))
        vc.set_coarse(cp.ascontiguousarray(ainv), stream)
        vc.finalize()
    return vc


def ground_node_of(nodes_mm: cp.ndarray) -> int:
    """The grounded DOF: the lowest node in z (shared by forward and adjoint systems)."""
    return int(cp.argmin(nodes_mm[:, 2]))


def grounded_index(n: int, ground_node: int) -> cp.ndarray:
    """Row/column index that drops ``ground_node`` from an ``n``-DOF system."""
    idx = cp.arange(n - 1, dtype=cp.int32)
    idx[ground_node:] += 1
    return idx


# Morton keys quantise to a 2^21 grid per axis, which is the most that fits three ways into a
# 64-bit key and is far finer than any mesh's node spacing. _spread3's masks are written for that
# width rather than derived from it, so changing this constant alone would be silently wrong.
_MORTON_BITS = 21


def _spread3(v: cp.ndarray) -> cp.ndarray:
    """Insert two zero bits between each of the low 21 bits of ``v``."""
    v = v & cp.uint64(0x1FFFFF)
    v = (v | (v << cp.uint64(32))) & cp.uint64(0x1F00000000FFFF)
    v = (v | (v << cp.uint64(16))) & cp.uint64(0x1F0000FF0000FF)
    v = (v | (v << cp.uint64(8))) & cp.uint64(0x100F00F00F00F00F)
    v = (v | (v << cp.uint64(4))) & cp.uint64(0x10C30C30C30C30C3)
    return (v | (v << cp.uint64(2))) & cp.uint64(0x1249249249249249)


def morton_order(points_mm: cp.ndarray) -> cp.ndarray:
    """Order rows along a 3-D Morton (Z-order) curve over the points' bounding box."""
    lo = points_mm.min(axis=0)
    span = cp.maximum(points_mm.max(axis=0) - lo, 1e-12)
    q = ((points_mm - lo) / span * float((1 << _MORTON_BITS) - 1)).astype(cp.uint64)
    key = cp.zeros(int(points_mm.shape[0]), dtype=cp.uint64)
    for axis in range(3):
        key |= _spread3(q[:, axis]) << cp.uint64(axis)
    return cp.argsort(key, kind="stable").astype(cp.int32)


def invert_permutation(perm: cp.ndarray) -> cp.ndarray:
    """The index that undoes ``perm``."""
    inverse = cp.empty_like(perm)
    inverse[perm] = cp.arange(int(perm.shape[0]), dtype=perm.dtype)
    return inverse


def spatial_order(
    nodes_mm: cp.ndarray, tet_nodes: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Node and tetrahedron permutations that lay a mesh out along a Morton curve.

    Returns the node permutation, its inverse (old node id to new), and the tetrahedron
    permutation.

    Nodes are ordered by position and tetrahedra by their lowest renumbered node. Gmsh order
    bears no relation to position: on a head mesh the four node indices of a tetrahedron span
    ~380k, which is what every element-centric gather pays, and ~9k once ordered.

    A mesh already in this order sorts to the identity, so reordering is idempotent.
    """
    node_perm = morton_order(nodes_mm)
    inverse = invert_permutation(node_perm)
    tets = cp.ascontiguousarray(tet_nodes, dtype=cp.int32)
    lowest = cp.empty(int(tets.shape[0]), dtype=cp.int32)
    tet_lowest_node(inverse, tets, lowest, cp.cuda.get_current_stream().ptr)
    tet_perm = cp.argsort(lowest, kind="stable").astype(cp.int32)
    return node_perm, inverse, tet_perm


def solve_ordering(nodes_mm: cp.ndarray, ground_node: int) -> cp.ndarray:
    """Reduced-system row index: the mesh nodes bar the grounded one.

    The rows inherit the mesh's spatial order (see :func:`spatial_order`), which is what keeps
    the SpMV gather over ``x`` local: on a head mesh 92% of the reduced operator's nonzeros sit
    within a 4096-column window.
    """
    return grounded_index(int(nodes_mm.shape[0]), ground_node)


def reduce_matrix(a: csp.csr_matrix, idx: cp.ndarray) -> csp.csr_matrix:
    """Remove the grounded DOF and canonicalise the reduced CSR."""
    a_red = a[idx][:, idx].tocsr()
    a_red.sum_duplicates()
    return a_red


class SolverConvergenceError(RuntimeError):
    """The mixed PCG missed tolerance even after rebuilding the preconditioner.

    ``column`` is the offending column of the block, or ``None`` for a single-RHS solve. Blocks
    are contiguous slices of the sweep, so the placement is the block's first index plus it.
    """

    def __init__(
        self,
        iterations: int,
        relative_residual: float,
        tolerance: float,
        *,
        column: int | None = None,
    ) -> None:
        where = "" if column is None else f" (column {column} of the block)"
        super().__init__(
            f"PCG did not converge{where}: {iterations} iterations, relative residual "
            f"{relative_residual:.3e} > tolerance {tolerance:.3e}, after rebuilding the "
            f"preconditioner at the current matrix values"
        )
        self.iterations = iterations
        self.relative_residual = relative_residual
        self.tolerance = tolerance
        self.column = column


class ColumnTelemetry(NamedTuple):
    """How the linear solve went for one right-hand side.

    ``iterations`` is that column's own count, the iteration at which its residual first met
    tolerance and it froze, not the block's. A retried column is charged both passes, since the
    retry restarts from the failed iterate rather than from zero.
    """

    iterations: int
    relative_residual: float


@dataclass
class GroundedSolver:
    """Reduced SPD system with one grounded potential.

    Grounding removes the free additive constant in ``v``. It does not change
    ``E = -grad(v) - dA/dt``.
    """

    n: int
    idx: cp.ndarray
    # The fp32 apply inside solve_mixed. Mutable: a solve that misses tolerance rebuilds it
    # from the current values (see rebuild_preconditioner).
    precond: NativeVCycle
    pcg: PcgAmgSolver
    tolerance: float
    max_iters: int
    row_ptr: cp.ndarray
    col_idx: cp.ndarray
    values: cp.ndarray
    last_iterations: int = 0
    last_relative_residual: float = 0.0

    def rebuild_preconditioner(self) -> None:
        """Rebuild the fp32 V-cycle from the solver's current fp64 values.

        Only reached when the mixed PCG misses tolerance, which in practice means the
        hierarchy no longer matches the matrix. A preconditioner cannot change where fp64 PCG
        converges, only how fast, so rebuilding it is the whole remedy: there is no more
        accurate solver to fall back to. Installing a new NativeVCycle bumps its generation
        and so invalidates PcgAmgSolver's captured CG graph.
        """
        self.precond = build_native_vcycle(
            self.row_ptr,
            self.col_idx,
            self.values.astype(cp.float32),
        )


def prepare_grounded_solver(
    a: csp.csr_matrix,
    nodes_mm: cp.ndarray,
    ground_node: int,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> GroundedSolver:
    """Remove the ground DOF and build the mixed-precision solver."""
    n = a.shape[0]
    with build_arena() as arena:
        with arena.resident():
            idx = solve_ordering(nodes_mm, ground_node)
        a_red = reduce_matrix(a, idx)
        row_ptr = arena.keep(a_red.indptr.astype(cp.int32, copy=False))
        col_idx = arena.keep(a_red.indices.astype(cp.int32, copy=False))
        values = arena.keep(a_red.data.astype(cp.float64, copy=False))
        del a_red
        precond = build_native_vcycle(row_ptr, col_idx, values.astype(cp.float32))
        pcg = PcgAmgSolver(row_ptr, col_idx, values, cp.cuda.get_current_stream().ptr)
    return GroundedSolver(
        n=n,
        idx=idx,
        precond=precond,
        pcg=pcg,
        tolerance=tolerance,
        max_iters=max_iters,
        row_ptr=row_ptr,
        col_idx=col_idx,
        values=values,
    )


def solve_grounded(solver: GroundedSolver, b: cp.ndarray) -> cp.ndarray:
    """Solve one RHS on the prepared hierarchy.

    A miss rebuilds the preconditioner and restarts from the failed iterate, so the second
    attempt runs against a single fixed operator and CG's short recurrence stays valid.
    """
    b_red = cp.ascontiguousarray(b[solver.idx], dtype=cp.float64)
    x_red = cp.empty(int(solver.idx.shape[0]), dtype=cp.float64)
    stream = cp.cuda.get_current_stream().ptr
    iters, rel = solver.pcg.solve_mixed(
        solver.precond, b_red, x_red, solver.tolerance, solver.max_iters, stream
    )
    if float(rel) > solver.tolerance:
        solver.rebuild_preconditioner()
        iters, rel = solver.pcg.solve_mixed(
            solver.precond, b_red, x_red, solver.tolerance, solver.max_iters, stream, x_red
        )
        if float(rel) > solver.tolerance:
            raise SolverConvergenceError(int(iters), float(rel), solver.tolerance)
    solver.last_iterations = int(iters)
    solver.last_relative_residual = float(rel)
    v = cp.zeros(solver.n, dtype=cp.float64)
    v[solver.idx] = x_red
    return v


# Compiled block widths of the k-RHS solve kernels; smaller batches pad up by replicating the
# last column, which costs bandwidth but no extra matrix reads. Read from the extension rather
# than restated, so adding a width is a one-place change.
BLOCK_SIZES = _BLOCK_SIZES
MAX_BLOCK = BLOCK_SIZES[-1]


def _solve_grounded_block_mat(
    solver: GroundedSolver, B: cp.ndarray, k: int, X0: cp.ndarray | None = None
) -> tuple[cp.ndarray, list[ColumnTelemetry]]:
    """Block-solve a padded (n_red, k_pad) RHS matrix; returns X and the k columns' telemetry.

    Each of the k chains is numerically independent: per-column reductions, and a column that
    reaches tolerance freezes rather than being carried along to the block's slowest one, so it
    stops exactly where it would have solved alone. They share only the stiffness and hierarchy
    reads. If any column misses tolerance the preconditioner is rebuilt once, then only the
    offending columns are re-solved singly.
    """
    X = cp.empty_like(B)
    stream = cp.cuda.get_current_stream().ptr
    iters, rels, col_iters = solver.pcg.solve_mixed_block(
        solver.precond, B, X, solver.tolerance, solver.max_iters, stream, X0
    )
    telemetry = [ColumnTelemetry(int(col_iters[c]), float(rels[c])) for c in range(k)]
    solver.last_iterations = int(iters)
    solver.last_relative_residual = float(max(rels[:k]))

    missed = [c for c in range(k) if rels[c] > solver.tolerance]
    if missed:
        solver.rebuild_preconditioner()
        worst = 0.0
        for c in missed:
            b_red = cp.ascontiguousarray(B[:, c])
            x_red = cp.ascontiguousarray(X[:, c])
            retry_iters, rel = solver.pcg.solve_mixed(
                solver.precond, b_red, x_red, solver.tolerance, solver.max_iters, stream, x_red
            )
            if float(rel) > solver.tolerance:
                raise SolverConvergenceError(
                    int(retry_iters), float(rel), solver.tolerance, column=c
                )
            X[:, c] = x_red
            telemetry[c] = ColumnTelemetry(
                telemetry[c].iterations + int(retry_iters), float(rel)
            )
            solver.last_iterations = max(solver.last_iterations, int(retry_iters))
            worst = max(worst, float(rel))
        # The retried columns now bound the block: every other column already met tolerance.
        solver.last_relative_residual = max(
            worst, max((rels[c] for c in range(k) if c not in missed), default=0.0)
        )
    return X, telemetry


def _pad_block(B: cp.ndarray, k: int) -> cp.ndarray:
    """Pad the (n_red, k) RHS matrix to a compiled block width by replicating a column."""
    k_pad = next(s for s in BLOCK_SIZES if s >= k)
    if k_pad == k:
        return cp.ascontiguousarray(B)
    padded = cp.empty((int(B.shape[0]), k_pad), dtype=B.dtype)
    padded[:, :k] = B
    padded[:, k:] = B[:, k - 1 : k]
    return padded


def _dadt_node_to_elm(dadt_nodes: cp.ndarray, tet_nodes: cp.ndarray) -> cp.ndarray:
    """Average nodal dA/dt over each tetrahedron."""
    dadt_elm = cp.empty((int(tet_nodes.shape[0]), 3), dtype=cp.float32)
    dadt_node_to_element(
        cp.ascontiguousarray(dadt_nodes),
        tet_nodes,
        dadt_elm,
        cp.cuda.get_current_stream().ptr,
    )
    return dadt_elm


def _assemble_rhs(ctx: SolverContext, dadt_block: cp.ndarray) -> cp.ndarray:
    """``(n_nodes, k)`` float32 RHS from an interleaved ``(n_nodes, k, 3)`` nodal dA/dt."""
    n, k, _ = dadt_block.shape
    b = cp.empty((n, k), dtype=cp.float32)
    rhs_operator_apply(
        ctx.rhs_indptr, ctx.rhs_op, dadt_block, b, cp.cuda.get_current_stream().ptr
    )
    return b


def build_rhs_operator(
    a: csp.csr_matrix,
    g: cp.ndarray,
    neg_vc: cp.ndarray,
    tet_nodes: cp.ndarray,
    ptr: cp.ndarray,
    idx: cp.ndarray,
) -> cp.ndarray:
    """The forward RHS as an operator on nodal dA/dt, packed over the stiffness pattern of ``a``.

    The RHS is linear in the nodal field once dA/dt reaches an element as the mean of its nodes,
    so ``b_i = sum_j B_ij . a_j`` with ``B_ij = sum_{e owns i, j} (neg_vc_e / 4) g[e, i]``. The
    result is ``(nnz, 4)`` float32, a nonzero's three components and then its column's bits.
    ``g`` and ``neg_vc`` are float64 so that B is contracted before it is rounded.
    """
    op = cp.empty((int(a.indices.shape[0]), 4), dtype=cp.float32)
    rhs_operator_fill(
        cp.ascontiguousarray(g, dtype=cp.float64),
        cp.ascontiguousarray(neg_vc * 0.25, dtype=cp.float64),
        tet_nodes,
        ptr,
        idx,
        a.indptr,
        a.indices,
        op,
        cp.cuda.get_current_stream().ptr,
    )
    return op


def _reconstruct_e_kernel(
    v: cp.ndarray,
    tet_nodes: cp.ndarray,
    g: cp.ndarray,
    dadt_elm: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Reconstruct E and its magnitude for each tetrahedron.

    The potential's differences across each element are taken in float64, because a grounded
    potential is large against its change across one element. See reconstruct.cu.
    """
    n_tet = int(tet_nodes.shape[0])
    e = cp.empty((n_tet, 3), dtype=cp.float32)
    magn = cp.empty(n_tet, dtype=cp.float32)
    reconstruct_e(
        cp.ascontiguousarray(v, dtype=cp.float64),
        tet_nodes,
        g,
        cp.ascontiguousarray(dadt_elm),
        e,
        magn,
        cp.cuda.get_current_stream().ptr,
    )
    return e, magn


@dataclass
class SolverContext:
    """GPU state shared by all placements for one mesh."""

    mesh: HeadMesh
    nodes_mm: cp.ndarray
    tet_nodes: cp.ndarray
    tet_tags: cp.ndarray
    n_nodes: int
    g: cp.ndarray
    vols: cp.ndarray
    # Where every metric and ROI locates an element. HeadMesh computes the same thing on the
    # host, which costs a second on a head mesh, so device consumers take it from here.
    tet_barycenters_mm: cp.ndarray
    neg_vc: cp.ndarray
    solver: GroundedSolver
    node2corner_ptr: cp.ndarray
    node2corner_idx: cp.ndarray
    # The forward RHS operator from build_rhs_operator, over the stiffness pattern's rows.
    rhs_indptr: cp.ndarray
    rhs_op: cp.ndarray
    # (n_tri, 3, 3) corner coordinates of the scalp triangles, and their outward normals.
    skin_verts: cp.ndarray
    skin_tri_normals: cp.ndarray
    # The same triangles as skin_verts, as node indices rather than gathered coordinates, and on
    # the device: mesh.skin_tris fetches to the host on first read.
    skin_tris: cp.ndarray
    # Filled on first use by ensure_recovery, never by build_context, so a run that asks for no
    # recovery pays neither the build nor the memory. Subject populates it under its own lock.
    recovery: dict[Recovery, RecoveryOperator] = field(default_factory=dict)
    # Same discipline for the reciprocity adjoint solver, keyed by (tolerance, max_iters): the
    # forward solver with a tighter stopping rule, which does not depend on the target.
    adjoint: dict[tuple[float, int], GroundedSolver] = field(default_factory=dict)
    # Set only by ensure_sweep_basis. While present, every placement block solve starts from its
    # Galerkin seed rather than from zero.
    sweep_basis: SweepBasis | None = None


def context_of(subject: Subject | SolverContext) -> SolverContext:
    """The solver context of ``subject``, which may also be a context already.

    Lets every public function that needs the mesh's GPU state take the :class:`Subject` a user
    holds, rather than making them reach for ``subject.context``.
    """
    return subject if isinstance(subject, SolverContext) else subject.context


def _spatially_ordered(
    mesh: HeadMesh, arena: Arena
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Reorder a mesh along a Morton curve, as device arrays the arena keeps.

    This order is the one every array in the context carries, and so the one a caller reads
    results against. Surface rows keep their file order; only the node ids they name move, so the
    surface tags do not move at all.
    """
    nodes_in = cp.asarray(mesh.nodes_mm)
    tets_in = cp.asarray(mesh.tet_nodes)
    node_perm, inverse, tet_perm = spatial_order(nodes_in, tets_in)
    tets_permuted = tets_in[tet_perm]
    tags_in = cp.asarray(mesh.tet_tags)
    surf = cp.asarray(mesh.surf_tris)
    with arena.resident():
        nodes_mm = cp.ascontiguousarray(nodes_in[node_perm])
        tet_nodes = cp.ascontiguousarray(inverse[tets_permuted])
        tet_tags = cp.ascontiguousarray(tags_in[tet_perm])
        surf_tris = cp.ascontiguousarray(inverse[surf]) if surf.shape[0] else surf
        surf_tags = cp.asarray(mesh.surf_tags)
    return nodes_mm, tet_nodes, tet_tags, surf_tris, surf_tags


def skin_triangle_normals(nodes_mm: cp.ndarray, skin_tris: cp.ndarray) -> cp.ndarray:
    """Smoothed outward normal per skin triangle, the surface a placement projects onto.

    Area-weighted face normals are spread to the triangles' nodes and gathered back twice, and the
    result is the normalised mean of each triangle's three node normals. The spread is a reduction
    over the triangles meeting at a node: it runs over a node-to-corner CSR built by a stable sort,
    in that CSR's order, so it reduces in the same order on every run.
    """
    tris = cp.ascontiguousarray(skin_tris, dtype=cp.int32)
    out = cp.empty((int(tris.shape[0]), 3), dtype=cp.float64)
    if tris.shape[0] == 0:
        return out
    ptr, idx = build_node2corner(tris, int(nodes_mm.shape[0]))
    skin_normals(
        cp.ascontiguousarray(nodes_mm, dtype=cp.float64),
        tris,
        ptr,
        idx,
        out,
        cp.cuda.get_current_stream().ptr,
    )
    return out


# Each tetrahedron's corner mean, gathered in one pass. Corners pair (0, 2) and (1, 3), the
# order cupy's mean over a corner axis sums in, so this matches HeadMesh.tet_barycenters_mm.
_barycentres = cp.ElementwiseKernel(
    "raw float64 x, raw int32 t",
    "float64 y",
    """
    const long long e = i / 3;
    const long long c = i % 3;
    const double p = x[3LL * t[4 * e] + c], q = x[3LL * t[4 * e + 1] + c];
    const double r = x[3LL * t[4 * e + 2] + c], s = x[3LL * t[4 * e + 3] + c];
    y = ((p + r) + (q + s)) / 4.0;
    """,
    "cunibs_barycentres",
)


def build_context(mesh: HeadMesh) -> SolverContext:
    """Build the GPU state shared by all placements.

    The mesh is first laid out along a Morton curve, and ``ctx.mesh`` is that reordering: it is
    what every array here, and every field a solve returns, is indexed by.

    Assemble ``g``, volumes, and stiffness in float64. Store ``g``, volumes, and
    ``-volume * conductivity`` in float32 for placement kernels.

    The context's arrays come from the caller's allocator; everything else is a temporary of the
    build's arena.
    """
    with build_arena() as arena:
        nodes_mm, tet_nodes, tet_tags, surf_tris, surf_tags = _spatially_ordered(mesh, arena)
        with arena.resident():
            skin_tris = cp.ascontiguousarray(surf_tris[surf_tags == SKIN_SURFACE_TAG])
        n_nodes = int(nodes_mm.shape[0])

        g, vols = gradient_operator(nodes_mm, tet_nodes)
        n_invalid = int(cp.count_nonzero((vols <= 0.0) | ~cp.isfinite(vols)))
        if n_invalid:
            subject = "tetrahedron has" if n_invalid == 1 else "tetrahedra have"
            raise ValueError(f"{n_invalid} {subject} zero or non-finite volume")
        cond = conductivity_per_tet(tet_tags)
        ptr, idx = (arena.keep(a) for a in build_node2corner(tet_nodes, n_nodes))
        stiffness = assemble_stiffness(
            g, vols, cond, n_nodes, tet_nodes, node2corner=(ptr, idx)
        )
        # The solver build below is the peak of the whole context, so everything it does not read
        # is finished and dropped first: the RHS operator is the last reader of the float64 g.
        rhs_op = arena.keep(
            build_rhs_operator(stiffness, g, -(vols * cond), tet_nodes, ptr, idx)
        )
        rhs_indptr = arena.keep(stiffness.indptr)
        with arena.resident():
            g32 = g.astype(RESIDENT_G_DTYPE)
            vols32 = vols.astype(cp.float32)
        neg_vc = arena.keep(-(vols32 * cond.astype(cp.float32)))
        del g, vols, cond
        with arena.resident():
            solver = prepare_grounded_solver(stiffness, nodes_mm, ground_node_of(nodes_mm))
        del stiffness

        with arena.resident():
            barycenters = cp.empty((int(tet_nodes.shape[0]), 3), dtype=cp.float64)
            _barycentres(nodes_mm, tet_nodes, barycenters)
            skin_verts = nodes_mm[skin_tris]
        skin_tri_normals = arena.keep(skin_triangle_normals(nodes_mm, skin_tris))
    # Device arrays throughout, so reading any of the mesh on the host is a caller's decision
    # rather than a cost of building the context.
    return SolverContext(
        HeadMesh(nodes_mm, tet_nodes, tet_tags, surf_tris, surf_tags),
        nodes_mm,
        tet_nodes,
        tet_tags,
        n_nodes,
        g32,
        vols32,
        barycenters,
        neg_vc,
        solver,
        ptr,
        idx,
        rhs_indptr,
        rhs_op,
        skin_verts,
        skin_tri_normals,
        skin_tris,
    )


class PlacementResult(TypedDict):
    """Arrays produced for one placement, plus how its linear solve went.

    ``E``/``magnE`` are whichever field the run's ``recovery`` mode asked for. ``E`` and ``v`` are
    present only when the caller asked for them, and ``E_slots`` only when a recovery mode ran and
    the caller asked for it.

    ``iterations``/``relative_residual`` are this placement's own, not its block's: in a k-RHS
    solve every column freezes at its own residual.
    """

    transform: npt.NDArray[np.float64]
    E: NotRequired[cp.ndarray]
    magnE: cp.ndarray
    v: NotRequired[cp.ndarray]
    iterations: int
    relative_residual: float
    E_slots: NotRequired[cp.ndarray]


def _check_operator(
    ctx: SolverContext, recovery: RecoveryOperator | None, nodal: bool, where: str
) -> None:
    """Validate a recovery operator handed to the FEM layer.

    This layer takes a built operator rather than a mode name because building one costs a full
    pass over the mesh and hundreds of megabytes of transient, so it must not happen inside a
    caller's scratch allocator or under a lock it does not hold.
    """
    if isinstance(recovery, str):
        raise TypeError(
            f"{where} takes a prebuilt RecoveryOperator, not the mode name {recovery!r}. Call "
            f"cunibs.fem.ensure_recovery(ctx, {recovery!r}) and pass the result, or use "
            f"Subject.simulate(..., recovery={recovery!r}), which builds it for you."
        )
    if nodal and recovery is None:
        raise ValueError(
            "nodal=True needs a recovery operator: the raw per-tetrahedron field has no nodal "
            "form. Pass one from cunibs.fem.ensure_recovery(ctx, 'harmonic')."
        )
    if recovery is None:
        return
    corners = int(recovery.slot_of_corner.shape[0])
    expected = 4 * int(ctx.tet_nodes.shape[0])
    if corners != expected:
        raise ValueError(
            f"recovery={recovery.mode!r} was built for a mesh with {corners // 4} tetrahedra "
            f"but this context has {expected // 4}. Build the operator from the same context "
            "you are solving on."
        )


def solve_placement(
    ctx: SolverContext,
    dip_pos_m: npt.ArrayLike,
    dip_moment: npt.ArrayLike,
    center_mm: npt.ArrayLike,
    handle_mm: npt.ArrayLike,
    distance_mm: float,
    didt: float,
    *,
    recovery: RecoveryOperator | None = None,
    nodal: bool = False,
    vectors: bool = True,
    potential: bool = True,
) -> PlacementResult:
    """Solve one placement and return device arrays plus the host transform.

    The width-1 case of :func:`solve_placements_block`, so a placement's result is the same
    whichever of the two solved it. ``recovery`` is a prebuilt operator from
    :func:`~cunibs.fem.ensure_recovery`, or ``None`` for the raw per-tetrahedron field; see
    :func:`_check_operator` for why this layer will not build one. ``nodal`` additionally keeps
    the recovered field on its slots, and ``vectors``/``potential`` keep ``E`` and ``v``.
    """
    site = (np.asarray(center_mm), np.asarray(handle_mm), float(distance_mm))
    (result,) = solve_placements_block(
        ctx,
        dip_pos_m,
        dip_moment,
        [site],
        didt,
        recovery=recovery,
        nodal=nodal,
        vectors=vectors,
        potential=potential,
    )
    return result


def solve_placements_block(
    ctx: SolverContext,
    dip_pos_m: npt.ArrayLike,
    dip_moment: npt.ArrayLike,
    sites: Sequence[tuple[npt.ArrayLike, npt.ArrayLike, float]],
    didt: float,
    *,
    recovery: RecoveryOperator | None = None,
    nodal: bool = False,
    tissues: Collection[int] | None = None,
    transforms: npt.NDArray[np.float64] | None = None,
    vectors: bool = True,
    potential: bool = True,
) -> list[PlacementResult]:
    """Solve up to MAX_BLOCK placements with block kernels end to end.

    ``sites`` are (center_mm, handle_mm, distance_mm) triples. All stages that read
    large placement-independent arrays (scalp projection, tet connectivity, weighted
    gradients, node2corner, g) run once per chunk rather than once per placement, and the
    linear solve shares every stiffness/hierarchy read across the k columns.

    Every chunk width reaches the same arithmetic, and :func:`solve_placement` is k = 1.

    ``recovery`` is a prebuilt operator from :func:`~cunibs.fem.ensure_recovery`, or ``None``
    for the raw per-tetrahedron field; see :func:`_check_operator` for why this layer will not
    build one. It is another placement-independent array read once per chunk, and its per-slot
    reduction order does not depend on k, so a placement's recovered field is the same at every
    block width. ``tissues`` limits the recovered field to those tissues' elements, leaving the
    rest of ``E`` and ``magnE`` unwritten; see :func:`~cunibs.fem.apply_recovery_into`.

    ``vectors`` and ``potential`` say whether each result carries ``E`` and ``v``. ``magnE`` is
    always computed; ``E`` is otherwise neither stored nor, where the field is reconstructed per
    element and nothing else reads it, written.

    ``transforms``, when given, are the sites' ``(k, 4, 4)`` coil transforms from
    :func:`~cunibs.fem.compute_coil_transforms`, which a sweep computes for all of its sites at
    once.
    """
    k = len(sites)
    solver = ctx.solver
    if k > MAX_BLOCK:
        raise ValueError(f"solve_placements_block: k={k} exceeds MAX_BLOCK={MAX_BLOCK}")
    _check_operator(ctx, recovery, nodal, "solve_placements_block")
    on_potential = recovery is not None and recovery.on_potential
    stream = cp.cuda.get_current_stream().ptr
    n_tet = int(ctx.tet_nodes.shape[0])

    if transforms is None:
        transforms = cp.asnumpy(
            compute_coil_transforms(
                ctx,
                np.asarray([s[0] for s in sites], dtype=np.float64).reshape(k, 3),
                np.asarray([s[1] for s in sites], dtype=np.float64).reshape(k, 3),
                np.asarray([s[2] for s in sites], dtype=np.float64),
            )
        )
    device_transforms = cp.asarray(transforms)

    # Staged interleaved, so that a node's k values are one contiguous gather for the RHS operator
    # and for the per-element field, which averages them onto each element as it reconstructs.
    dadt_block = cp.empty((ctx.n_nodes, k, 3), dtype=cp.float32)
    for i in range(k):
        coil_dadt_at_nodes(
            dip_pos_m,
            dip_moment,
            device_transforms[i],
            didt,
            ctx.nodes_mm,
            out=dadt_block,
            column=i,
        )

    b_block = _assemble_rhs(ctx, dadt_block)
    B_pad = _pad_block(b_block[solver.idx, :].astype(cp.float64), k)
    del b_block
    X0 = None if ctx.sweep_basis is None else ctx.sweep_basis.seed(B_pad)
    X, telemetry = _solve_grounded_block_mat(solver, B_pad, k, X0)
    del B_pad, X0

    v_block = cp.zeros((solver.n, k), dtype=cp.float64)
    v_block[solver.idx, :] = X[:, :k]
    del X

    # The potential path never reads the raw field, so it neither reconstructs nor allocates one.
    # An element-consuming recovery reads the raw E and replaces its magnitude, and with no
    # recovery the raw field is the result.
    es: list[cp.ndarray] = []
    magns: list[cp.ndarray] = []
    if not on_potential:
        if vectors or recovery is not None:
            es = [cp.empty((n_tet, 3), dtype=cp.float32) for _ in range(k)]
        if recovery is None:
            magns = [cp.empty(n_tet, dtype=cp.float32) for _ in range(k)]
        reconstruct_e_nodal_block(v_block, ctx.tet_nodes, ctx.g, dadt_block, es, magns, stream)

    slots: list[cp.ndarray] = []
    if recovery is not None:
        rec = RecoveredField.allocate(recovery, n_tet, k, vectors=vectors)
        apply_recovery_into(
            recovery,
            rec,
            elements=None if on_potential else es,
            potential=v_block,
            dadt_nodes=dadt_block if on_potential else None,
            tissues=tissues,
        )
        slots, es, magns = rec.E_slots, rec.E, rec.magnE
    del dadt_block

    results: list[PlacementResult] = [
        {
            "transform": transforms[i],
            "magnE": magns[i],
            "iterations": telemetry[i].iterations,
            "relative_residual": telemetry[i].relative_residual,
        }
        for i in range(k)
    ]
    for i, result in enumerate(results):
        if vectors:
            result["E"] = es[i]
        if potential:
            result["v"] = cp.ascontiguousarray(v_block[:, i])
        if nodal:
            result["E_slots"] = slots[i]
    return results
