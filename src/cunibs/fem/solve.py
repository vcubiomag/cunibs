"""Solve the P1 FEM system for the TMS E-field on the GPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, NotRequired, TypedDict

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import numpy.typing as npt

from cunibs.fem.assembly import (
    assemble_stiffness,
    build_node2corner,
    conductivity_per_tet,
    gradient_operator,
)
from cunibs.fem.placement import (
    coil_dadt_at_nodes,
    compute_coil_transform,
    compute_coil_transforms,
)
from cunibs.fem.recovery import (
    RecoveredField,
    Recovery,
    RecoveryOperator,
    apply_recovery,
    apply_recovery_into,
)
from cunibs.mesh import HeadMesh
from cunibs.solver import (
    BLOCK_SIZES as _BLOCK_SIZES,
)
from cunibs.solver import (
    NativeVCycle,
    PcgAmgSolver,
    dadt_node_to_element,
    l1_dinv,
    reconstruct_e,
    reconstruct_e_block,
    rhs_assemble_weighted,
    rhs_assemble_weighted_block,
    select_size4,
    weighted_gradient,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Stiffness assembly needs float64. Placement kernels use float32 to reduce memory use.
RESIDENT_G_DTYPE = cp.float32

# Stopping criterion of the mixed PCG: ||r||_2 / ||b||_2 <= DEFAULT_TOLERANCE. At 1e-6 the
# field is accurate to ~1.4e-6 relative against a 1e-10 reference, at ~59 iterations.
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_ITERS = 2000


def _csr_f32(a: csp.csr_matrix) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """``(indptr, indices, data)`` in the int32/fp32 layout the native kernels take."""
    return (
        cp.ascontiguousarray(a.indptr, dtype=cp.int32),
        cp.ascontiguousarray(a.indices, dtype=cp.int32),
        cp.ascontiguousarray(a.data, dtype=cp.float32),
    )


def _l1_dinv(a: csp.csr_matrix) -> cp.ndarray:
    """l1-Jacobi smoother scaling: 1 / d, with d_i = sign(a_ii) * sum_j |a_ij|.

    The diagonal is included in the L1 norm and the sign follows the diagonal, which is what
    keeps the smoother positive definite and so the whole V-cycle a valid PCG preconditioner.

    Including the diagonal in the row sum is also what bounds the smoother's spectrum. For SPD
    ``a``, ``D - A`` is weakly diagonally dominant with a non-negative diagonal, hence PSD, so
    ``rho(D^-1 A) <= 1`` on every level with no estimate needed. Both the Chebyshev interval in
    vcycle.cu and the damping below rest on that bound.

    The row sum is a kernel rather than an SpMV of ``|a|`` against a vector of ones so that its
    summation order is fixed by the CSR: dinv scales every entry of the smoothed prolongator and
    is uploaded to every V-cycle level, so any drift in it reaches the whole hierarchy.
    See l1.cu.
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
    counts = cp.bincount(agg, minlength=n_coarse).astype(cp.float32)
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
    sweep, and degree 2 buys back more than that in iterations. Cost is flat in ``lower_ratio``
    from 8 to 12 and rises either side, gently.
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
    """One coarsening step of the hierarchy."""

    a: csp.csr_matrix
    aggregates: cp.ndarray
    p: csp.csr_matrix

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
        agg, n_coarse = _aggregate(a, params, stream)
        if agg is None or n_coarse == n_rows or n_coarse < params.min_coarse_rows:
            break
        p = _smooth_prolongator(a, _tentative_prolongator(agg, n_coarse), _l1_dinv(a))
        levels.append(AggregationLevel(a, agg, p))
        a = _galerkin(a, p)
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
    """
    if smoother is None:
        smoother = SmootherParams()
    levels, coarse = aggregation_levels(row_ptr, col_idx, values_f32, params)
    vc = NativeVCycle()
    vc.set_smoother(smoother.degree, smoother.lower_ratio)
    for level in levels:
        a, p = level.a, level.p
        r = p.T.tocsr()
        r.sort_indices()
        vc.add_level(
            cp.ascontiguousarray(a.indptr.astype(cp.int32)),
            cp.ascontiguousarray(a.indices.astype(cp.int32)),
            cp.ascontiguousarray(a.data.astype(cp.float32)),
            _l1_dinv(a),
            cp.ascontiguousarray(p.indptr.astype(cp.int32)),
            cp.ascontiguousarray(p.indices.astype(cp.int32)),
            cp.ascontiguousarray(p.data.astype(cp.float32)),
            cp.ascontiguousarray(r.indptr.astype(cp.int32)),
            cp.ascontiguousarray(r.indices.astype(cp.int32)),
            cp.ascontiguousarray(r.data.astype(cp.float32)),
        )
    ainv = cp.linalg.inv(coarse.todense().astype(cp.float32))
    vc.set_coarse(cp.ascontiguousarray(ainv.astype(cp.float32)))
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


def spatial_order(nodes_mm: cp.ndarray, tet_nodes: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Node and tetrahedron permutations that lay a mesh out along a Morton curve.

    Nodes are ordered by position and tetrahedra by their lowest renumbered node. Gmsh order
    bears no relation to position: on a head mesh the four node indices of a tetrahedron span
    ~380k, which is what every element-centric gather pays, and ~9k once ordered.

    A mesh already in this order sorts to the identity, so reordering is idempotent.
    """
    node_perm = morton_order(nodes_mm)
    lowest = invert_permutation(node_perm)[tet_nodes].min(axis=1)
    tet_perm = cp.argsort(lowest, kind="stable").astype(cp.int32)
    return node_perm, tet_perm


def solve_ordering(nodes_mm: cp.ndarray, ground_node: int) -> cp.ndarray:
    """Reduced-system row index: the mesh nodes bar the grounded one.

    The rows inherit the mesh's spatial order (see :func:`spatial_order`), which is what keeps
    the SpMV gather over ``x`` local: 92% of the reduced operator's nonzeros sit within a
    4096-column window, against a quarter of them on an unordered mesh.
    """
    return grounded_index(int(nodes_mm.shape[0]), ground_node)


def reduce_matrix(a: csp.csr_matrix, idx: cp.ndarray) -> csp.csr_matrix:
    """Remove the grounded DOF and canonicalise the reduced CSR."""
    a_red = a[idx][:, idx].tocsr()
    a_red.sum_duplicates()
    return a_red


class SolverConvergenceError(RuntimeError):
    """The mixed PCG missed tolerance even after rebuilding the preconditioner."""

    def __init__(self, iterations: int, relative_residual: float, tolerance: float) -> None:
        super().__init__(
            f"PCG did not converge: {iterations} iterations, relative residual "
            f"{relative_residual:.3e} > tolerance {tolerance:.3e}, after rebuilding the "
            f"preconditioner at the current matrix values"
        )
        self.iterations = iterations
        self.relative_residual = relative_residual
        self.tolerance = tolerance


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
            cp.ascontiguousarray(self.values.astype(cp.float32)),
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
    idx = solve_ordering(nodes_mm, ground_node)
    a_red = reduce_matrix(a, idx)
    row_ptr = cp.ascontiguousarray(a_red.indptr.astype(cp.int32))
    col_idx = cp.ascontiguousarray(a_red.indices.astype(cp.int32))
    values = cp.ascontiguousarray(a_red.data.astype(cp.float64))
    values_f32 = cp.ascontiguousarray(values.astype(cp.float32))
    precond = build_native_vcycle(row_ptr, col_idx, values_f32)
    pcg = PcgAmgSolver(row_ptr, col_idx, values)
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


# Compiled block widths of the k-RHS solve kernels; smaller batches pad up by
# replicating the last column. The padded column costs bandwidth but no extra matrix reads.
# Read from the extension rather than restated, so adding a width is a one-place change.
BLOCK_SIZES = _BLOCK_SIZES
MAX_BLOCK = BLOCK_SIZES[-1]


def _solve_grounded_block_mat(solver: GroundedSolver, B: cp.ndarray, k: int) -> cp.ndarray:
    """Block-solve a padded (n_red, k_pad) RHS matrix; returns X with retries applied.

    Each of the k chains is numerically independent: per-column reductions, and a column that
    reaches tolerance freezes rather than being carried along to the block's slowest one, so it
    stops exactly where it would have solved alone. They share only the stiffness and hierarchy
    reads. If any column misses tolerance the preconditioner is rebuilt once, then only the
    offending columns are re-solved singly.
    """
    X = cp.empty_like(B)
    stream = cp.cuda.get_current_stream().ptr
    iters, rels = solver.pcg.solve_mixed_block(
        solver.precond, B, X, solver.tolerance, solver.max_iters, stream
    )
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
                raise SolverConvergenceError(int(retry_iters), float(rel), solver.tolerance)
            X[:, c] = x_red
            solver.last_iterations = max(solver.last_iterations, int(retry_iters))
            worst = max(worst, float(rel))
        # The retried columns now bound the block: every other column already met tolerance.
        solver.last_relative_residual = max(
            worst, max((rels[c] for c in range(k) if c not in missed), default=0.0)
        )
    return X


def _pad_block(B: cp.ndarray, k: int) -> cp.ndarray:
    """Pad the (n_red, k) RHS matrix to a compiled block width by replicating a column."""
    k_pad = next(s for s in BLOCK_SIZES if s >= k)
    if k_pad == k:
        return cp.ascontiguousarray(B)
    padded = cp.empty((int(B.shape[0]), k_pad), dtype=B.dtype)
    padded[:, :k] = B
    for c in range(k, k_pad):
        padded[:, c] = B[:, k - 1]
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


def _assemble_rhs_weighted_kernel(
    dadt_elm: cp.ndarray,
    wg: cp.ndarray,
    node2corner_ptr: cp.ndarray,
    node2corner_idx: cp.ndarray,
    n_nodes: int,
) -> cp.ndarray:
    b = cp.empty(n_nodes, dtype=cp.float32)
    rhs_assemble_weighted(
        cp.ascontiguousarray(dadt_elm),
        wg,
        node2corner_ptr,
        node2corner_idx,
        b,
        cp.cuda.get_current_stream().ptr,
    )
    return b


def _weighted_gradient_kernel(g: cp.ndarray, neg_vc: cp.ndarray) -> cp.ndarray:
    wg = cp.empty_like(g)
    weighted_gradient(g, neg_vc, wg, cp.cuda.get_current_stream().ptr)
    return wg


def _reconstruct_e_kernel(
    v: cp.ndarray,
    tet_nodes: cp.ndarray,
    g: cp.ndarray,
    dadt_elm: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Reconstruct E and its magnitude for each tetrahedron.

    Accumulate ``grad(v)`` in float64 because ``-grad(v) - dA/dt`` has strong cancellation.
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
    wg: cp.ndarray
    vols: cp.ndarray
    neg_vc: cp.ndarray
    solver: GroundedSolver
    node2corner_ptr: cp.ndarray
    node2corner_idx: cp.ndarray
    skin_a: cp.ndarray
    skin_b: cp.ndarray
    skin_c: cp.ndarray
    skin_tri_normals: cp.ndarray
    # Filled on first use by ensure_recovery, never by build_context, so a run that asks for no
    # recovery pays neither the build nor the memory. Subject populates it under its own lock.
    recovery: dict[Recovery, RecoveryOperator] = field(default_factory=dict)


def _spatially_ordered(
    mesh: HeadMesh,
) -> tuple[HeadMesh, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Reorder a mesh along a Morton curve and return it with its device arrays.

    The returned mesh is the order every array in the context carries, host and device alike,
    and so the one a caller reads results against. Surface rows keep their file order; only the
    node ids they name move.
    """
    nodes_mm = cp.asarray(mesh.nodes_mm)
    tet_nodes = cp.asarray(mesh.tet_nodes)
    node_perm, tet_perm = spatial_order(nodes_mm, tet_nodes)
    inverse = invert_permutation(node_perm)

    nodes_mm = cp.ascontiguousarray(nodes_mm[node_perm])
    tet_nodes = cp.ascontiguousarray(inverse[tet_nodes[tet_perm]])
    tet_tags = cp.ascontiguousarray(cp.asarray(mesh.tet_tags)[tet_perm])
    skin_tris = cp.ascontiguousarray(inverse[cp.asarray(mesh.skin_tris)])

    ordered = HeadMesh(
        nodes_mm=cp.asnumpy(nodes_mm),
        tet_nodes=cp.asnumpy(tet_nodes),
        tet_tags=cp.asnumpy(tet_tags),
        skin_tris=cp.asnumpy(skin_tris),
    )
    return ordered, nodes_mm, tet_nodes, tet_tags, skin_tris


def build_context(mesh: HeadMesh) -> SolverContext:
    """Build the GPU state shared by all placements.

    The mesh is first laid out along a Morton curve, and ``ctx.mesh`` is that reordering: it is
    what every array here, and every field a solve returns, is indexed by.

    Assemble ``g``, volumes, and stiffness in float64. Store ``g``, volumes, and
    ``-volume * conductivity`` in float32 for placement kernels.
    """
    mesh, nodes_mm, tet_nodes, tet_tags, skin_tris = _spatially_ordered(mesh)

    g, vols = gradient_operator(nodes_mm, tet_nodes)
    cond = conductivity_per_tet(tet_tags)
    stiffness = assemble_stiffness(g, vols, cond, mesh.n_nodes, tet_nodes)
    ground_node = ground_node_of(nodes_mm)
    solver = prepare_grounded_solver(stiffness, nodes_mm, ground_node)
    del stiffness

    g = cp.ascontiguousarray(g.astype(RESIDENT_G_DTYPE))
    neg_vc = cp.ascontiguousarray(-(vols.astype(cp.float32) * cond.astype(cp.float32)))
    wg = _weighted_gradient_kernel(g, neg_vc)
    vols = cp.ascontiguousarray(vols.astype(cp.float32))
    del cond
    ptr, idx = build_node2corner(tet_nodes, mesh.n_nodes)

    skin_a = cp.ascontiguousarray(nodes_mm[skin_tris[:, 0]])
    skin_b = cp.ascontiguousarray(nodes_mm[skin_tris[:, 1]])
    skin_c = cp.ascontiguousarray(nodes_mm[skin_tris[:, 2]])
    skin_tri_normals = cp.asarray(mesh.skin_triangle_normals)
    ctx = SolverContext(
        mesh,
        nodes_mm,
        tet_nodes,
        tet_tags,
        mesh.n_nodes,
        g,
        wg,
        vols,
        neg_vc,
        solver,
        ptr,
        idx,
        skin_a,
        skin_b,
        skin_c,
        skin_tri_normals,
    )
    cp.get_default_memory_pool().free_all_blocks()
    return ctx


class PlacementResult(TypedDict):
    """Arrays produced for one placement.

    ``E``/``magnE`` are whichever field the run's ``recovery`` mode asked for. ``E_slots`` is
    present only when a recovery mode ran and the caller asked for it.
    """

    transform: npt.NDArray[np.float64]
    dadt_elm: cp.ndarray
    E: cp.ndarray
    magnE: cp.ndarray
    v: cp.ndarray
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
    pos_ydir_mm: npt.ArrayLike,
    distance_mm: float,
    didt: float,
    *,
    recovery: RecoveryOperator | None = None,
    nodal: bool = False,
) -> PlacementResult:
    """Solve one placement and return device arrays plus the host transform.

    ``recovery`` is a prebuilt operator from :func:`~cunibs.fem.ensure_recovery`, or ``None``
    for the raw per-tetrahedron field; see :func:`_check_operator` for why this layer will not
    build one. ``nodal`` additionally keeps the recovered field on its slots.
    """
    _check_operator(ctx, recovery, nodal, "solve_placement")

    transform = compute_coil_transform(ctx, center_mm, pos_ydir_mm, distance_mm)
    dadt_nodes = coil_dadt_at_nodes(dip_pos_m, dip_moment, transform, didt, ctx.nodes_mm)
    dadt_elm = _dadt_node_to_elm(dadt_nodes, ctx.tet_nodes)

    b = _assemble_rhs_weighted_kernel(
        dadt_elm,
        ctx.wg,
        ctx.node2corner_ptr,
        ctx.node2corner_idx,
        ctx.n_nodes,
    )

    v = solve_grounded(ctx.solver, b)

    n_tet = int(ctx.tet_nodes.shape[0])
    # The potential path never reads the raw field, so reconstructing it there would be a full
    # mesh pass and an (n_tet, 3) + (n_tet,) transient thrown straight away.
    on_potential = recovery is not None and recovery.on_potential
    e = magn_e = slots = None
    if not on_potential:
        e, magn_e = _reconstruct_e_kernel(v, ctx.tet_nodes, ctx.g, dadt_elm)
    if recovery is not None:
        rec = apply_recovery(
            recovery,
            n_tet,
            elements=None if on_potential else [e],
            potential=cp.ascontiguousarray(v.reshape(-1, 1)) if on_potential else None,
            dadt_nodes=[cp.ascontiguousarray(dadt_nodes)] if on_potential else None,
        )
        e, magn_e, slots = rec.E[0], rec.magnE[0], rec.E_slots[0]

    result: PlacementResult = {
        "transform": transform,
        "dadt_elm": dadt_elm,
        "E": e,
        "magnE": magn_e,
        "v": v,
    }
    if nodal:
        result["E_slots"] = slots
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
) -> list[PlacementResult]:
    """Solve up to MAX_BLOCK placements with block kernels end to end.

    ``sites`` are (center_mm, pos_ydir_mm, distance_mm) triples. All stages that read
    large placement-independent arrays (scalp projection, tet connectivity, weighted
    gradients, node2corner, g) run once per chunk rather than once per placement, and the
    linear solve shares every stiffness/hierarchy read across the k columns.

    k = 1 runs the width-1 block kernels rather than routing to ``solve_placement``, so that
    every block_k a caller can pick reaches the same arithmetic.

    ``recovery`` is a prebuilt operator from :func:`~cunibs.fem.ensure_recovery`, or ``None``
    for the raw per-tetrahedron field; see :func:`_check_operator` for why this layer will not
    build one. It is another placement-independent array read once per chunk, and its per-slot
    reduction order does not depend on k, so a placement's recovered field is the same at every
    block width.
    """
    k = len(sites)
    solver = ctx.solver
    if k > MAX_BLOCK:
        raise ValueError(f"solve_placements_block: k={k} exceeds MAX_BLOCK={MAX_BLOCK}")
    _check_operator(ctx, recovery, nodal, "solve_placements_block")
    on_potential = recovery is not None and recovery.on_potential
    stream = cp.cuda.get_current_stream().ptr
    n_tet = int(ctx.tet_nodes.shape[0])

    transforms = cp.asnumpy(
        compute_coil_transforms(
            ctx,
            np.asarray([s[0] for s in sites], dtype=np.float64).reshape(k, 3),
            np.asarray([s[1] for s in sites], dtype=np.float64).reshape(k, 3),
            np.asarray([s[2] for s in sites], dtype=np.float64),
        )
    )

    # dA/dt node->element stays per placement: a k-wide version would gather from k separate
    # nodal arrays at once and overflow L2, which costs more than the shared tet_nodes read
    # it would amortize.
    dadt_elms = []
    # A potential-consuming operator adds dA/dt back at the nodes, where it is exact, so that
    # path keeps the nodal field the others average away.
    dadt_nodal: list[cp.ndarray] = []
    for i in range(k):
        dadt_nodes = coil_dadt_at_nodes(
            dip_pos_m, dip_moment, transforms[i], didt, ctx.nodes_mm
        )
        dadt_elms.append(_dadt_node_to_elm(dadt_nodes, ctx.tet_nodes))
        if on_potential:
            dadt_nodal.append(cp.ascontiguousarray(dadt_nodes))
        del dadt_nodes

    b_block = cp.empty((ctx.n_nodes, k), dtype=cp.float32)
    rhs_assemble_weighted_block(
        dadt_elms,
        ctx.wg,
        ctx.node2corner_ptr,
        ctx.node2corner_idx,
        b_block,
        stream,
    )

    B = cp.ascontiguousarray(b_block[solver.idx, :].astype(cp.float64))
    B_pad = _pad_block(B, k)
    X = _solve_grounded_block_mat(solver, B_pad, k)

    v_block = cp.zeros((solver.n, k), dtype=cp.float64)
    v_block[solver.idx, :] = X[:, :k]
    v_block = cp.ascontiguousarray(v_block)

    # The potential path never reads the raw field, so it neither reconstructs nor allocates one.
    es: list[cp.ndarray] = []
    magns: list[cp.ndarray] = []
    if not on_potential:
        es = [cp.empty((n_tet, 3), dtype=cp.float32) for _ in range(k)]
        magns = [cp.empty(n_tet, dtype=cp.float32) for _ in range(k)]
        reconstruct_e_block(v_block, ctx.tet_nodes, ctx.g, dadt_elms, es, magns, stream)

    slots: list[cp.ndarray] = []
    if recovery is not None:
        rec = RecoveredField.allocate(recovery, n_tet, k)
        apply_recovery_into(
            recovery,
            rec,
            elements=None if on_potential else es,
            potential=v_block,
            dadt_nodes=dadt_nodal if on_potential else None,
        )
        slots, es, magns = rec.E_slots, rec.E, rec.magnE

    results: list[PlacementResult] = [
        {
            "transform": transforms[i],
            "dadt_elm": dadt_elms[i],
            "E": es[i],
            "magnE": magns[i],
            "v": cp.ascontiguousarray(v_block[:, i]),
        }
        for i in range(k)
    ]
    if nodal:
        for i, result in enumerate(results):
            result["E_slots"] = slots[i]

    return results
