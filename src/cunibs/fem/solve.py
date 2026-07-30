"""Solve the P1 FEM system for the TMS E-field on the GPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

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
from cunibs.solver import (
    NativeVCycle,
    PcgAmgSolver,
    dadt_node_to_element,
    reconstruct_e,
    reconstruct_e_block,
    rhs_assemble_weighted,
    rhs_assemble_weighted_block,
    select_size4,
    weighted_gradient,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cunibs.mesh import HeadMesh

# Stiffness assembly needs float64. Placement kernels use float32 to reduce memory use.
RESIDENT_G_DTYPE = cp.float32

# Stopping criterion of the mixed PCG: ||r||_2 / ||b||_2 <= DEFAULT_TOLERANCE. At 1e-6 the
# field is accurate to ~1.4e-6 relative against a 1e-10 reference, at ~59 iterations.
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_ITERS = 2000


def _l1_dinv(a: csp.csr_matrix, omega: float = 0.9) -> cp.ndarray:
    """l1-Jacobi smoother scaling: omega / d, with d_i = sign(a_ii) * sum_j |a_ij|.

    The diagonal is included in the L1 norm and the sign follows the diagonal, which is what
    keeps the smoother positive definite and so the whole V-cycle a valid PCG preconditioner.
    ``omega`` is the relaxation factor. The zero-row guard cannot fire for the SPD reduced
    stiffness, but keeps the reciprocal finite for any input.
    """
    abs_a = csp.csr_matrix((cp.abs(a.data), a.indices, a.indptr), shape=a.shape)
    d = abs_a.dot(cp.ones(a.shape[1], dtype=cp.float32))
    d = cp.where(a.diagonal() < 0, -d, d)
    d = cp.where(d != 0, d, cp.float32(1.0))
    return cp.ascontiguousarray((cp.float32(omega) / d).astype(cp.float32))


def _galerkin(a: csp.csr_matrix, agg: cp.ndarray, n_coarse: int) -> csp.csr_matrix:
    """A_{l+1} = P^T A_l P, with P the boolean aggregate map of unsmoothed aggregation."""
    p = csp.csr_matrix(
        (
            cp.ones(a.shape[0], dtype=cp.float32),
            agg,
            cp.arange(a.shape[0] + 1, dtype=cp.int32),
        ),
        shape=(a.shape[0], n_coarse),
    )
    coarse = (p.T.tocsr() @ a @ p).tocsr()
    coarse.sum_duplicates()
    coarse.sort_indices()
    return coarse


@dataclass(frozen=True)
class AggregationParams:
    """Hierarchy shape controls.

    ``min_coarse_rows`` bounds the dense coarse solve, whose inverse costs O(n_coarse^2)
    memory. It applies to the *next* level's size, so the coarsest level sits above it: 164
    rows on sub-001, 457 on the test patch.
    """

    min_coarse_rows: int = 128
    max_levels: int = 50


def aggregation_levels(
    row_ptr: cp.ndarray,
    col_idx: cp.ndarray,
    values_f32: cp.ndarray,
    params: AggregationParams | None = None,
) -> tuple[list[tuple[csp.csr_matrix, cp.ndarray, int]], csp.csr_matrix]:
    """Coarsen until the dense-solve floor; return the per-level operators and the coarsest.

    Each entry is ``(A_l, aggregates_l, n_coarse_l)``, finest first. Coarsening stops when a
    level cannot coarsen at all or the next level would undershoot ``min_coarse_rows``; that
    level is discarded and its matrix becomes the coarsest.
    """
    if params is None:
        params = AggregationParams()
    n = int(row_ptr.shape[0]) - 1
    a = csp.csr_matrix((values_f32, col_idx, row_ptr), shape=(n, n))
    stream = cp.cuda.get_current_stream().ptr
    levels: list[tuple[csp.csr_matrix, cp.ndarray, int]] = []
    for _ in range(params.max_levels):
        n_rows = a.shape[0]
        if n_rows <= params.min_coarse_rows:
            break
        agg = cp.empty(n_rows, dtype=cp.int32)
        n_coarse = select_size4(
            cp.ascontiguousarray(a.indptr.astype(cp.int32)),
            cp.ascontiguousarray(a.indices.astype(cp.int32)),
            cp.ascontiguousarray(a.data.astype(cp.float32)),
            agg,
            stream,
        )
        if n_coarse == n_rows or n_coarse < params.min_coarse_rows:
            break
        levels.append((a, agg, n_coarse))
        a = _galerkin(a, agg, n_coarse)
    return levels, a


def build_native_vcycle(
    row_ptr: cp.ndarray,
    col_idx: cp.ndarray,
    values_f32: cp.ndarray,
    params: AggregationParams | None = None,
) -> NativeVCycle:
    """Build the fp32 V-cycle preconditioner: aggregate, then upload each level's operators.

    Unsmoothed aggregation makes P a boolean map, so the restriction row order is just the
    stable sort by aggregate, which also fixes the reduction order inside the restrict kernel.
    """
    levels, coarse = aggregation_levels(row_ptr, col_idx, values_f32, params)
    vc = NativeVCycle()
    for a, agg, n_coarse in levels:
        order = cp.ascontiguousarray(cp.argsort(agg, kind="stable").astype(cp.int32))
        r_ptr = cp.zeros(n_coarse + 1, dtype=cp.int64)
        cp.cumsum(cp.bincount(agg, minlength=n_coarse), out=r_ptr[1:])
        vc.add_level(
            cp.ascontiguousarray(a.indptr.astype(cp.int32)),
            cp.ascontiguousarray(a.indices.astype(cp.int32)),
            cp.ascontiguousarray(a.data.astype(cp.float32)),
            _l1_dinv(a),
            cp.ascontiguousarray(r_ptr.astype(cp.int32)),
            order,
            agg,
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
    ground_node: int,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> GroundedSolver:
    """Remove the ground DOF and build the mixed-precision solver."""
    n = a.shape[0]
    idx = grounded_index(n, ground_node)
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
BLOCK_SIZES = (2, 4, 8)
MAX_BLOCK = BLOCK_SIZES[-1]


@dataclass
class BlockWarmStart:
    """Carry the previous chunk's solutions to warm-start the next chunk.

    One instance per batched simulate call; solve_placements_block updates it in place.
    """

    centers: np.ndarray | None = None  # (k_prev, 3) placement centers
    x_red: cp.ndarray | None = None  # (n_red, k_prev) f64 reduced solutions


def _warm_x0(
    carry: BlockWarmStart | None, centers: np.ndarray, n_red: int, k: int, k_pad: int
) -> cp.ndarray | None:
    """Build the (n_red, k_pad) initial guess from the nearest prior placements.

    The solver's stopping criterion is ||r||/||b|| <= tol, measured against ||b|| rather than
    the initial residual, so seeding x0 changes only the iteration count and not the accuracy
    of the returned field.
    """
    if carry is None or carry.centers is None:
        return None
    assert carry.x_red is not None
    nearest = np.linalg.norm(centers[:, None, :] - carry.centers[None, :, :], axis=2).argmin(
        axis=1
    )
    x0 = cp.empty((n_red, k_pad), dtype=cp.float64)
    for c in range(k):
        x0[:, c] = carry.x_red[:, int(nearest[c])]
    for c in range(k, k_pad):
        x0[:, c] = x0[:, k - 1]
    return cp.ascontiguousarray(x0)


def _solve_grounded_block_mat(
    solver: GroundedSolver, B: cp.ndarray, k: int, x0: cp.ndarray | None = None
) -> cp.ndarray:
    """Block-solve a padded (n_red, k_pad) RHS matrix; returns X with retries applied.

    Each of the k chains is numerically independent (per-column reductions), they just
    share every stiffness/hierarchy matrix read. If any column misses tolerance the
    preconditioner is rebuilt once, then only the offending columns are re-solved singly.
    """
    X = cp.empty_like(B)
    stream = cp.cuda.get_current_stream().ptr
    iters, rels = solver.pcg.solve_mixed_block(
        solver.precond, B, X, solver.tolerance, solver.max_iters, stream, x0
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


def build_context(mesh: HeadMesh) -> SolverContext:
    """Build the GPU state shared by all placements.

    Assemble ``g``, volumes, and stiffness in float64. Store ``g``, volumes, and
    ``-volume * conductivity`` in float32 for placement kernels.
    """
    nodes_mm = cp.asarray(mesh.nodes_mm)
    tet_nodes = cp.asarray(mesh.tet_nodes)
    tet_tags = cp.asarray(mesh.tet_tags)

    g, vols = gradient_operator(nodes_mm * 1e-3, tet_nodes)
    cond = conductivity_per_tet(tet_tags)
    stiffness = assemble_stiffness(g, vols, cond, mesh.n_nodes, tet_nodes)
    ground_node = ground_node_of(nodes_mm)
    solver = prepare_grounded_solver(stiffness, ground_node)
    del stiffness

    g = cp.ascontiguousarray(g.astype(RESIDENT_G_DTYPE))
    neg_vc = cp.ascontiguousarray(-(vols.astype(cp.float32) * cond.astype(cp.float32)))
    wg = _weighted_gradient_kernel(g, neg_vc)
    vols = cp.ascontiguousarray(vols.astype(cp.float32))
    del cond
    ptr, idx = build_node2corner(tet_nodes, mesh.n_nodes)

    skin_tris = cp.asarray(mesh.skin_tris)
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
    """Arrays produced for one placement."""

    transform: npt.NDArray[np.float64]
    dadt_elm: cp.ndarray
    E: cp.ndarray
    magnE: cp.ndarray
    v: cp.ndarray


def solve_placement(
    ctx: SolverContext,
    dip_pos_m: npt.ArrayLike,
    dip_moment: npt.ArrayLike,
    center_mm: npt.ArrayLike,
    pos_ydir_mm: npt.ArrayLike,
    distance_mm: float,
    didt: float,
) -> PlacementResult:
    """Solve one placement and return device arrays plus the host transform."""
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

    e, magn_e = _reconstruct_e_kernel(v, ctx.tet_nodes, ctx.g, dadt_elm)

    return {
        "transform": transform,
        "dadt_elm": dadt_elm,
        "E": e,
        "magnE": magn_e,
        "v": v,
    }


def solve_placements_block(
    ctx: SolverContext,
    dip_pos_m: npt.ArrayLike,
    dip_moment: npt.ArrayLike,
    sites: Sequence[tuple[npt.ArrayLike, npt.ArrayLike, float]],
    didt: float,
    warm: BlockWarmStart | None = None,
) -> list[PlacementResult]:
    """Solve up to MAX_BLOCK placements with block kernels end to end.

    ``sites`` are (center_mm, pos_ydir_mm, distance_mm) triples. All stages that read
    large placement-independent arrays (scalp projection, tet connectivity, weighted
    gradients, node2corner, g) run once per chunk rather than once per placement, and the
    linear solve shares every stiffness/hierarchy read across the k columns.
    """
    k = len(sites)
    solver = ctx.solver
    if k == 1:
        center_mm, pos_ydir_mm, distance_mm = sites[0]
        return [
            solve_placement(
                ctx, dip_pos_m, dip_moment, center_mm, pos_ydir_mm, distance_mm, didt
            )
        ]
    if k > MAX_BLOCK:
        raise ValueError(f"solve_placements_block: k={k} exceeds MAX_BLOCK={MAX_BLOCK}")
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
    for i in range(k):
        dadt_nodes = coil_dadt_at_nodes(
            dip_pos_m, dip_moment, transforms[i], didt, ctx.nodes_mm
        )
        dadt_elms.append(_dadt_node_to_elm(dadt_nodes, ctx.tet_nodes))
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
    centers = np.asarray([np.asarray(s[0], dtype=np.float64).reshape(3) for s in sites])
    x0 = _warm_x0(warm, centers, int(B.shape[0]), k, int(B_pad.shape[1]))
    X = _solve_grounded_block_mat(solver, B_pad, k, x0)
    if warm is not None:
        warm.centers = centers
        warm.x_red = X[:, :k]

    v_block = cp.zeros((solver.n, k), dtype=cp.float64)
    v_block[solver.idx, :] = X[:, :k]
    v_block = cp.ascontiguousarray(v_block)

    es = [cp.empty((n_tet, 3), dtype=cp.float32) for _ in range(k)]
    magns = [cp.empty(n_tet, dtype=cp.float32) for _ in range(k)]
    reconstruct_e_block(v_block, ctx.tet_nodes, ctx.g, dadt_elms, es, magns, stream)

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

    return results
