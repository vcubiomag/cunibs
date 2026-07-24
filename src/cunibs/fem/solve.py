"""Solve the P1 FEM system for the TMS E-field on the GPU."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

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
from cunibs.mesh import HeadMesh
from cunibs.solver import (
    AMGXFloatSolver,
    AMGXSolver,
    NativeVCycle,
    PcgAmgSolver,
    dadt_node_to_element,
    reconstruct_e,
    reconstruct_e_block,
    rhs_assemble,
    rhs_assemble_weighted,
    rhs_assemble_weighted_block,
    weighted_gradient,
)

# Stiffness assembly needs float64. Placement kernels use float32 to reduce memory use.
RESIDENT_G_DTYPE = cp.float32

# PCG requires a symmetric preconditioner. MULTICOLOR_GS and DILU stall near 1e-5.
# JACOBI_L1 reaches the 1e-6 target across the tissue conductivity range.
AMGX_CONFIG = (
    "config_version=2, determinism_flag=1, solver=PCG, tolerance=1e-6, max_iters=2000, "
    "norm=L2, convergence=RELATIVE_INI_CORE, monitor_residual=1, "
    "preconditioner(amg)=AMG, amg:algorithm=AGGREGATION, amg:selector=SIZE_2, "
    "amg:smoother=JACOBI_L1, amg:presweeps=1, amg:postsweeps=1, amg:max_iters=1, "
    "amg:cycle=V, amg:coarse_solver=DENSE_LU_SOLVER, amg:min_coarse_rows=32, amg:max_levels=50"
)

# Reuse the aggregation graph across resetups when only the matrix values change (e.g. a
# conductivity Monte Carlo). resetup then rebuilds only the Galerkin operators and smoothers.
UQ_AMGX_CONFIG = AMGX_CONFIG + ", structure_reuse_levels=-1"

AMGX_PRECONDITIONER_CONFIG = (
    "config_version=2, determinism_flag=1, solver=AMG, max_iters=1, "
    "monitor_residual=0, algorithm=AGGREGATION, selector=SIZE_4, "
    "smoother=JACOBI_L1, presweeps=1, postsweeps=1, cycle=V, "
    "coarse_solver=DENSE_LU_SOLVER, min_coarse_rows=32, max_levels=50"
)


def _amgx_config_value(config: str, key: str, default: str) -> str:
    prefix = f"{key}="
    for part in config.split(","):
        item = part.strip()
        if item.startswith(prefix):
            return item[len(prefix) :]
    return default


def _l1_dinv(a: csp.csr_matrix, omega: float = 0.9) -> cp.ndarray:
    """AMGx JACOBI_L1 smoother scaling: omega / d, d_i = sign(a_ii) * sum_j |a_ij|.

    Matches the fork's compute_d_kernel (diagonal included in the L1 norm, sign flipped
    for negative diagonals) with the default relaxation_factor 0.9 folded in. The
    zero-guard never triggers for the SPD reduced stiffness but is kept for parity.
    """
    abs_a = csp.csr_matrix((cp.abs(a.data), a.indices, a.indptr), shape=a.shape)
    d = abs_a.dot(cp.ones(a.shape[1], dtype=cp.float32))
    d = cp.where(a.diagonal() < 0, -d, d)
    d = cp.where(d != 0, d, cp.float32(1.0))
    return cp.ascontiguousarray((cp.float32(omega) / d).astype(cp.float32))


def build_native_vcycle(
    float_preconditioner: AMGXFloatSolver,
    row_ptr: cp.ndarray,
    col_idx: cp.ndarray,
    values_f32: cp.ndarray,
) -> NativeVCycle:
    """Rebuild the AMGx preconditioner's V-cycle operators for the native apply.

    AMGx contributes only the per-level aggregate maps (the hard part of setup); the
    Galerkin products, l1-Jacobi diagonals, restriction ordering and the dense coarse
    inverse are recomputed here. Unsmoothed aggregation makes P a boolean map, so
    A_{l+1} = P^T A_l P and the restriction row order is the stable sort by aggregate,
    matching the fork's computeRestrictionOperator.
    """
    n_levels = int(float_preconditioner.amg_num_levels())
    n = int(row_ptr.shape[0]) - 1
    a = csp.csr_matrix((values_f32, col_idx, row_ptr), shape=(n, n))
    vc = NativeVCycle()
    for level in range(n_levels - 1):
        n_rows, _, n_coarse = float_preconditioner.amg_level_dims(level)
        if n_rows != a.shape[0]:
            raise RuntimeError(
                f"native V-cycle: level {level} has {n_rows} rows in AMGx but "
                f"{a.shape[0]} in the recomputed Galerkin chain"
            )
        agg = cp.empty(n_rows, dtype=cp.int32)
        float_preconditioner.download_aggregates(level, agg)
        order = cp.ascontiguousarray(cp.argsort(agg, kind="stable").astype(cp.int32))
        counts = cp.bincount(agg, minlength=n_coarse)
        r_ptr = cp.zeros(n_coarse + 1, dtype=cp.int64)
        cp.cumsum(counts, out=r_ptr[1:])
        r_ptr = cp.ascontiguousarray(r_ptr.astype(cp.int32))
        vc.add_level(
            cp.ascontiguousarray(a.indptr.astype(cp.int32)),
            cp.ascontiguousarray(a.indices.astype(cp.int32)),
            cp.ascontiguousarray(a.data.astype(cp.float32)),
            _l1_dinv(a),
            r_ptr,
            order,
            agg,
        )
        p = csp.csr_matrix(
            (
                cp.ones(n_rows, dtype=cp.float32),
                agg,
                cp.arange(n_rows + 1, dtype=cp.int32),
            ),
            shape=(n_rows, n_coarse),
        )
        a = (p.T.tocsr() @ a @ p).tocsr()
        a.sum_duplicates()
        a.sort_indices()
    ainv = cp.linalg.inv(a.todense().astype(cp.float32))
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


@dataclass
class GroundedSolver:
    """Reduced SPD system with one grounded potential.

    Grounding removes the free additive constant in ``v``. It does not change
    ``E = -grad(v) - dA/dt``.
    """

    n: int
    idx: cp.ndarray
    # The fp32 apply inside solve_mixed: the native V-cycle rebuilt from the exported
    # AMGx hierarchy (the AMGx float solver itself is dropped right after the export).
    precond: NativeVCycle
    pcg: PcgAmgSolver
    tolerance: float
    max_iters: int
    # Retained to build the fp64 fallback lazily (see ``ensure_amgx``).
    config: str
    row_ptr: cp.ndarray
    col_idx: cp.ndarray
    values: cp.ndarray
    amgx: AMGXSolver | None = None
    last_iterations: int = 0
    last_relative_residual: float = 0.0

    def ensure_amgx(self) -> AMGXSolver:
        """Build the fp64 AMGx fallback solver on first use.

        The fallback only runs when the mixed-precision PCG misses tolerance, so building its full
        double AMG hierarchy eagerly holds a second (rarely used) hierarchy on the device for every
        subject's lifetime. Deferring it raises how many subjects fit on one GPU. The reduced CSR is
        already resident, so this only pays the one-time AMGx setup, not a reassembly.
        """
        if self.amgx is None:
            amgx = AMGXSolver(self.config)
            amgx.setup(self.row_ptr, self.col_idx, self.values)
            self.amgx = amgx
        return self.amgx


def prepare_grounded_solver(
    a: csp.csr_matrix, ground_node: int, config: str = AMGX_CONFIG
) -> GroundedSolver:
    """Remove the ground DOF and build the mixed-precision solver (fp64 fallback built lazily)."""
    n = a.shape[0]
    idx = grounded_index(n, ground_node)
    a_red = reduce_matrix(a, idx)
    row_ptr = cp.ascontiguousarray(a_red.indptr.astype(cp.int32))
    col_idx = cp.ascontiguousarray(a_red.indices.astype(cp.int32))
    values = cp.ascontiguousarray(a_red.data.astype(cp.float64))
    values_f32 = cp.ascontiguousarray(values.astype(cp.float32))
    # AMGx only builds the aggregation hierarchy; the export-derived native V-cycle owns
    # device copies of everything it needs, so the AMGx float solver is dropped at
    # return, releasing its matrix + hierarchy (~160 MB per subject).
    amgx_precond = AMGXFloatSolver(AMGX_PRECONDITIONER_CONFIG)
    amgx_precond.setup(row_ptr, col_idx, values_f32)
    precond = build_native_vcycle(amgx_precond, row_ptr, col_idx, values_f32)
    del amgx_precond
    pcg = PcgAmgSolver(row_ptr, col_idx, values)
    tolerance = float(_amgx_config_value(config, "tolerance", "1e-6"))
    max_iters = int(_amgx_config_value(config, "max_iters", "2000"))
    return GroundedSolver(
        n=n,
        idx=idx,
        precond=precond,
        pcg=pcg,
        tolerance=tolerance,
        max_iters=max_iters,
        config=config,
        row_ptr=row_ptr,
        col_idx=col_idx,
        values=values,
    )


def solve_grounded(solver: GroundedSolver, b: cp.ndarray) -> cp.ndarray:
    """Solve one RHS on the prepared hierarchy."""
    b_red = cp.ascontiguousarray(b[solver.idx], dtype=cp.float64)
    x_red = cp.empty(int(solver.idx.shape[0]), dtype=cp.float64)
    iters, rel = solver.pcg.solve_mixed(
        solver.precond,
        b_red,
        x_red,
        solver.tolerance,
        solver.max_iters,
        cp.cuda.get_current_stream().ptr,
    )
    solver.last_iterations = int(iters)
    solver.last_relative_residual = float(rel)
    if solver.last_relative_residual > solver.tolerance:
        amgx = solver.ensure_amgx()
        amgx.solve(b_red, x_red, cp.cuda.get_current_stream().ptr)
        solver.last_iterations = amgx.iterations()
        solver.last_relative_residual = 0.0
    v = cp.zeros(solver.n, dtype=cp.float64)
    v[solver.idx] = x_red
    return v


# Compiled block widths of the k-RHS solve kernels; smaller batches pad up by
# replicating the last column (the padded column costs bandwidth but no extra
# matrix reads, which is the point of the block path).
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

    Warm and cold starts converge to the same ||r||/||b|| <= tol criterion (the
    reference norm is ||b||, not the warm residual), so fields stay within the solver
    tolerance either way; a warm start only changes the iteration count.
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
    """Block-solve a padded (n_red, k_pad) RHS matrix; returns X with fallbacks applied.

    Each of the k chains is numerically independent (per-column reductions), they just
    share every stiffness/hierarchy matrix read. Columns whose lockstep residual misses
    tolerance are re-solved individually by the fp64 AMGx fallback.
    """
    X = cp.empty_like(B)
    iters, rels = solver.pcg.solve_mixed_block(
        solver.precond,
        B,
        X,
        solver.tolerance,
        solver.max_iters,
        cp.cuda.get_current_stream().ptr,
        x0,
    )
    solver.last_iterations = int(iters)
    solver.last_relative_residual = float(max(rels[:k]))
    for c in range(k):
        if rels[c] > solver.tolerance:
            amgx = solver.ensure_amgx()
            b_red = cp.ascontiguousarray(B[:, c])
            x_red = cp.empty(int(B.shape[0]), dtype=cp.float64)
            amgx.solve(b_red, x_red, cp.cuda.get_current_stream().ptr)
            X[:, c] = x_red
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


def _assemble_rhs_kernel(
    dadt_elm: cp.ndarray,
    g: cp.ndarray,
    neg_vc: cp.ndarray,
    node2corner_ptr: cp.ndarray,
    node2corner_idx: cp.ndarray,
    n_nodes: int,
) -> cp.ndarray:
    """Assemble the float32 RHS in a fixed reduction order."""
    b = cp.empty(n_nodes, dtype=cp.float32)
    rhs_assemble(
        cp.ascontiguousarray(dadt_elm),
        g,
        neg_vc,
        node2corner_ptr,
        node2corner_idx,
        b,
        cp.cuda.get_current_stream().ptr,
    )
    return b


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
    # Negating after multiplication preserves the previous IEEE rounding order.
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
    gradients, node2corner, g) run once per chunk instead of once per placement; the
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

    # dA/dt node->element stays per placement: a k-wide version gathers from k separate
    # 8.4 MB nodal arrays at once, which overflows L2 and measured 3x SLOWER than k
    # serial passes (the shared tet_nodes read it would amortize is the smaller cost).
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

    results: list[PlacementResult] = []
    for i in range(k):
        results.append(
            {
                "transform": transforms[i],
                "dadt_elm": dadt_elms[i],
                "E": es[i],
                "magnE": magns[i],
                "v": cp.ascontiguousarray(v_block[:, i]),
            }
        )
    return results
