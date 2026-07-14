"""Solve the P1 FEM system for the TMS E-field on the GPU."""

from __future__ import annotations

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
from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform
from cunibs.mesh import HeadMesh
from cunibs.solver import (
    AMGXFloatSolver,
    AMGXSolver,
    PcgAmgSolver,
    dadt_node_to_element,
    reconstruct_e,
    rhs_assemble,
    rhs_assemble_weighted,
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
    float_preconditioner: AMGXFloatSolver
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
    float_preconditioner = AMGXFloatSolver(AMGX_PRECONDITIONER_CONFIG)
    float_preconditioner.setup(
        row_ptr, col_idx, cp.ascontiguousarray(values.astype(cp.float32))
    )
    pcg = PcgAmgSolver(row_ptr, col_idx, values)
    tolerance = float(_amgx_config_value(config, "tolerance", "1e-6"))
    max_iters = int(_amgx_config_value(config, "max_iters", "2000"))
    return GroundedSolver(
        n=n,
        idx=idx,
        float_preconditioner=float_preconditioner,
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
        solver.float_preconditioner,
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
