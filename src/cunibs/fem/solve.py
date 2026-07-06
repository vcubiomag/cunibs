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
from cunibs.solver import AMGXSolver, reconstruct_e, rhs_assemble

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


@dataclass
class GroundedSolver:
    """Reduced SPD system with one grounded potential.

    Grounding removes the free additive constant in ``v``. It does not change
    ``E = -grad(v) - dA/dt``.
    """

    n: int
    idx: cp.ndarray
    amgx: AMGXSolver


def prepare_grounded_solver(
    a: csp.csr_matrix, ground_node: int, config: str = AMGX_CONFIG
) -> GroundedSolver:
    """Remove the ground DOF and build the AMGx hierarchy."""
    n = a.shape[0]
    idx = cp.arange(n - 1, dtype=cp.int32)
    idx[ground_node:] += 1
    a_red = a[idx][:, idx].tocsr()
    a_red.sum_duplicates()
    amgx = AMGXSolver(config)
    amgx.setup(
        a_red.indptr.astype(cp.int32),
        a_red.indices.astype(cp.int32),
        a_red.data.astype(cp.float64),
    )
    return GroundedSolver(n, idx, amgx)


def solve_grounded(solver: GroundedSolver, b: cp.ndarray) -> cp.ndarray:
    """Solve one RHS on the prepared hierarchy."""
    b_red = cp.ascontiguousarray(b[solver.idx], dtype=cp.float64)
    x_red = cp.empty(int(solver.idx.shape[0]), dtype=cp.float64)
    solver.amgx.solve(b_red, x_red)
    v = cp.zeros(solver.n, dtype=cp.float64)
    v[solver.idx] = x_red
    return v


def _dadt_node_to_elm(dadt_nodes: cp.ndarray, tet_nodes: cp.ndarray) -> cp.ndarray:
    """Average nodal dA/dt over each tetrahedron."""
    dadt_elm = dadt_nodes[tet_nodes[:, 0]]
    for vertex in range(1, 4):
        dadt_elm += dadt_nodes[tet_nodes[:, vertex]]
    dadt_elm *= 0.25
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
    ground_node = int(cp.argmin(nodes_mm[:, 2]))
    solver = prepare_grounded_solver(stiffness, ground_node)
    del stiffness

    g = cp.ascontiguousarray(g.astype(RESIDENT_G_DTYPE))
    # Negating after multiplication preserves the previous IEEE rounding order.
    neg_vc = cp.ascontiguousarray(-(vols.astype(cp.float32) * cond.astype(cp.float32)))
    vols = cp.ascontiguousarray(vols.astype(cp.float32))
    del cond
    ptr, idx = build_node2corner(tet_nodes, mesh.n_nodes)

    skin_tris = cp.asarray(mesh.skin_tris)
    skin_a = cp.ascontiguousarray(nodes_mm[skin_tris[:, 0]])
    skin_b = cp.ascontiguousarray(nodes_mm[skin_tris[:, 1]])
    skin_c = cp.ascontiguousarray(nodes_mm[skin_tris[:, 2]])
    skin_tri_normals = cp.asarray(mesh.skin_triangle_normals)
    return SolverContext(
        mesh,
        nodes_mm,
        tet_nodes,
        tet_tags,
        mesh.n_nodes,
        g,
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

    b = _assemble_rhs_kernel(
        dadt_elm,
        ctx.g,
        ctx.neg_vc,
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
