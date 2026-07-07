"""GPU P1 FEM backend for the TMS E-field (cupy + AMGx)."""

from cunibs.fem.assembly import (
    GM_TAG,
    GRADIENT_TILE_TETS,
    STIFFNESS_TILE_TETS,
    TISSUE_CONDUCTIVITY,
    assemble_stiffness,
    build_node2corner,
    conductivity_per_tet,
    gradient_operator,
)
from cunibs.fem.placement import (
    MU0_OVER_4PI,
    coil_dadt_at_nodes,
    compute_coil_transform,
    compute_coil_transforms,
)
from cunibs.fem.solve import (
    AMGX_CONFIG,
    UQ_AMGX_CONFIG,
    GroundedSolver,
    PlacementResult,
    SolverContext,
    build_context,
    ground_node_of,
    grounded_index,
    prepare_grounded_solver,
    reduce_matrix,
    solve_grounded,
    solve_placement,
)

__all__ = [
    "GM_TAG",
    "GRADIENT_TILE_TETS",
    "STIFFNESS_TILE_TETS",
    "TISSUE_CONDUCTIVITY",
    "MU0_OVER_4PI",
    "AMGX_CONFIG",
    "UQ_AMGX_CONFIG",
    "assemble_stiffness",
    "build_node2corner",
    "conductivity_per_tet",
    "gradient_operator",
    "coil_dadt_at_nodes",
    "compute_coil_transform",
    "compute_coil_transforms",
    "GroundedSolver",
    "PlacementResult",
    "SolverContext",
    "build_context",
    "ground_node_of",
    "grounded_index",
    "reduce_matrix",
    "prepare_grounded_solver",
    "solve_grounded",
    "solve_placement",
]
