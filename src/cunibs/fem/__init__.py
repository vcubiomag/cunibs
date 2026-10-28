"""GPU P1 FEM backend for the TMS E-field (cupy + custom CUDA kernels)."""

from cunibs.fem.assembly import TISSUE_CONDUCTIVITY
from cunibs.fem.placement import (
    coil_dadt_at_nodes,
    coil_dadt_block,
    compute_coil_transform,
    compute_coil_transforms,
)
from cunibs.fem.recovery import (
    RECOVERY_MODES,
    TISSUE_SLOT_MODES,
    Recovery,
    RecoveryOperator,
    apply_recovery,
    ensure_recovery,
    outer_boundary_nodes,
    tissue_boundary_nodes,
    validate_recovery,
)
from cunibs.fem.solve import (
    MAX_BLOCK,
    PlacementResult,
    SolverContext,
    SolverConvergenceError,
    build_context,
    solve_grounded,
    solve_placement,
    solve_placements_block,
)
from cunibs.fem.sweep import ensure_sweep_basis

__all__ = [
    "MAX_BLOCK",
    "RECOVERY_MODES",
    "TISSUE_CONDUCTIVITY",
    "TISSUE_SLOT_MODES",
    "PlacementResult",
    "Recovery",
    "RecoveryOperator",
    "SolverContext",
    "SolverConvergenceError",
    "apply_recovery",
    "build_context",
    "coil_dadt_at_nodes",
    "coil_dadt_block",
    "compute_coil_transform",
    "compute_coil_transforms",
    "ensure_recovery",
    "ensure_sweep_basis",
    "outer_boundary_nodes",
    "solve_grounded",
    "solve_placement",
    "solve_placements_block",
    "tissue_boundary_nodes",
    "validate_recovery",
]
