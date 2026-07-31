# Must precede _solver_ext: loads the CUDA libraries the extension links against.
from cunibs.solver import _cuda_preload

try:
    from cunibs.solver._solver_ext import (
        BLOCK_SIZES,
        MAX_STAGE_BLOCK,
        NativeVCycle,
        PcgAmgSolver,
        accumulate_moments,
        assemble_stiffness_values,
        dadt_nbody,
        dadt_node_to_element,
        element_weight,
        node_scatter3,
        place_transforms,
        reconstruct_e,
        reconstruct_e_block,
        rhs_assemble,
        rhs_assemble_weighted,
        rhs_assemble_weighted_block,
        select_size4,
        weighted_gradient,
    )
except ImportError as exc:
    raise ImportError(_cuda_preload.describe_failure(exc)) from exc

__all__ = [
    "BLOCK_SIZES",
    "MAX_STAGE_BLOCK",
    "NativeVCycle",
    "PcgAmgSolver",
    "accumulate_moments",
    "assemble_stiffness_values",
    "dadt_nbody",
    "dadt_node_to_element",
    "element_weight",
    "node_scatter3",
    "place_transforms",
    "reconstruct_e",
    "reconstruct_e_block",
    "rhs_assemble",
    "rhs_assemble_weighted",
    "rhs_assemble_weighted_block",
    "select_size4",
    "weighted_gradient",
]
