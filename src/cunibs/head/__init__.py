"""Derived head geometry and the companion files CHARM writes beside a mesh.

Where the cortex is under a scalp point, what a 10-20 label resolves to, and which candidate
positions a placement search should cover.
"""

from cunibs.head.candidates import ScalpGrid, scalp_grid
from cunibs.head.eeg import (
    DEFAULT_CAP,
    find_cap,
    read_cap,
    read_fiducials,
    read_subject_cap,
)
from cunibs.head.geometry import (
    depth_to_surface_mm,
    project_to_scalp,
    scalp_to_cortex_mm,
    tissue_depths_mm,
    tissue_thickness_mm,
)

__all__ = [
    "DEFAULT_CAP",
    "ScalpGrid",
    "depth_to_surface_mm",
    "find_cap",
    "project_to_scalp",
    "read_cap",
    "read_fiducials",
    "read_subject_cap",
    "scalp_grid",
    "scalp_to_cortex_mm",
    "tissue_depths_mm",
    "tissue_thickness_mm",
]
