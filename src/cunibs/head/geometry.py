"""Derived scalp geometry: how far the cortex is, and how thick the layers between are.

Distances are measured against the tissue surfaces CHARM writes into the mesh (tags 1001-1010),
so a depth is an intersection with the same geometry the solver discretises. The scalp point and
the inward normal come from the placement kernel, so a distance is measured from exactly the point
a coil would be placed over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from cunibs.fem.placement import compute_coil_transforms
from cunibs.metrics import region_tag
from cunibs.solver import ray_first_hit

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from cunibs.fem.solve import SolverContext
    from cunibs.metrics import Region

# A surface tag is its volume tag plus a thousand; see cunibs.mesh.SURFACE_KEY_TO_LABEL.
_SURFACE_TAG_OFFSET = 1000

# Hits closer than this are the surface the ray started on, not the one it is looking for.
_SELF_HIT_EPS_MM = 1e-6


def _depth_to_tag(
    ctx: SolverContext, origins: cp.ndarray, inward: cp.ndarray, tag: int
) -> npt.NDArray[np.float64]:
    """Distance from each origin along its inward normal to surface ``tag``, ``NaN`` on a miss."""
    verts = cp.ascontiguousarray(ctx.nodes_mm[cp.asarray(ctx.mesh.surface(tag))])
    t = cp.empty(origins.shape[0], dtype=cp.float64)
    ray_first_hit(
        origins,
        inward,
        verts,
        t,
        _SELF_HIT_EPS_MM,
        cp.cuda.get_current_stream().ptr,
    )
    return cp.asnumpy(cp.where(cp.isfinite(t), t, cp.nan))


def project_to_scalp(
    ctx: SolverContext, points_mm: npt.ArrayLike
) -> tuple[cp.ndarray, cp.ndarray]:
    """Scalp projection and inward unit normal for each point, as a placement would derive them.

    Goes through the placement kernel rather than repeating its closest-point rule, so a distance
    is measured from the point a coil would actually sit over.
    """
    pts = cp.ascontiguousarray(cp.asarray(points_mm, dtype=cp.float64).reshape(-1, 3))
    transforms = compute_coil_transforms(ctx, pts, None, cp.zeros(pts.shape[0]))
    # Column 2 is the coil's z axis, which the kernel sets to the inward scalp normal; the
    # translation is the projection itself, distance being zero.
    origins = cp.ascontiguousarray(transforms[:, :3, 3])
    inward = cp.ascontiguousarray(transforms[:, :3, 2])
    return origins, inward


def depth_to_surface_mm(
    ctx: SolverContext, points_mm: npt.ArrayLike, region: Region = "gray_matter"
) -> npt.NDArray[np.float64]:
    """Distance in mm from each point's scalp projection, inward, to ``region``'s surface.

    With the default this is the scalp-to-cortex distance. ``NaN`` where the inward normal leaves
    the head without crossing that surface, which says "not measurable here" rather than reading
    as a deep target.
    """
    origins, inward = project_to_scalp(ctx, points_mm)
    return _depth_to_tag(ctx, origins, inward, region_tag(region) + _SURFACE_TAG_OFFSET)


def scalp_to_cortex_mm(ctx: SolverContext, points_mm: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Scalp-to-cortex distance in mm: :func:`depth_to_surface_mm` at the gray-matter surface."""
    return depth_to_surface_mm(ctx, points_mm, "gray_matter")


def tissue_depths_mm(
    ctx: SolverContext, points_mm: npt.ArrayLike, regions: Sequence[Region]
) -> dict[str, npt.NDArray[np.float64]]:
    """Depth to each region's surface, from one scalp projection per point.

    Sharing the projection is what makes the depths comparable: differencing two of them gives the
    thickness of what lies between along the same ray, which is not true of depths measured from
    separately projected points.
    """
    origins, inward = project_to_scalp(ctx, points_mm)
    return {
        str(region): _depth_to_tag(
            ctx, origins, inward, region_tag(region) + _SURFACE_TAG_OFFSET
        )
        for region in regions
    }


def tissue_thickness_mm(
    ctx: SolverContext, points_mm: npt.ArrayLike, outer: Region, inner: Region
) -> npt.NDArray[np.float64]:
    """Thickness in mm between two tissue surfaces along the inward normal.

    ``outer`` is the surface entered first. Skull thickness is ``outer="cortical_bone"`` and
    ``inner="csf"``; the value is ``NaN`` where either surface is not crossed.
    """
    depths = tissue_depths_mm(ctx, points_mm, [outer, inner])
    return depths[str(inner)] - depths[str(outer)]
