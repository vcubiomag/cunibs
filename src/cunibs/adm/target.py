"""Specify a cortical target for reciprocity-based E-field evaluation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.metrics import Region, region_mask

if TYPE_CHECKING:
    from cunibs.fem.solve import SolverContext

# Offset past which resolve_target warns that the point lies outside its region. A scalp
# coordinate lands 10-20 mm from gray matter.
FAR_FROM_REGION_MM = 5.0


def _as_point(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {np.shape(value)}.")
    return p


@dataclass(frozen=True, init=False)
class Target:
    """Define the cortical target of a reciprocity solve.

    ``position_mm`` is the target location. ``direction`` fixes the E-field component of interest;
    ``None`` requests the direction-free magnitude (three adjoint solves). ``region`` restricts the
    target to a tissue type. ``radius_mm`` averages the functional over all region elements whose
    barycentre lies within the ball; ``0`` selects the single nearest element.

    The functional is linear in the field: the volume-weighted ROI mean of the raw per-element
    ``E``, then its component along ``direction``, or its norm for a magnitude target.
    """

    position_mm: npt.NDArray[np.float64]
    direction: npt.NDArray[np.float64] | None
    region: Region
    radius_mm: float

    def __init__(
        self,
        position_mm: npt.ArrayLike,
        direction: npt.ArrayLike | None = None,
        *,
        region: Region = "gray_matter",
        radius_mm: float = 0.0,
    ) -> None:
        object.__setattr__(self, "position_mm", _as_point(position_mm))
        if direction is None:
            object.__setattr__(self, "direction", None)
        else:
            d = _as_point(direction)
            n = float(np.linalg.norm(d))
            if n == 0.0:
                raise ValueError("direction must be non-zero.")
            object.__setattr__(self, "direction", d / n)
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "radius_mm", float(radius_mm))


@dataclass
class ResolvedTarget:
    """Target reduced to weighted mesh elements and the adjoint directions."""

    elem_idx: cp.ndarray  # (K,) int32 element indices in the ROI
    weights: cp.ndarray  # (K,) float64, sum to 1 (volume-weighted ROI average)
    directions: cp.ndarray  # (D, 3) float64 orthonormal adjoint directions
    magnitude: bool  # True when directions are the canonical basis for |E|
    barycenter_mm: cp.ndarray  # (3,) float64 ROI centroid
    # How far the requested position lies outside the region, in mm, measured to the nearest
    # element's bounding sphere.
    offset_mm: float = 0.0


def resolve_target(ctx: SolverContext, target: Target | ResolvedTarget) -> ResolvedTarget:
    """Map a :class:`Target` to ROI elements, weights, and adjoint directions.

    A :class:`ResolvedTarget` is returned as it is, so anything that takes a target takes either.
    When no region element lies within ``radius_mm`` the ROI falls back to the nearest one, and
    ``offset_mm`` on the result says how far away that was. Past :data:`FAR_FROM_REGION_MM` it
    also warns, because the point was then most likely given in the wrong frame or region.
    """
    if isinstance(target, ResolvedTarget):
        return target
    barys = ctx.tet_barycenters_mm
    mask = region_mask(ctx.tet_tags, target.region)
    region_ids = cp.where(mask)[0].astype(cp.int32)
    if region_ids.shape[0] == 0:
        raise ValueError(f"Region {target.region!r} contains no elements.")

    pos = cp.asarray(target.position_mm)
    region_barys = barys[region_ids]
    d2 = ((region_barys - pos) ** 2).sum(1)

    if target.radius_mm <= 0.0:
        sel = region_ids[cp.argmin(d2)][None]
    else:
        within = d2 <= target.radius_mm**2
        if not bool(within.any()):
            within = d2 == d2.min()
        sel = region_ids[within]
    corners = ctx.nodes_mm[ctx.tet_nodes[region_ids]]
    reach = cp.sqrt(((corners - region_barys[:, None, :]) ** 2).sum(-1).max(axis=1))
    offset_mm = float(cp.maximum(cp.sqrt(d2) - reach, 0.0).min())
    del corners
    if offset_mm > FAR_FROM_REGION_MM:
        warnings.warn(
            f"The target at {np.array2string(target.position_mm, precision=1)} mm lies "
            f"{offset_mm:.1f} mm outside {target.region}, so the ROI is the nearest "
            f"{target.region} instead. Check that the point is in mesh coordinates and that the "
            "region is the tissue it lies in.",
            stacklevel=2,
        )

    vols = ctx.vols[sel].astype(cp.float64)
    weights = vols / vols.sum()
    centroid = (weights[:, None] * barys[sel]).sum(0)

    if target.direction is None:
        directions = cp.eye(3, dtype=cp.float64)
        magnitude = True
    else:
        directions = cp.asarray(target.direction, dtype=cp.float64)[None]
        magnitude = False

    return ResolvedTarget(
        elem_idx=cp.ascontiguousarray(sel),
        weights=cp.ascontiguousarray(weights),
        directions=cp.ascontiguousarray(directions),
        magnitude=magnitude,
        barycenter_mm=centroid,
        offset_mm=offset_mm,
    )
