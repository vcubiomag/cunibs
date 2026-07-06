"""Specify a cortical target for reciprocity-based E-field evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.metrics import Region, region_mask

if TYPE_CHECKING:
    from cunibs.fem.solve import SolverContext


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


def resolve_target(ctx: "SolverContext", target: Target) -> ResolvedTarget:
    """Map a :class:`Target` to ROI elements, weights, and adjoint directions."""
    barys = cp.asarray(ctx.mesh.tet_barycenters_mm)
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
    )
