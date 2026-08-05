"""Compute volume-weighted E-field metrics for tetrahedral meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.mesh import VOLUME_KEY_TO_LABEL, TissueLabel

type ArrayT = cp.ndarray | np.ndarray
type Region = TissueLabel | Literal["all"]

_LABEL_TO_TAG: dict[str, int] = {label: tag for tag, label in VOLUME_KEY_TO_LABEL.items()}

_SUMMARY_PERCENTILES = (50.0, 95.0, 99.0, 99.9)
_SUMMARY_FOCALITY_FRACS = (0.5, 0.75)

# Focality is measured against a high percentile rather than the maximum: on a tetrahedral
# mesh the max |E| is routinely set by one sliver element at a tissue boundary, too noisy to
# anchor a threshold to.
FOCALITY_ANCHOR_PERCENTILE = 99.9


class FieldMetrics(TypedDict):
    """TMS metrics for one tissue region."""

    region: Region
    peak_magnE: float
    peak_location_mm: npt.NDArray[np.float64]
    center_of_gravity_mm: npt.NDArray[np.float64]
    region_volume_m3: float
    focality_m3: dict[str, float]
    distribution: dict[str, float]


def region_tag(region: Region) -> int:
    """The volume tag a tissue label names. ``"all"`` names no single tag, so it raises."""
    tag = _LABEL_TO_TAG.get(region)
    if tag is None:
        raise ValueError(
            f"Unknown region {region!r}; use a tissue label: {sorted(_LABEL_TO_TAG)}."
        )
    return tag


def region_mask(tet_tags: ArrayT, region: Region) -> ArrayT:
    """Boolean per-tet mask for ``region`` (``"all"`` or any volume tissue label)."""
    if region == "all":
        xp = cp.get_array_module(tet_tags)
        return xp.ones(tet_tags.shape[0], dtype=bool)
    return tet_tags == region_tag(region)


def _prefix_sum(w: ArrayT, tile: int = 1024) -> ArrayT:
    """Inclusive prefix sum in float64, with an association order fixed by size.

    cupy's ``cumsum`` is a decoupled-lookback scan, so which partial sums combine with which
    follows block scheduling rather than the data, and one array can yield several distinct totals
    across calls. Splitting into fixed-size tiles pins the order: each tile is scanned
    independently, and the tile offsets are scanned the same way, recursing until one tile covers
    the array. Accumulating in float64 bounds the drift over a million adds.
    """
    xp = cp.get_array_module(w)
    a = w.astype(xp.float64, copy=False)
    n = int(a.size)
    # numpy's cumsum is sequential, so it is already both fixed-order and float64 here.
    if xp is np or n <= tile:
        return xp.cumsum(a)
    pad = (-n) % tile
    if pad:
        a = xp.concatenate([a, xp.zeros(pad, a.dtype)])
    within = xp.cumsum(a.reshape(-1, tile), axis=1)
    totals = within[:, -1]
    offsets = _prefix_sum(totals, tile) - totals
    return (within + offsets[:, None]).ravel()[:n]


def weighted_quantiles(values: ArrayT, weights: ArrayT, qs: ArrayT) -> ArrayT:
    """Volume-weighted quantiles of ``values`` (``qs`` in [0, 1])."""
    xp = cp.get_array_module(values)
    # Stable so that ties in |E|, of which a million tetrahedra have tens of thousands, always
    # send the same weight to the same slot of the prefix sum.
    order = xp.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = _prefix_sum(w)
    # Midpoint positions prevent a single element from spanning its full weight interval.
    pos = (cw - 0.5 * w) / cw[-1]
    # Evaluate at pos's precision: in float32 a tail quantile like 0.999 lands ~1e-8 off, and
    # where the tail is steep that moves a focality threshold enough to flip elements across it.
    return xp.interp(xp.asarray(qs, dtype=pos.dtype), pos, v)


def peak_magnitude(magnE: ArrayT, mask: ArrayT) -> float:
    """Largest |E| in the region.

    This is the true maximum. For a value less sensitive to a single outlier element, use
    :func:`percentile_magnitude`.
    """
    return float(magnE[mask].max())


# The helpers below take a region's values and weights already gathered. Both the mask-taking
# functions and RegionSlice.summarize route through them, so each metric is defined once.


def _percentile(m: ArrayT, w: ArrayT, percentile: float) -> float:
    xp = cp.get_array_module(m)
    q = xp.asarray([percentile / 100.0], dtype=m.dtype)
    return float(cp.asnumpy(weighted_quantiles(m, w, q))[0])


def _stimulated(m: ArrayT, w: ArrayT, threshold: float) -> float:
    return float(w[m >= threshold].sum())


def _center_of_gravity_mm(m: ArrayT, w: ArrayT, bary: ArrayT) -> npt.NDArray[np.float64]:
    ww = w * m
    return cp.asnumpy((ww[:, None] * bary).sum(0) / ww.sum())


def _distribution(m: ArrayT, w: ArrayT, percentiles: tuple[float, ...]) -> dict[str, float]:
    xp = cp.get_array_module(m)
    wsum = w.sum()
    mean = float((w * m).sum() / wsum)
    var = float((w * (m - mean) ** 2).sum() / wsum)
    qs = xp.asarray([p / 100.0 for p in percentiles], dtype=m.dtype)
    pvals = cp.asnumpy(weighted_quantiles(m, w, qs))
    out = {"mean": mean, "std": float(np.sqrt(var))}
    out.update({f"p{p:g}": float(val) for p, val in zip(percentiles, pvals, strict=False)})
    return out


def percentile_magnitude(magnE: ArrayT, vols: ArrayT, mask: ArrayT, percentile: float) -> float:
    """Volume-weighted percentile of |E| in the region."""
    return _percentile(magnE[mask], vols[mask], percentile)


def peak_location_mm(
    magnE: ArrayT, barycenters_mm: ArrayT, mask: ArrayT
) -> npt.NDArray[np.float64]:
    """Barycentre (mm) of the tetrahedron carrying the peak |E| in the region."""
    xp = cp.get_array_module(magnE)
    idx = xp.where(mask)[0]
    peak = idx[xp.argmax(magnE[mask])]
    return cp.asnumpy(barycenters_mm[peak])


def stimulated_volume(magnE: ArrayT, vols: ArrayT, mask: ArrayT, threshold: float) -> float:
    """Total tissue volume (m³) with |E| ≥ ``threshold`` in the region."""
    hit = mask & (magnE >= threshold)
    return float(vols[hit].sum())


def focality(
    magnE: ArrayT,
    vols: ArrayT,
    mask: ArrayT,
    frac: float = 0.5,
    anchor_percentile: float = FOCALITY_ANCHOR_PERCENTILE,
) -> float:
    """Volume with ``|E| >= frac * P``, ``P`` being the volume-weighted ``anchor_percentile``.

    Pass ``anchor_percentile=100`` to measure against the true maximum instead.
    """
    anchor = percentile_magnitude(magnE, vols, mask, anchor_percentile)
    return stimulated_volume(magnE, vols, mask, frac * anchor)


def center_of_gravity_mm(
    magnE: ArrayT, vols: ArrayT, barycenters_mm: ArrayT, mask: ArrayT
) -> npt.NDArray[np.float64]:
    """Volume·|E|-weighted centroid (mm) of the field in the region."""
    return _center_of_gravity_mm(magnE[mask], vols[mask], barycenters_mm[mask])


def distribution(
    magnE: ArrayT,
    vols: ArrayT,
    mask: ArrayT,
    percentiles: tuple[float, ...] = _SUMMARY_PERCENTILES,
) -> dict[str, float]:
    """Volume-weighted mean/std and percentiles of |E| in the region."""
    return _distribution(magnE[mask], vols[mask], percentiles)


@dataclass(frozen=True)
class RegionSlice:
    """The field-independent half of a region summary, gathered once.

    Holds the compacted element index, volumes and barycentres for one region, so a sweep pays
    the mask and the gathers once rather than once per metric per placement. On a 3.9M-tetrahedron
    mesh the gray matter slice is 1.16M elements.

    Build with :func:`region_slice`, outside any scratch allocator whose lifetime is shorter
    than the sweep.
    """

    region: Region
    index: ArrayT
    vols: ArrayT
    barycenters_mm: ArrayT
    volume_m3: float

    def summarize(
        self,
        magnE: ArrayT,
        *,
        focality_fracs: tuple[float, ...] = _SUMMARY_FOCALITY_FRACS,
        percentiles: tuple[float, ...] = _SUMMARY_PERCENTILES,
    ) -> FieldMetrics:
        """All E-field metrics for the region, reading one gathered copy of ``magnE``.

        ``peak_magnE`` is the true maximum; the focality volumes are measured against
        :data:`FOCALITY_ANCHOR_PERCENTILE` of |E| instead (see :func:`focality`).
        """
        xp = cp.get_array_module(magnE)
        m = magnE[self.index]
        w = self.vols
        dist = _distribution(m, w, percentiles)
        # Reuse the percentile the distribution already sorted for, and share the one anchor
        # across every fraction.
        anchor_key = f"p{FOCALITY_ANCHOR_PERCENTILE:g}"
        anchor = (
            dist[anchor_key]
            if anchor_key in dist
            else _percentile(m, w, FOCALITY_ANCHOR_PERCENTILE)
        )
        return {
            "region": self.region,
            "peak_magnE": float(m.max()),
            "peak_location_mm": cp.asnumpy(self.barycenters_mm[xp.argmax(m)]),
            "center_of_gravity_mm": _center_of_gravity_mm(m, w, self.barycenters_mm),
            "region_volume_m3": self.volume_m3,
            "focality_m3": {
                f"{frac:g}": _stimulated(m, w, frac * anchor) for frac in focality_fracs
            },
            "distribution": dist,
        }


def region_slice(
    vols: ArrayT, barycenters_mm: ArrayT, tet_tags: ArrayT, region: Region = "gray_matter"
) -> RegionSlice:
    """Gather the field-independent arrays for ``region`` once."""
    xp = cp.get_array_module(vols)
    index = xp.where(region_mask(tet_tags, region))[0]
    v = vols[index]
    return RegionSlice(
        region=region,
        index=index,
        vols=v,
        barycenters_mm=barycenters_mm[index],
        volume_m3=float(v.sum()),
    )


def compute_metrics(
    magnE: ArrayT,
    vols: ArrayT,
    barycenters_mm: ArrayT,
    tet_tags: ArrayT,
    *,
    region: Region = "gray_matter",
    focality_fracs: tuple[float, ...] = _SUMMARY_FOCALITY_FRACS,
    percentiles: tuple[float, ...] = _SUMMARY_PERCENTILES,
) -> FieldMetrics:
    """Compute all E-field metrics for one tissue region.

    Builds a :class:`RegionSlice` per call. To summarise many fields over the same region, build
    one with :func:`region_slice` and call :meth:`RegionSlice.summarize` instead.
    """
    return region_slice(vols, barycenters_mm, tet_tags, region).summarize(
        magnE, focality_fracs=focality_fracs, percentiles=percentiles
    )
