"""Compute volume-weighted E-field metrics for tetrahedral meshes."""

from __future__ import annotations

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


def region_mask(tet_tags: ArrayT, region: Region) -> ArrayT:
    """Boolean per-tet mask for ``region`` (``"all"`` or any volume tissue label)."""
    xp = cp.get_array_module(tet_tags)
    if region == "all":
        return xp.ones(tet_tags.shape[0], dtype=bool)
    tag = _LABEL_TO_TAG.get(region)
    if tag is None:
        raise ValueError(
            f"Unknown region {region!r}; use 'all' or a tissue label: {sorted(_LABEL_TO_TAG)}."
        )
    return tet_tags == tag


def _prefix_sum(w: ArrayT, tile: int = 1024) -> ArrayT:
    """Inclusive prefix sum in float64, with an association order fixed by size.

    cupy's ``cumsum`` is a decoupled-lookback scan, so which partial sums combine with which
    follows block scheduling rather than the data, and one array can yield several distinct
    totals across calls. Splitting into fixed-size tiles pins the order: each tile is scanned
    independently, and the tile offsets are scanned the same way, recursing until one tile
    covers the array. Accumulating in float64 also bounds the drift over a million adds.
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


def percentile_magnitude(magnE: ArrayT, vols: ArrayT, mask: ArrayT, percentile: float) -> float:
    """Volume-weighted percentile of |E| in the region."""
    xp = cp.get_array_module(magnE)
    m = magnE[mask]
    q = xp.asarray([percentile / 100.0], dtype=m.dtype)
    return float(cp.asnumpy(weighted_quantiles(m, vols[mask], q))[0])


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
    w = vols[mask] * magnE[mask]
    cog = (w[:, None] * barycenters_mm[mask]).sum(0) / w.sum()
    return cp.asnumpy(cog)


def distribution(
    magnE: ArrayT,
    vols: ArrayT,
    mask: ArrayT,
    percentiles: tuple[float, ...] = _SUMMARY_PERCENTILES,
) -> dict[str, float]:
    """Volume-weighted mean/std and percentiles of |E| in the region."""
    xp = cp.get_array_module(magnE)
    m = magnE[mask]
    w = vols[mask]
    wsum = w.sum()
    mean = float((w * m).sum() / wsum)
    var = float((w * (m - mean) ** 2).sum() / wsum)
    qs = xp.asarray([p / 100.0 for p in percentiles], dtype=m.dtype)
    pvals = cp.asnumpy(weighted_quantiles(m, w, qs))
    out = {"mean": mean, "std": float(np.sqrt(var))}
    out.update({f"p{p:g}": float(val) for p, val in zip(percentiles, pvals, strict=False)})
    return out


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

    ``peak_magnE`` is the true maximum; the focality volumes are measured against
    :data:`FOCALITY_ANCHOR_PERCENTILE` of |E| instead (see :func:`focality`).
    """
    mask = region_mask(tet_tags, region)
    peak = peak_magnitude(magnE, mask)
    dist = distribution(magnE, vols, mask, percentiles)
    # Reuse the percentile the distribution already sorted for, and share the one anchor
    # across every fraction.
    anchor_key = f"p{FOCALITY_ANCHOR_PERCENTILE:g}"
    anchor = (
        dist[anchor_key]
        if anchor_key in dist
        else percentile_magnitude(magnE, vols, mask, FOCALITY_ANCHOR_PERCENTILE)
    )
    return {
        "region": region,
        "peak_magnE": peak,
        "peak_location_mm": peak_location_mm(magnE, barycenters_mm, mask),
        "center_of_gravity_mm": center_of_gravity_mm(magnE, vols, barycenters_mm, mask),
        "region_volume_m3": float(vols[mask].sum()),
        "focality_m3": {
            f"{frac:g}": stimulated_volume(magnE, vols, mask, frac * anchor)
            for frac in focality_fracs
        },
        "distribution": dist,
    }
