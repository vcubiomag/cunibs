"""Compute volume-weighted E-field metrics for tetrahedral meshes."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.fem.assembly import GM_TAG

ArrayT: TypeAlias = cp.ndarray | np.ndarray

DEFAULT_PERCENTILES = (50.0, 95.0, 99.0, 99.9)
DEFAULT_FOCALITY_FRAC = 0.5


class FieldMetrics(TypedDict):
    """TMS metrics for one tissue region."""

    region: str
    peak_magnE: float
    peak_location_mm: npt.NDArray[np.float64]
    center_of_gravity_mm: npt.NDArray[np.float64]
    region_volume_m3: float
    focality_m3: dict[str, float]
    distribution: dict[str, float]


def region_mask(tet_tags: ArrayT, region: str) -> ArrayT:
    """Boolean per-tet mask for ``region`` (``"gray_matter"`` or ``"all"``)."""
    xp = cp.get_array_module(tet_tags)
    if region == "all":
        return xp.ones(tet_tags.shape[0], dtype=bool)
    if region == "gray_matter":
        return tet_tags == GM_TAG
    raise ValueError(f"Unknown region {region!r}; use 'gray_matter' or 'all'.")


def _weighted_quantiles(values: ArrayT, weights: ArrayT, qs: ArrayT) -> ArrayT:
    """Volume-weighted quantiles of ``values`` (``qs`` in [0, 1])."""
    xp = cp.get_array_module(values)
    order = xp.argsort(values)
    v = values[order]
    w = weights[order]
    cw = xp.cumsum(w)
    # Midpoint positions prevent a single element from spanning its full weight interval.
    pos = (cw - 0.5 * w) / cw[-1]
    return xp.interp(qs, pos, v)


def peak_magnitude(magnE: ArrayT, mask: ArrayT) -> float:
    return float(magnE[mask].max())


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
    magnE: ArrayT, vols: ArrayT, mask: ArrayT, frac: float = DEFAULT_FOCALITY_FRAC
) -> float:
    """Return the volume with ``|E| >= frac * peak(|E|)``."""
    return stimulated_volume(magnE, vols, mask, frac * peak_magnitude(magnE, mask))


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
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Volume-weighted mean/std and percentiles of |E| in the region."""
    xp = cp.get_array_module(magnE)
    m = magnE[mask]
    w = vols[mask]
    wsum = w.sum()
    mean = float((w * m).sum() / wsum)
    var = float((w * (m - mean) ** 2).sum() / wsum)
    qs = xp.asarray([p / 100.0 for p in percentiles], dtype=m.dtype)
    pvals = cp.asnumpy(_weighted_quantiles(m, w, qs))
    out = {"mean": mean, "std": float(np.sqrt(var))}
    out.update({f"p{p:g}": float(val) for p, val in zip(percentiles, pvals)})
    return out


def compute_metrics(
    magnE: ArrayT,
    vols: ArrayT,
    barycenters_mm: ArrayT,
    tet_tags: ArrayT,
    *,
    region: str = "gray_matter",
    focality_fracs: tuple[float, ...] = (DEFAULT_FOCALITY_FRAC,),
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> FieldMetrics:
    """Compute all E-field metrics for one tissue region."""
    mask = region_mask(tet_tags, region)
    peak = peak_magnitude(magnE, mask)
    dist = distribution(magnE, vols, mask, percentiles)
    return {
        "region": region,
        "peak_magnE": peak,
        "peak_location_mm": peak_location_mm(magnE, barycenters_mm, mask),
        "center_of_gravity_mm": center_of_gravity_mm(magnE, vols, barycenters_mm, mask),
        "region_volume_m3": float(vols[mask].sum()),
        "focality_m3": {
            f"{frac:g}": focality(magnE, vols, mask, frac) for frac in focality_fracs
        },
        "distribution": dist,
    }
