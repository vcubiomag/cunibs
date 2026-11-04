"""Compute volume-weighted E-field metrics for tetrahedral meshes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.mesh import VOLUME_KEY_TO_LABEL, TissueLabel
from cunibs.solver import (
    MAX_STAGE_BLOCK,
    SUMMARY_LAYOUT,
    region_summary,
    region_summary_work_size,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

type ArrayT = cp.ndarray | np.ndarray
type Region = TissueLabel | Literal["all"]

_LABEL_TO_TAG: dict[str, int] = {label: tag for tag, label in VOLUME_KEY_TO_LABEL.items()}

_PEAK, _PEAK_LOCATION, _CENTER_OF_GRAVITY, _MEAN, _STD, _QUANTILES = SUMMARY_LAYOUT

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


# One ROI's terms |E|[e] * w, gathered and widened in one pass; summing them gives the ROI's
# volume-weighted mean.
_roi_terms = cp.ElementwiseKernel(
    "raw T magn, I elem, float64 w",
    "float64 y",
    "y = (double)magn[elem] * w",
    "cunibs_roi_terms",
)


def roi_mean(magnE: cp.ndarray, elem_idx: cp.ndarray, weights: cp.ndarray) -> cp.ndarray:
    """Volume-weighted mean |E| over an ROI's elements, as a device scalar.

    ``weights`` are the ROI's normalised volumes, as :class:`~cunibs.ResolvedTarget` carries them.
    """
    return _roi_terms(magnE, elem_idx, weights).sum()


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


def weighted_quantiles(
    values: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    qs: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """Volume-weighted quantiles of host arrays (``qs`` in [0, 1]).

    For a device field, use :meth:`RegionSlice.summary_table`, which takes the same ``qs``.
    """
    if values.size == 0:
        raise ValueError("Quantiles of a region with no elements are undefined.")
    # Stable so that ties in |E|, of which a million tetrahedra have tens of thousands, always
    # send the same weight to the same slot of the prefix sum.
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    # In float64 throughout: in float32 a tail quantile like 0.999 lands ~1e-8 off, and where the
    # tail is steep that moves a focality threshold enough to flip elements across it.
    cw = np.cumsum(w, dtype=np.float64)
    # Midpoint positions prevent a single element from spanning its full weight interval.
    pos = (cw - 0.5 * w) / cw[-1]
    return np.interp(np.asarray(qs, dtype=np.float64), pos, v)


def _check_percentiles(percentiles: Sequence[float]) -> None:
    for percentile in percentiles:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile!r}.")


def _check_fractions(fractions: Sequence[float]) -> None:
    for frac in fractions:
        if not (math.isfinite(frac) and frac >= 0.0):
            raise ValueError(f"focality fraction must be finite and >= 0, got {frac!r}.")


def _percentile(m: npt.NDArray, w: npt.NDArray, percentile: float) -> float:
    return float(weighted_quantiles(m, w, [percentile / 100.0])[0])


def _stimulated(m: npt.NDArray, w: npt.NDArray, threshold: float) -> float:
    return float(w[m >= threshold].sum())


def _center_of_gravity_mm(
    m: npt.NDArray, w: npt.NDArray, bary: npt.NDArray
) -> npt.NDArray[np.float64]:
    ww = w * m
    return (ww[:, None] * bary).sum(0) / ww.sum()


def _distribution(
    m: npt.NDArray, w: npt.NDArray, percentiles: tuple[float, ...]
) -> dict[str, float]:
    wsum = w.sum()
    mean = float((w * m).sum() / wsum)
    var = float((w * (m.astype(np.float64) - mean) ** 2).sum() / wsum)
    pvals = weighted_quantiles(m, w, [p / 100.0 for p in percentiles])
    out = {"mean": mean, "std": float(np.sqrt(var))}
    out.update({f"p{p:g}": float(val) for p, val in zip(percentiles, pvals, strict=True)})
    return out


def percentile_magnitude(
    magnE: npt.NDArray, vols: npt.NDArray, mask: npt.NDArray, percentile: float
) -> float:
    """Volume-weighted percentile of a host |E| in the region."""
    _check_percentiles((percentile,))
    return _percentile(magnE[mask], vols[mask].astype(np.float64), percentile)


def stimulated_volume(
    magnE: npt.NDArray, vols: npt.NDArray, mask: npt.NDArray, threshold: float
) -> float:
    """Total tissue volume (m³) with a host |E| ≥ ``threshold`` in the region."""
    hit = mask & (magnE >= threshold)
    return float(vols[hit].sum(dtype=np.float64))


def focality(
    magnE: npt.NDArray, vols: npt.NDArray, mask: npt.NDArray, frac: float = 0.5
) -> float:
    """Volume with ``|E| >= frac * P``, ``P`` being :data:`FOCALITY_ANCHOR_PERCENTILE` of a host |E|.

    For any other anchor, pass ``frac`` times that anchor to :func:`stimulated_volume`.
    """
    _check_fractions((frac,))
    anchor = percentile_magnitude(magnE, vols, mask, FOCALITY_ANCHOR_PERCENTILE)
    return stimulated_volume(magnE, vols, mask, frac * anchor)


@dataclass(frozen=True, eq=False)
class RegionSlice:
    """The field-independent half of a region summary, gathered once.

    Holds the compacted element index, volumes and barycentres for one region, so a sweep pays
    the mask and the gathers once rather than once per metric per placement. On a 3.9M-tetrahedron
    mesh the gray matter slice is 1.16M elements.

    Build with :func:`region_slice`, outside any scratch allocator whose lifetime is shorter
    than the sweep. Volumes and barycentres are float64; a device slice's index is int32, the
    layout the native summary reads.
    """

    region: Region
    index: ArrayT
    vols: ArrayT
    barycenters_mm: ArrayT
    volume_m3: float
    n_mesh_elements: int

    def _check_lengths(self, fields: Sequence[ArrayT]) -> None:
        # The device gather does not bounds-check, so a short field would read past its array.
        for field in fields:
            if field.shape[0] != self.n_mesh_elements:
                raise ValueError(
                    "Expected a field with one value per element of the mesh "
                    f"({self.n_mesh_elements}), got shape {field.shape}."
                )

    @property
    def on_device(self) -> bool:
        return isinstance(self.index, cp.ndarray)

    @property
    def empty(self) -> bool:
        """Whether the mesh holds no element of this region."""
        return self.index.size == 0

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
        if self.empty:
            raise ValueError(f"Region {self.region!r} holds no elements in this mesh.")
        if self.on_device:
            (summary,) = self.summarize_block(
                [magnE], focality_fracs=focality_fracs, percentiles=percentiles
            )
            return summary
        _check_percentiles(percentiles)
        _check_fractions(focality_fracs)
        self._check_lengths([magnE])
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
            "peak_location_mm": np.asarray(self.barycenters_mm[np.argmax(m)]),
            "center_of_gravity_mm": _center_of_gravity_mm(m, w, self.barycenters_mm),
            "region_volume_m3": self.volume_m3,
            "focality_m3": {
                f"{frac:g}": _stimulated(m, w, frac * anchor) for frac in focality_fracs
            },
            "distribution": dist,
        }

    def summarize_block(
        self,
        fields: Sequence[cp.ndarray],
        *,
        focality_fracs: tuple[float, ...] = _SUMMARY_FOCALITY_FRACS,
        percentiles: tuple[float, ...] = _SUMMARY_PERCENTILES,
    ) -> list[FieldMetrics]:
        """:meth:`summarize` for several device fields at once, in one copy to the host.

        Each field's metrics are the same whatever else is summarised with it.
        """
        qs, anchor = _probe_quantiles(percentiles)
        out = cp.asnumpy(self.summary_table(fields, qs, anchor, focality_fracs))
        return [
            {
                "region": self.region,
                "peak_magnE": float(row[_PEAK]),
                "peak_location_mm": row[_PEAK_LOCATION : _PEAK_LOCATION + 3].copy(),
                "center_of_gravity_mm": row[_CENTER_OF_GRAVITY : _CENTER_OF_GRAVITY + 3].copy(),
                "region_volume_m3": self.volume_m3,
                "focality_m3": {
                    f"{frac:g}": float(row[_QUANTILES + len(qs) + f])
                    for f, frac in enumerate(focality_fracs)
                },
                "distribution": {
                    "mean": float(row[_MEAN]),
                    "std": float(row[_STD]),
                    **{
                        f"p{p:g}": float(row[_QUANTILES + i]) for i, p in enumerate(percentiles)
                    },
                },
            }
            for row in out
        ]

    def summary_table(
        self,
        fields: Sequence[cp.ndarray],
        qs: Sequence[float],
        anchor: int,
        focality_fracs: Sequence[float],
    ) -> cp.ndarray:
        """The native summary of ``fields`` as a device ``(len(fields), n_out)`` table.

        Columns follow ``cunibs.solver.SUMMARY_LAYOUT``: quantiles ``qs`` (in [0, 1]) start at its
        last offset, then the volumes at or above each fraction of quantile ``qs[anchor]``.
        Nothing is copied to the host, so a caller can keep the table on the device.
        """
        _check_fractions(focality_fracs)
        if not self.on_device:
            raise ValueError("summary_table needs a slice built from device arrays")
        self._check_lengths(fields)
        index, w, bary = self.index, self.vols, self.barycenters_mm
        stream = cp.cuda.get_current_stream().ptr
        rows = []
        for start in range(0, len(fields), MAX_STAGE_BLOCK):
            chunk = [cp.ascontiguousarray(f) for f in fields[start : start + MAX_STAGE_BLOCK]]
            f64 = chunk[0].dtype == cp.float64
            out = cp.empty((len(chunk), _QUANTILES + len(qs) + len(focality_fracs)))
            work = cp.empty(
                region_summary_work_size(int(index.shape[0]), len(chunk), f64), dtype=cp.uint8
            )
            region_summary(
                chunk, index, w, bary, list(qs), anchor, list(focality_fracs), work, out, stream
            )
            rows.append(out)
        return rows[0] if len(rows) == 1 else cp.concatenate(rows)


def _probe_quantiles(percentiles: tuple[float, ...]) -> tuple[list[float], int]:
    """The quantiles a summary probes, and which one anchors focality."""
    _check_percentiles(percentiles)
    qs = [p / 100.0 for p in percentiles]
    if FOCALITY_ANCHOR_PERCENTILE in percentiles:
        return qs, percentiles.index(FOCALITY_ANCHOR_PERCENTILE)
    return [*qs, FOCALITY_ANCHOR_PERCENTILE / 100.0], len(qs)


def region_slice(
    vols: ArrayT, barycenters_mm: ArrayT, tet_tags: ArrayT, region: Region = "gray_matter"
) -> RegionSlice:
    """Gather the field-independent arrays for ``region`` once."""
    xp = cp.get_array_module(vols)
    index = xp.flatnonzero(region_mask(tet_tags, region))
    if xp is cp:
        index = index.astype(cp.int32)
    v = xp.ascontiguousarray(vols[index], dtype=xp.float64)
    return RegionSlice(
        region=region,
        index=index,
        vols=v,
        barycenters_mm=xp.ascontiguousarray(barycenters_mm[index], dtype=xp.float64),
        volume_m3=float(v.sum()),
        n_mesh_elements=tet_tags.shape[0],
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
