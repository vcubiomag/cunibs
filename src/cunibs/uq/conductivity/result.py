"""Per-element uncertainty statistics from Monte Carlo conductivity UQ."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cupy as cp
import h5py
import numpy as np
import numpy.typing as npt

from cunibs import metrics
from cunibs.simulation import (
    _DEFAULT_REGION,
    Placement,
    _check_format_version,
    _opt_array,
    _read_metrics,
    _write_metrics,
)

if TYPE_CHECKING:
    from cunibs.metrics import ArrayT

_FORMAT_VERSION = 1


def _tissue_sensitivity(result, output: str) -> dict[int, float]:
    """First-order variance share of each perturbed tissue from a log-linear regression.

    Regresses ``log`` of the chosen per-draw scalar on the ``log`` conductivity draws. Because the
    ensemble is i.i.d., the inputs are independent, so ``beta_j^2·Var(log σ_j)/Var(log y)`` is tag
    ``j``'s first-order (Sobol) share of the output variance under a local log-linear response. It
    is not a Saltelli Sobol estimate and captures only the linear-in-log part of the sensitivity.
    """
    if output == "peak":
        y = result.peak_samples
    elif output == "focality":
        y = result.focality_samples
    elif result.roi_samples is not None and output in result.roi_samples:
        y = result.roi_samples[output]
    else:
        y = None
    if y is None:
        available = ["peak", "focality", *(result.roi_samples or {})]
        raise ValueError(
            f"No per-draw samples for output {output!r} (available: {available}); "
            "run simulate_conductivity_uq(..., record_rois={...}) with that ROI name."
        )
    x = np.log(np.asarray(result.sigma_samples, dtype=np.float64))
    logy = np.log(np.asarray(y, dtype=np.float64))
    xc = x - x.mean(0)
    yc = logy - logy.mean()
    var_y = float(yc.var())
    if var_y == 0.0:
        return {int(t): 0.0 for t in result.perturbed_tags}
    beta, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    contrib = beta**2 * xc.var(0) / var_y
    return {int(t): float(c) for t, c in zip(result.perturbed_tags, contrib, strict=False)}


@dataclass
class ConductivityUQSummary:
    """Metrics of the *mean* field, plus the peak local CoV, for one placement.

    For a nonlinear metric (peak, focality) the metric of the mean is **not** the mean of
    the metric over the ensemble: ``peak_cov`` is the CoV of the mean field, not the CoV of
    the per-draw peak. To characterise the distribution of a scalar metric, run with
    ``record_rois=...`` and use the per-draw arrays on the result.
    """

    mean_field: metrics.FieldMetrics
    peak_cov: float


@dataclass
class ConductivityUQResult:
    """Conductivity-UQ metrics for one placement, plus the moment arrays if retained.

    ``mean_magnE``, ``std_magnE`` and ``cov_magnE`` are the per-tetrahedron moments of
    ``|E|`` over the ensemble, ``cov`` being the coefficient of variation (std / mean).
    They are ``None`` unless the run asked for them with ``moments=True``.

    Results from :class:`~cunibs.Subject` are NumPy and always carry a ``summary``. The
    lower-level :func:`~cunibs.uq.run_conductivity_uq` returns device arrays and no
    summary; call :meth:`compute_summary` and :meth:`to_numpy` yourself.
    """

    mean_magnE: ArrayT | None
    std_magnE: ArrayT | None
    cov_magnE: ArrayT | None
    n_samples: int
    perturbed_tags: tuple[int, ...]
    sigma_samples: npt.NDArray[np.float64]  # (n_samples, n_perturbed), host
    vols: ArrayT | None
    tet_tags: ArrayT | None
    barycenters_mm: ArrayT | None
    placement: Placement
    coil_name: str
    didt: float
    summary: ConductivityUQSummary | None = None
    # per-draw samples, present only when the run was given record_rois=
    roi_samples: dict[str, npt.NDArray[np.float64]] | None = (
        None  # name -> (n_samples,) ROI mean |E|
    )
    peak_samples: npt.NDArray[np.float64] | None = None  # (n_samples,) gray-matter peak |E|
    focality_samples: npt.NDArray[np.float64] | None = (
        None  # (n_samples,) stimulated volume (m^3)
    )
    peak_location_samples: npt.NDArray[np.float64] | None = (
        None  # (n_samples, 3) peak location (mm)
    )

    def compute_summary(
        self, region: metrics.Region = _DEFAULT_REGION
    ) -> ConductivityUQSummary:
        """Summarise the moment arrays over ``region``, which needs ``moments=True``.

        Results from :class:`~cunibs.Subject` already carry a gray-matter ``summary``. One
        straight from :func:`~cunibs.uq.run_conductivity_uq` does not, so call this before
        :meth:`save`.
        """
        if self.mean_magnE is None or self.cov_magnE is None:
            raise ValueError(
                f"Metrics for region {region!r} need the per-element moment arrays, which "
                "this result did not retain. Re-run with moments=True."
            )
        assert self.vols is not None and self.tet_tags is not None
        assert self.barycenters_mm is not None
        return ConductivityUQSummary(
            mean_field=metrics.compute_metrics(
                self.mean_magnE, self.vols, self.barycenters_mm, self.tet_tags, region=region
            ),
            peak_cov=metrics.peak_magnitude(
                self.cov_magnE, metrics.region_mask(self.tet_tags, region)
            ),
        )

    def _summary_for(self, region: metrics.Region) -> ConductivityUQSummary:
        if self.summary is not None and region == self.summary.mean_field["region"]:
            return self.summary
        return self.compute_summary(region)

    def peak_mean_magnE(self, region: metrics.Region = _DEFAULT_REGION) -> float:
        """Peak of the mean field in a region."""
        return self._summary_for(region).mean_field["peak_magnE"]

    def peak_cov(self, region: metrics.Region = _DEFAULT_REGION) -> float:
        """Largest local coefficient of variation in a region."""
        return self._summary_for(region).peak_cov

    def tissue_sensitivity(self, output: str = "peak") -> dict[int, float]:
        """First-order share of each perturbed tissue in a per-draw output's variance.

        ``output`` is ``"peak"``, ``"focality"``, or a recorded ROI name, so the run needs
        ``record_rois={...}``. Regressing the log output on the log conductivity draws gives
        each tag's first-order share under a local log-linear response; it captures only the
        linear-in-log part of the sensitivity and is not a Saltelli Sobol estimate.
        """
        return _tissue_sensitivity(self, output)

    def to_numpy(self) -> ConductivityUQResult:
        """Copy any device arrays to NumPy."""

        def host(a: ArrayT | None) -> npt.NDArray[Any] | None:
            return None if a is None else cp.asnumpy(a)

        return replace(
            self,
            mean_magnE=host(self.mean_magnE),
            std_magnE=host(self.std_magnE),
            cov_magnE=host(self.cov_magnE),
            vols=host(self.vols),
            tet_tags=host(self.tet_tags),
            barycenters_mm=host(self.barycenters_mm),
        )

    def save(self, path: str | Path) -> None:
        """Write the conductivity-UQ result to a self-contained HDF5 file.

        Moment arrays the run did not retain are absent from the file and load back as
        ``None``.
        """
        if self.summary is None:
            raise ValueError(
                "This result has no summary to save; call compute_summary() first "
                "(results from Subject already carry one)."
            )
        with h5py.File(Path(path), "w") as h5f:
            for name in (
                "mean_magnE",
                "std_magnE",
                "cov_magnE",
                "vols",
                "tet_tags",
                "barycenters_mm",
            ):
                arr = getattr(self, name)
                if arr is None:
                    continue
                h5f.create_dataset(name, data=cp.asnumpy(arr), compression="gzip")
            h5f.create_dataset("sigma_samples", data=np.asarray(self.sigma_samples))
            grp = h5f.create_group("summary")
            _write_metrics(grp.create_group("mean_field"), self.summary.mean_field)
            grp.attrs["peak_cov"] = self.summary.peak_cov
            if self.roi_samples is not None:
                roi = h5f.create_group("roi_samples")
                for roi_name, arr in self.roi_samples.items():
                    roi.create_dataset(roi_name, data=np.asarray(arr))
            for name in ("peak_samples", "focality_samples", "peak_location_samples"):
                value = getattr(self, name)
                if value is not None:
                    h5f.create_dataset(name, data=np.asarray(value))
            h5f.attrs["format_version"] = _FORMAT_VERSION
            h5f.attrs["n_samples"] = self.n_samples
            h5f.attrs["perturbed_tags"] = np.asarray(self.perturbed_tags, dtype=np.int32)
            h5f.attrs["coil_name"] = self.coil_name
            h5f.attrs["didt"] = self.didt
            h5f.attrs["placement_center_mm"] = self.placement.center_mm
            h5f.attrs["placement_handle_mm"] = self.placement.handle_mm
            h5f.attrs["placement_distance_mm"] = self.placement.distance_mm

    @classmethod
    def load(cls, path: str | Path) -> ConductivityUQResult:
        """Read a saved conductivity-UQ result into NumPy arrays."""
        with h5py.File(Path(path), "r") as h5f:
            _check_format_version(h5f, path, _FORMAT_VERSION)
            return cls(
                mean_magnE=_opt_array(h5f, "mean_magnE"),
                std_magnE=_opt_array(h5f, "std_magnE"),
                cov_magnE=_opt_array(h5f, "cov_magnE"),
                n_samples=int(h5f.attrs["n_samples"]),
                perturbed_tags=tuple(int(t) for t in h5f.attrs["perturbed_tags"]),
                sigma_samples=np.asarray(h5f["sigma_samples"]),
                vols=_opt_array(h5f, "vols"),
                tet_tags=_opt_array(h5f, "tet_tags"),
                barycenters_mm=_opt_array(h5f, "barycenters_mm"),
                placement=Placement(
                    center_mm=h5f.attrs["placement_center_mm"],
                    handle_mm=h5f.attrs["placement_handle_mm"],
                    distance_mm=float(h5f.attrs["placement_distance_mm"]),
                ),
                coil_name=str(h5f.attrs["coil_name"]),
                didt=float(h5f.attrs["didt"]),
                summary=ConductivityUQSummary(
                    mean_field=_read_metrics(h5f["summary"]["mean_field"]),
                    peak_cov=float(h5f["summary"].attrs["peak_cov"]),
                ),
                roi_samples=(
                    {k: np.asarray(v) for k, v in h5f["roi_samples"].items()}
                    if "roi_samples" in h5f
                    else None
                ),
                peak_samples=_opt_array(h5f, "peak_samples"),
                focality_samples=_opt_array(h5f, "focality_samples"),
                peak_location_samples=_opt_array(h5f, "peak_location_samples"),
            )
