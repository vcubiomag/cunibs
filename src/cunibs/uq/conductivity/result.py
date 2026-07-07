"""Per-element uncertainty statistics from Monte Carlo conductivity UQ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import h5py
import numpy as np
import numpy.typing as npt

from cunibs import metrics
from cunibs.simulation import ArrayT, Placement

_FORMAT_VERSION = 1


@dataclass
class ConductivityUQResult:
    """Per-tetrahedron moments of ``|E|`` over the conductivity ensemble.

    ``cov_magnE`` is the coefficient of variation (std / mean), the dimensionless local
    sensitivity of the field to conductivity uncertainty. Arrays are CuPy until :meth:`to_numpy`.
    """

    mean_magnE: ArrayT
    std_magnE: ArrayT
    cov_magnE: ArrayT
    n_samples: int
    perturbed_tags: tuple[int, ...]
    sigma_samples: npt.NDArray[np.float64]  # (n_samples, n_perturbed), host
    vols: ArrayT
    tet_tags: ArrayT
    barycenters_mm: ArrayT
    placement: Placement
    coil_name: str
    didt: float

    def _mask(self, region: metrics.Region) -> ArrayT:
        return metrics.region_mask(self.tet_tags, region)

    def peak_mean_magnE(self, region: metrics.Region = "gray_matter") -> float:
        """Peak of the mean field in a region."""
        return metrics.peak_magnitude(self.mean_magnE, self._mask(region))

    def peak_cov(self, region: metrics.Region = "gray_matter") -> float:
        """Largest local coefficient of variation in a region."""
        return metrics.peak_magnitude(self.cov_magnE, self._mask(region))

    def to_numpy(self) -> "ConductivityUQResult":
        """Copy device arrays to NumPy."""
        return ConductivityUQResult(
            mean_magnE=cp.asnumpy(self.mean_magnE),
            std_magnE=cp.asnumpy(self.std_magnE),
            cov_magnE=cp.asnumpy(self.cov_magnE),
            n_samples=self.n_samples,
            perturbed_tags=self.perturbed_tags,
            sigma_samples=np.asarray(self.sigma_samples),
            vols=cp.asnumpy(self.vols),
            tet_tags=cp.asnumpy(self.tet_tags),
            barycenters_mm=cp.asnumpy(self.barycenters_mm),
            placement=self.placement,
            coil_name=self.coil_name,
            didt=self.didt,
        )

    def save(self, path: str | Path) -> None:
        """Write the conductivity-UQ result to a self-contained HDF5 file."""
        with h5py.File(Path(path), "w") as h5f:
            for name in (
                "mean_magnE",
                "std_magnE",
                "cov_magnE",
                "vols",
                "tet_tags",
                "barycenters_mm",
            ):
                h5f.create_dataset(
                    name, data=cp.asnumpy(getattr(self, name)), compression="gzip"
                )
            h5f.create_dataset("sigma_samples", data=np.asarray(self.sigma_samples))
            h5f.attrs["format_version"] = _FORMAT_VERSION
            h5f.attrs["n_samples"] = self.n_samples
            h5f.attrs["perturbed_tags"] = np.asarray(self.perturbed_tags, dtype=np.int32)
            h5f.attrs["coil_name"] = self.coil_name
            h5f.attrs["didt"] = self.didt
            h5f.attrs["placement_center_mm"] = self.placement.center_mm
            h5f.attrs["placement_handle_mm"] = self.placement.handle_mm
            h5f.attrs["placement_distance_mm"] = self.placement.distance_mm

    @classmethod
    def load(cls, path: str | Path) -> "ConductivityUQResult":
        """Read a saved conductivity-UQ result into NumPy arrays."""
        with h5py.File(Path(path), "r") as h5f:
            data = {
                k: np.asarray(h5f[k])
                for k in (
                    "mean_magnE",
                    "std_magnE",
                    "cov_magnE",
                    "vols",
                    "tet_tags",
                    "barycenters_mm",
                    "sigma_samples",
                )
            }
            placement = Placement(
                center_mm=h5f.attrs["placement_center_mm"],
                handle_mm=h5f.attrs["placement_handle_mm"],
                distance_mm=float(h5f.attrs["placement_distance_mm"]),
            )
            return cls(
                mean_magnE=data["mean_magnE"],
                std_magnE=data["std_magnE"],
                cov_magnE=data["cov_magnE"],
                n_samples=int(h5f.attrs["n_samples"]),
                perturbed_tags=tuple(int(t) for t in h5f.attrs["perturbed_tags"]),
                sigma_samples=data["sigma_samples"],
                vols=data["vols"],
                tet_tags=data["tet_tags"],
                barycenters_mm=data["barycenters_mm"],
                placement=placement,
                coil_name=str(h5f.attrs["coil_name"]),
                didt=float(h5f.attrs["didt"]),
            )
