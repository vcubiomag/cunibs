"""Run TMS simulations and store their results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TypeAlias, overload

import cupy as cp
import h5py
import numpy as np
import numpy.typing as npt

from cunibs import metrics
from cunibs.coil import Coil
from cunibs.fem import SolverContext, build_context, solve_placement
from cunibs.mesh import HeadMesh, load_mesh

_FORMAT_VERSION = 1

ArrayT: TypeAlias = cp.ndarray | np.ndarray


def _as_point(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {np.shape(value)}.")
    return p


@dataclass(frozen=True, init=False)
class Placement:
    """Define a coil placement on the scalp in millimetres.

    ``center_mm`` is the scalp target. ``handle_mm`` defines the positive handle direction.
    ``distance_mm`` sets the offset along the outward normal.
    """

    center_mm: npt.NDArray[np.float64]
    handle_mm: npt.NDArray[np.float64]
    distance_mm: float = 4.0

    def __init__(
        self,
        center_mm: npt.ArrayLike,
        handle_mm: npt.ArrayLike,
        distance_mm: float = 4.0,
    ) -> None:
        object.__setattr__(self, "center_mm", _as_point(center_mm))
        object.__setattr__(self, "handle_mm", _as_point(handle_mm))
        object.__setattr__(self, "distance_mm", float(distance_mm))


class Subject:
    """Hold a head mesh and its reusable GPU solver state."""

    def __init__(self, mesh: HeadMesh) -> None:
        self._mesh = mesh
        self._ctx: SolverContext | None = None
        self._barycenters_mm: cp.ndarray | None = None

    @classmethod
    def from_mesh(cls, mesh_file: str | Path) -> "Subject":
        return cls(load_mesh(mesh_file))

    @property
    def mesh(self) -> HeadMesh:
        return self._mesh

    @property
    def context(self) -> SolverContext:
        if self._ctx is None:
            self._ctx = build_context(self._mesh)
        return self._ctx

    @overload
    def simulate(
        self, coil: Coil, placements: Placement, didt: float = ...
    ) -> "FieldResult": ...
    @overload
    def simulate(
        self, coil: Coil, placements: Sequence[Placement], didt: float = ...
    ) -> list["FieldResult"]: ...

    def simulate(
        self,
        coil: Coil,
        placements: Placement | Sequence[Placement],
        didt: float = 1e6,
    ) -> "FieldResult | list[FieldResult]":
        """Solve one placement or a sequence of placements."""
        single = isinstance(placements, Placement)
        sites = [placements] if single else list(placements)
        ctx = self.context
        if self._barycenters_mm is None:
            self._barycenters_mm = cp.asarray(self._mesh.tet_barycenters_mm)

        results: list[FieldResult] = []
        for site in sites:
            out = solve_placement(
                ctx,
                coil.positions_m,
                coil.moments,
                site.center_mm,
                site.handle_mm,
                site.distance_mm,
                didt,
            )
            results.append(
                FieldResult(
                    E=out["E"],
                    magnE=out["magnE"],
                    v=out["v"],
                    transform=out["transform"],
                    placement=site,
                    coil_name=coil.name,
                    didt=didt,
                    vols=ctx.vols,
                    tet_tags=ctx.tet_tags,
                    barycenters_mm=self._barycenters_mm,
                )
            )
        return results[0] if single else results


@dataclass
class FieldResult:
    """Store the E-field and metric inputs for one placement.

    Arrays use CuPy after simulation and NumPy after :meth:`load` or :meth:`to_numpy`.
    """

    E: ArrayT
    magnE: ArrayT
    v: ArrayT
    transform: npt.NDArray[np.float64]
    placement: Placement
    coil_name: str
    didt: float
    vols: ArrayT
    tet_tags: ArrayT
    barycenters_mm: ArrayT
    _summaries: dict[str, metrics.FieldMetrics] = field(default_factory=dict, repr=False)

    def _mask(self, region: str) -> ArrayT:
        return metrics.region_mask(self.tet_tags, region)

    def peak_magnE(self, region: str = "gray_matter") -> float:
        return metrics.peak_magnitude(self.magnE, self._mask(region))

    def peak_location_mm(self, region: str = "gray_matter") -> npt.NDArray[np.float64]:
        return metrics.peak_location_mm(self.magnE, self.barycenters_mm, self._mask(region))

    def focality(self, frac: float = 0.5, region: str = "gray_matter") -> float:
        return metrics.focality(self.magnE, self.vols, self._mask(region), frac)

    def summary(self, region: str = "gray_matter") -> metrics.FieldMetrics:
        """Return cached metrics for a tissue region."""
        if region not in self._summaries:
            self._summaries[region] = metrics.compute_metrics(
                self.magnE, self.vols, self.barycenters_mm, self.tet_tags, region=region
            )
        return self._summaries[region]

    def to_numpy(self) -> "FieldResult":
        """Copy all arrays to NumPy."""
        return FieldResult(
            E=cp.asnumpy(self.E),
            magnE=cp.asnumpy(self.magnE),
            v=cp.asnumpy(self.v),
            transform=np.asarray(self.transform),
            placement=self.placement,
            coil_name=self.coil_name,
            didt=self.didt,
            vols=cp.asnumpy(self.vols),
            tet_tags=cp.asnumpy(self.tet_tags),
            barycenters_mm=cp.asnumpy(self.barycenters_mm),
        )

    def save(self, path: str | Path) -> None:
        """Write the result to a self-contained HDF5 file."""
        with h5py.File(Path(path), "w") as h5f:
            for name in ("E", "magnE", "v", "vols", "tet_tags", "barycenters_mm"):
                h5f.create_dataset(
                    name, data=cp.asnumpy(getattr(self, name)), compression="gzip"
                )
            h5f.create_dataset("transform", data=np.asarray(self.transform))
            h5f.attrs["format_version"] = _FORMAT_VERSION
            h5f.attrs["coil_name"] = self.coil_name
            h5f.attrs["didt"] = self.didt
            h5f.attrs["placement_center_mm"] = self.placement.center_mm
            h5f.attrs["placement_handle_mm"] = self.placement.handle_mm
            h5f.attrs["placement_distance_mm"] = self.placement.distance_mm

    @classmethod
    def load(cls, path: str | Path) -> "FieldResult":
        """Read a saved result into NumPy arrays."""
        with h5py.File(Path(path), "r") as h5f:
            data = {
                k: np.asarray(h5f[k])
                for k in ("E", "magnE", "v", "transform", "vols", "tet_tags", "barycenters_mm")
            }
            placement = Placement(
                center_mm=h5f.attrs["placement_center_mm"],
                handle_mm=h5f.attrs["placement_handle_mm"],
                distance_mm=float(h5f.attrs["placement_distance_mm"]),
            )
            coil_name = str(h5f.attrs["coil_name"])
            didt = float(h5f.attrs["didt"])
        return cls(
            E=data["E"],
            magnE=data["magnE"],
            v=data["v"],
            transform=data["transform"],
            placement=placement,
            coil_name=coil_name,
            didt=didt,
            vols=data["vols"],
            tet_tags=data["tet_tags"],
            barycenters_mm=data["barycenters_mm"],
        )
