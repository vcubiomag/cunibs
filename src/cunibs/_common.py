"""Pieces shared by the result types and the modules that build them."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from cunibs import metrics

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import h5py

    from cunibs.adm.target import ResolvedTarget

# Root attributes that describe the run rather than the result, read back as a unit. Additive, so
# a file written before they existed still loads and reports None.
PROVENANCE_KEYS = (
    "software",
    "softwareVersion",
    "softwareDependencies",
    "date",
    "machine",
    "mesh_content_hash",
    "mesh_n_nodes",
    "mesh_n_tets",
    "coil_n_dipoles",
    "solver_tolerance",
    "sweep_basis_coil",
    "sweep_basis_size",
)

# The region every result is summarised over up front. Others are computed on demand from
# a retained magnE.
DEFAULT_REGION: metrics.Region = "gray_matter"

NO_ROIS: Mapping[str, ResolvedTarget] = MappingProxyType({})


def write_provenance(h5f: h5py.File, provenance: Mapping[str, object]) -> None:
    """Stamp the provenance as root attributes, beside the inputs already written there."""
    for key, value in provenance.items():
        if value is not None:
            h5f.attrs[key] = value


def plain(value: Any) -> Any:
    """An HDF5 attribute as a plain Python value."""
    if isinstance(value, bytes):
        return value.decode()
    return value.item() if hasattr(value, "item") else value


def read_provenance(h5f: h5py.File) -> dict[str, Any] | None:
    """The provenance attributes, or ``None`` for a file that carries none."""
    return {k: plain(h5f.attrs[k]) for k in PROVENANCE_KEYS if k in h5f.attrs} or None


def write_metrics(grp: h5py.Group, m: metrics.FieldMetrics) -> None:
    grp.attrs["region"] = m["region"]
    grp.attrs["peak_magnE"] = m["peak_magnE"]
    grp.attrs["region_volume_m3"] = m["region_volume_m3"]
    grp.create_dataset("peak_location_mm", data=m["peak_location_mm"])
    grp.create_dataset("center_of_gravity_mm", data=m["center_of_gravity_mm"])
    for name in ("focality_m3", "distribution"):
        sub = grp.create_group(name)
        for key, value in m[name].items():
            sub.attrs[key] = float(value)


def read_metrics(grp: h5py.Group) -> metrics.FieldMetrics:
    return metrics.FieldMetrics(
        region=cast("metrics.Region", str(grp.attrs["region"])),
        peak_magnE=float(grp.attrs["peak_magnE"]),
        peak_location_mm=np.asarray(grp["peak_location_mm"]),
        center_of_gravity_mm=np.asarray(grp["center_of_gravity_mm"]),
        region_volume_m3=float(grp.attrs["region_volume_m3"]),
        focality_m3={k: float(v) for k, v in grp["focality_m3"].attrs.items()},
        distribution={k: float(v) for k, v in grp["distribution"].attrs.items()},
    )


def check_format_version(h5f: h5py.File, path: str | Path, expected: int) -> None:
    stored = int(h5f.attrs.get("format_version", 0))
    if stored != expected:
        raise ValueError(
            f"{path}: file format version {stored} is not readable by this version of "
            f"cuNIBS (expected {expected})."
        )


def opt_array(h5f: h5py.File, name: str) -> npt.NDArray[Any] | None:
    """Read a dataset that the run may not have retained."""
    return np.asarray(h5f[name]) if name in h5f else None


def as_point(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {np.shape(value)}.")
    return p
