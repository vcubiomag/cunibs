"""Pieces shared by the result types and the modules that build them."""

from __future__ import annotations

from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from cunibs import metrics

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
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
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, str | int | float) else str(value)


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


@contextmanager
def corrupt_as_value_error(path: str | Path) -> Iterator[None]:
    """Report damage inside a valid HDF5 file as a ValueError.

    h5py raises whichever of these its failing step happens to.
    """
    try:
        yield
    except (RuntimeError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"{path}: corrupt HDF5 file: {exc}") from exc


def check_same_length(h5f: h5py.File, path: str | Path, names: Iterable[str]) -> None:
    """Require the datasets among ``names`` that the file holds to agree in their first axis."""
    lengths = {name: h5f[name].shape[0] for name in names if name in h5f}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{path}: corrupt HDF5 file: dataset lengths differ: {lengths}")


def check_roi_names(names: Iterable[str]) -> None:
    """Reject ROI names HDF5 would resolve to another path, and so lose on a round trip."""
    for name in names:
        parts = name.split("/")
        if not all(parts) or "." in parts or ".." in parts:
            raise ValueError(f"ROI name {name!r} is not a usable HDF5 path.")


def opt_array(h5f: h5py.File, name: str) -> npt.NDArray[Any] | None:
    """Read a dataset that the run may not have retained."""
    return np.asarray(h5f[name]) if name in h5f else None


def check_didt(didt: float) -> None:
    if not np.isfinite(didt):
        raise ValueError(f"didt must be finite, got {didt!r}.")


def as_point(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.array(value, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {np.shape(value)}.")
    return p
