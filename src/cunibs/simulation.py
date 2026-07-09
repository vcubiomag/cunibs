"""Run TMS simulations and store their results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence, TypeAlias, overload

import cupy as cp
import h5py
import numpy as np
import numpy.typing as npt

from cunibs import metrics
from cunibs.coil import Coil
from cunibs.fem import SolverContext, build_context, solve_placement
from cunibs.mesh import HeadMesh, load_mesh

if TYPE_CHECKING:
    from cunibs.uq import (
        ConductivityUQConfig,
        ConductivityUQPrecompute,
        ConductivityUQResult,
        ConductivityUQSummary,
    )

_FORMAT_VERSION = 1

ArrayT: TypeAlias = cp.ndarray | np.ndarray
Device: TypeAlias = Literal["cpu", "gpu"]


def _as_point(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {np.shape(value)}.")
    return p


def _to_numpy_array(value: ArrayT | npt.ArrayLike) -> npt.NDArray:
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def _copy_metrics(m: metrics.FieldMetrics) -> metrics.FieldMetrics:
    return {
        "region": m["region"],
        "peak_magnE": float(m["peak_magnE"]),
        "peak_location_mm": np.asarray(m["peak_location_mm"], dtype=np.float64),
        "center_of_gravity_mm": np.asarray(m["center_of_gravity_mm"], dtype=np.float64),
        "region_volume_m3": float(m["region_volume_m3"]),
        "focality_m3": dict(m["focality_m3"]),
        "distribution": dict(m["distribution"]),
    }


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
        self._host_vols: npt.NDArray[np.float32] | None = None
        self._host_tet_tags: npt.NDArray[np.int32] | None = None
        self._host_barycenters_mm: npt.NDArray[np.float64] | None = None
        self._conductivity_uq_pre: dict[tuple[int, ...], "ConductivityUQPrecompute"] = {}

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

    def _conductivity_uq_precompute(
        self, config: "ConductivityUQConfig"
    ) -> "ConductivityUQPrecompute":
        """Build (and cache) the per-tissue stiffness components for a UQ configuration.

        Cached by the set of perturbed tissues so repeated UQ runs on the same subject reuse the
        components and the nominal-σ AMG hierarchy.
        """
        from cunibs.uq import build_conductivity_uq_precompute

        ctx = self.context
        if config.perturbed_tags is not None:
            tags = tuple(sorted(int(t) for t in config.perturbed_tags))
        else:
            tags = tuple(int(t) for t in cp.asnumpy(cp.unique(ctx.tet_tags)))
        if tags not in self._conductivity_uq_pre:
            self._conductivity_uq_pre[tags] = build_conductivity_uq_precompute(ctx, tags)
        return self._conductivity_uq_pre[tags]

    def _host_metric_inputs(
        self, ctx: SolverContext
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
    ]:
        if self._host_vols is None:
            self._host_vols = cp.asnumpy(ctx.vols)
            self._host_tet_tags = cp.asnumpy(ctx.tet_tags)
            self._host_barycenters_mm = np.asarray(self._mesh.tet_barycenters_mm)
        assert self._host_tet_tags is not None
        assert self._host_barycenters_mm is not None
        return self._host_vols, self._host_tet_tags, self._host_barycenters_mm

    def _field_summary(
        self,
        out: dict[str, ArrayT],
        site: Placement,
        coil: Coil,
        didt: float,
        barycenters_mm: ArrayT,
    ) -> "FieldSummary":
        summary = metrics.compute_metrics(
            out["magnE"],
            self.context.vols,
            barycenters_mm,
            self.context.tet_tags,
            region="gray_matter",
        )
        return FieldSummary(
            summary=_copy_metrics(summary),
            transform=np.asarray(out["transform"]),
            placement=site,
            coil_name=coil.name,
            didt=didt,
        )

    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Placement,
        didt: float = ...,
        conductivity_uq: None = ...,
        retain_fields: Literal[False] = ...,
        device: Device = ...,
    ) -> "FieldSummary": ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        didt: float = ...,
        conductivity_uq: None = ...,
        retain_fields: Literal[False] = ...,
        device: Device = ...,
    ) -> list["FieldSummary"]: ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Placement,
        didt: float = ...,
        conductivity_uq: None = ...,
        retain_fields: Literal[True] = ...,
        device: Literal["cpu"] = ...,
    ) -> "FieldResult": ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        didt: float = ...,
        conductivity_uq: None = ...,
        retain_fields: Literal[True] = ...,
        device: Literal["cpu"] = ...,
    ) -> list["FieldResult"]: ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Placement,
        didt: float = ...,
        conductivity_uq: None = ...,
        retain_fields: Literal[True] = ...,
        device: Literal["gpu"] = ...,
    ) -> "FieldResult": ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        didt: float = ...,
        conductivity_uq: None = ...,
        retain_fields: Literal[True] = ...,
        device: Literal["gpu"] = ...,
    ) -> list["FieldResult"]: ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Placement,
        didt: float = ...,
        conductivity_uq: "ConductivityUQConfig" = ...,
        retain_fields: Literal[False] = ...,
        device: Device = ...,
    ) -> "ConductivityUQSummary": ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        didt: float = ...,
        conductivity_uq: "ConductivityUQConfig" = ...,
        retain_fields: Literal[False] = ...,
        device: Device = ...,
    ) -> list["ConductivityUQSummary"]: ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Placement,
        didt: float = ...,
        conductivity_uq: "ConductivityUQConfig" = ...,
        retain_fields: Literal[True] = ...,
        device: Device = ...,
    ) -> "ConductivityUQResult": ...
    @overload
    def simulate(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        didt: float = ...,
        conductivity_uq: "ConductivityUQConfig" = ...,
        retain_fields: Literal[True] = ...,
        device: Device = ...,
    ) -> list["ConductivityUQResult"]: ...

    def simulate(
        self,
        coil: Coil,
        placements: Placement | Sequence[Placement],
        didt: float = 1e6,
        conductivity_uq: "ConductivityUQConfig | None" = None,
        *,
        retain_fields: bool = False,
        device: Device = "cpu",
    ) -> (
        "FieldSummary | list[FieldSummary] | FieldResult | list[FieldResult] | "
        "ConductivityUQSummary | list[ConductivityUQSummary] | "
        "ConductivityUQResult | list[ConductivityUQResult]"
    ):
        """Solve one placement or a sequence of placements.

        With ``conductivity_uq`` set, run a conductivity Monte Carlo per placement and return
        :class:`~cunibs.uq.ConductivityUQResult` moments instead of a deterministic
        :class:`FieldResult`.

        By default (``retain_fields=False``) only compact host-side summaries are returned and
        the full-volume field arrays are freed, so callers can loop over many subjects without
        accumulating device memory. Pass ``retain_fields=True`` to get the full result back.

        ``device`` selects where retained fields live (``"gpu"`` keeps them on the device,
        ``"cpu"`` copies them to host). It has no effect when ``retain_fields=False``, since no
        field arrays are kept in that case.
        """
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'.")
        single = isinstance(placements, Placement)
        sites = [placements] if single else list(placements)
        ctx = self.context

        if conductivity_uq is not None:
            from cunibs.uq import (
                ConductivityUQResult,
                ConductivityUQSummary,
                run_conductivity_uq,
            )

            pre = self._conductivity_uq_precompute(conductivity_uq)
            uq_results: list[ConductivityUQResult | ConductivityUQSummary] = []
            temp_pool = cp.cuda.MemoryPool()
            for site in sites:
                if retain_fields and device == "gpu":
                    result = run_conductivity_uq(ctx, pre, coil, site, conductivity_uq, didt)
                else:
                    with cp.cuda.using_allocator(temp_pool.malloc):
                        result = run_conductivity_uq(
                            ctx, pre, coil, site, conductivity_uq, didt
                        )
                        if not retain_fields:
                            uq_results.append(result.summary())
                            del result
                            continue
                if not retain_fields:
                    continue
                if device == "gpu":
                    uq_results.append(result)
                    continue

                vols, tet_tags, barycenters_mm = self._host_metric_inputs(ctx)
                uq_results.append(
                    ConductivityUQResult(
                        mean_magnE=cp.asnumpy(result.mean_magnE),
                        std_magnE=cp.asnumpy(result.std_magnE),
                        cov_magnE=cp.asnumpy(result.cov_magnE),
                        n_samples=result.n_samples,
                        perturbed_tags=result.perturbed_tags,
                        sigma_samples=np.asarray(result.sigma_samples),
                        vols=vols,
                        tet_tags=tet_tags,
                        barycenters_mm=barycenters_mm,
                        placement=result.placement,
                        coil_name=result.coil_name,
                        didt=result.didt,
                    )
                )
                del result
            temp_pool.free_all_blocks()
            return uq_results[0] if single else uq_results

        results: list[FieldResult | FieldSummary] = []
        dip_pos_m = cp.asarray(coil.positions_m)
        dip_moment = cp.asarray(coil.moments)
        temp_pool = cp.cuda.MemoryPool()
        for site in sites:
            if retain_fields and device == "gpu":
                out = solve_placement(
                    ctx,
                    dip_pos_m,
                    dip_moment,
                    site.center_mm,
                    site.handle_mm,
                    site.distance_mm,
                    didt,
                )
            else:
                with cp.cuda.using_allocator(temp_pool.malloc):
                    out = solve_placement(
                        ctx,
                        dip_pos_m,
                        dip_moment,
                        site.center_mm,
                        site.handle_mm,
                        site.distance_mm,
                        didt,
                    )
                    if not retain_fields:
                        barycenters_mm = cp.asarray(self._mesh.tet_barycenters_mm)
                        results.append(
                            self._field_summary(out, site, coil, didt, barycenters_mm)
                        )
                        del out
                        continue
            if not retain_fields:
                continue

            if device == "gpu":
                if self._barycenters_mm is None:
                    self._barycenters_mm = cp.asarray(self._mesh.tet_barycenters_mm)
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
                continue

            vols, tet_tags, barycenters_mm = self._host_metric_inputs(ctx)
            results.append(
                FieldResult(
                    E=cp.asnumpy(out["E"]),
                    magnE=cp.asnumpy(out["magnE"]),
                    v=cp.asnumpy(out["v"]),
                    transform=np.asarray(out["transform"]),
                    placement=site,
                    coil_name=coil.name,
                    didt=didt,
                    vols=vols,
                    tet_tags=tet_tags,
                    barycenters_mm=barycenters_mm,
                )
            )
            del out
        temp_pool.free_all_blocks()
        return results[0] if single else results


@dataclass
class FieldSummary:
    """Compact CPU-side metrics for one deterministic placement."""

    summary: metrics.FieldMetrics
    transform: npt.NDArray[np.float64]
    placement: Placement
    coil_name: str
    didt: float

    def peak_magnE(self, region: metrics.Region = "gray_matter") -> float:
        if region != self.summary["region"]:
            raise ValueError("Only the default gray_matter summary is retained.")
        return self.summary["peak_magnE"]

    def peak_location_mm(
        self, region: metrics.Region = "gray_matter"
    ) -> npt.NDArray[np.float64]:
        if region != self.summary["region"]:
            raise ValueError("Only the default gray_matter summary is retained.")
        return self.summary["peak_location_mm"]

    def focality(self, frac: float = 0.5, region: metrics.Region = "gray_matter") -> float:
        if region != self.summary["region"]:
            raise ValueError("Only the default gray_matter summary is retained.")
        key = f"{frac:g}"
        if key not in self.summary["focality_m3"]:
            available = ", ".join(sorted(self.summary["focality_m3"]))
            raise ValueError(
                f"Focality at frac={key} was not retained (available: {available}). "
                "Pass retain_fields=True to compute focality at arbitrary fractions."
            )
        return self.summary["focality_m3"][key]


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

    def _mask(self, region: metrics.Region) -> ArrayT:
        return metrics.region_mask(self.tet_tags, region)

    def peak_magnE(self, region: metrics.Region = "gray_matter") -> float:
        return metrics.peak_magnitude(self.magnE, self._mask(region))

    def peak_location_mm(
        self, region: metrics.Region = "gray_matter"
    ) -> npt.NDArray[np.float64]:
        return metrics.peak_location_mm(self.magnE, self.barycenters_mm, self._mask(region))

    def focality(self, frac: float = 0.5, region: metrics.Region = "gray_matter") -> float:
        return metrics.focality(self.magnE, self.vols, self._mask(region), frac)

    def summary(self, region: metrics.Region = "gray_matter") -> metrics.FieldMetrics:
        """Return cached metrics for a tissue region."""
        if region not in self._summaries:
            self._summaries[region] = metrics.compute_metrics(
                self.magnE, self.vols, self.barycenters_mm, self.tet_tags, region=region
            )
        return self._summaries[region]

    def to_numpy(self) -> "FieldResult":
        """Copy all arrays to NumPy."""
        return FieldResult(
            E=_to_numpy_array(self.E),
            magnE=_to_numpy_array(self.magnE),
            v=_to_numpy_array(self.v),
            transform=np.asarray(self.transform),
            placement=self.placement,
            coil_name=self.coil_name,
            didt=self.didt,
            vols=_to_numpy_array(self.vols),
            tet_tags=_to_numpy_array(self.tet_tags),
            barycenters_mm=_to_numpy_array(self.barycenters_mm),
        )

    def save(self, path: str | Path) -> None:
        """Write the result to a self-contained HDF5 file."""
        with h5py.File(Path(path), "w") as h5f:
            for name in ("E", "magnE", "v", "vols", "tet_tags", "barycenters_mm"):
                h5f.create_dataset(
                    name, data=_to_numpy_array(getattr(self, name)), compression="gzip"
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
