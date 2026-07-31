"""Run TMS simulations and store their results."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import cupy as cp
import h5py
import numpy as np
import numpy.typing as npt

from cunibs import metrics
from cunibs.fem import (
    MAX_BLOCK,
    PlacementResult,
    SolverContext,
    build_context,
    solve_placements_block,
)
from cunibs.mesh import HeadMesh, load_mesh

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Self

    from cunibs.adm.target import ResolvedTarget
    from cunibs.coil import Coil
    from cunibs.uq import (
        ConductivityUQConfig,
        ConductivityUQPrecompute,
        ConductivityUQResult,
    )

_FORMAT_VERSION = 1

# The region every result is summarised over up front. Others are computed on demand from
# a retained magnE.
_DEFAULT_REGION: metrics.Region = "gray_matter"

_NO_ROIS: Mapping[str, ResolvedTarget] = MappingProxyType({})

# Per-tetrahedron metric inputs on the host, shared by reference across every FieldResult.
type _HostMetricInputs = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float64]
]


def _write_metrics(grp: h5py.Group, m: metrics.FieldMetrics) -> None:
    grp.attrs["region"] = m["region"]
    grp.attrs["peak_magnE"] = m["peak_magnE"]
    grp.attrs["region_volume_m3"] = m["region_volume_m3"]
    grp.create_dataset("peak_location_mm", data=m["peak_location_mm"])
    grp.create_dataset("center_of_gravity_mm", data=m["center_of_gravity_mm"])
    for name in ("focality_m3", "distribution"):
        sub = grp.create_group(name)
        for key, value in m[name].items():
            sub.attrs[key] = float(value)


def _read_metrics(grp: h5py.Group) -> metrics.FieldMetrics:
    return metrics.FieldMetrics(
        region=cast("metrics.Region", str(grp.attrs["region"])),
        peak_magnE=float(grp.attrs["peak_magnE"]),
        peak_location_mm=np.asarray(grp["peak_location_mm"]),
        center_of_gravity_mm=np.asarray(grp["center_of_gravity_mm"]),
        region_volume_m3=float(grp.attrs["region_volume_m3"]),
        focality_m3={k: float(v) for k, v in grp["focality_m3"].attrs.items()},
        distribution={k: float(v) for k, v in grp["distribution"].attrs.items()},
    )


def _check_format_version(h5f: h5py.File, path: str | Path, expected: int) -> None:
    stored = int(h5f.attrs.get("format_version", 0))
    if stored != expected:
        raise ValueError(
            f"{path}: file format version {stored} is not readable by this version of "
            f"cuNIBS (expected {expected})."
        )


def _opt_array(h5f: h5py.File, name: str) -> npt.NDArray[Any] | None:
    """Read a dataset that the run may not have retained."""
    return np.asarray(h5f[name]) if name in h5f else None


def _as_point(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {np.shape(value)}.")
    return p


def _sites(placements: Sequence[Placement], method: str) -> list[Placement]:
    if isinstance(placements, Placement):
        raise TypeError(
            f"{method}() takes a sequence of placements; wrap a single one in a list, or "
            "use the matching simulate() method."
        )
    return list(placements)


@dataclass(frozen=True)
class _Retain:
    """Which full-volume arrays a result keeps. Metrics are always computed regardless."""

    magnitude: bool
    vectors: bool
    potential: bool


@dataclass(frozen=True, init=False)
class Placement:
    """Define a coil placement on the scalp in millimetres.

    ``center_mm`` is the scalp target. ``handle_mm`` defines the positive handle direction.
    ``distance_mm`` sets the offset along the outward normal.

    All three must be finite, and ``handle_mm`` must differ from ``center_mm``. A handle on
    the outward normal through the scalp projection of ``center_mm`` is rejected too, but only
    once a mesh is available to project against.
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
        center = _as_point(center_mm)
        handle = _as_point(handle_mm)
        distance = float(distance_mm)
        if not (np.isfinite(center).all() and np.isfinite(handle).all()):
            raise ValueError("center_mm and handle_mm must be finite 3-vectors.")
        if not np.isfinite(distance):
            raise ValueError("distance_mm must be finite.")
        if np.array_equal(center, handle):
            raise ValueError(
                "handle_mm must differ from center_mm; it defines the coil's handle direction."
            )
        object.__setattr__(self, "center_mm", center)
        object.__setattr__(self, "handle_mm", handle)
        object.__setattr__(self, "distance_mm", distance)


class Subject:
    """Hold a head mesh and its reusable GPU solver state."""

    def __init__(self, mesh: HeadMesh) -> None:
        self._mesh = mesh
        self._ctx: SolverContext | None = None
        self._host_metrics: _HostMetricInputs | None = None
        self._conductivity_uq_pre: dict[tuple[int, ...], ConductivityUQPrecompute] = {}

    @classmethod
    def from_mesh(cls, mesh_file: str | Path) -> Subject:
        return cls(load_mesh(mesh_file))

    def free(self) -> None:
        """Release cached GPU state (solver context, AMG hierarchies, UQ precompute).

        Dropping the cached solver objects triggers their teardown, so a loop over many subjects
        can reclaim device memory as it goes. The subject stays usable afterwards; cached state is
        rebuilt lazily on the next call. Also available via the context manager:
        ``with Subject.from_mesh(path) as subject: ...``.
        """
        self._conductivity_uq_pre.clear()
        self._ctx = None
        self._host_metrics = None
        cp.get_default_memory_pool().free_all_blocks()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.free()

    @property
    def mesh(self) -> HeadMesh:
        return self._mesh

    @property
    def context(self) -> SolverContext:
        if self._ctx is None:
            self._ctx = build_context(self._mesh)
        return self._ctx

    def roi(
        self,
        point_mm: npt.ArrayLike,
        radius_mm: float = 0.0,
        region: metrics.Region = "gray_matter",
    ) -> ResolvedTarget:
        """Volume-weighted ROI of ``region`` elements around ``point_mm`` (nearest one if radius 0).

        Returns a :class:`~cunibs.adm.target.ResolvedTarget` usable as a ``record_rois`` probe or an
        ``adm`` target.
        """
        from cunibs.adm.target import Target, resolve_target

        return resolve_target(
            self.context, Target(point_mm, region=region, radius_mm=radius_mm)
        )

    def depth_probes(
        self,
        cortical_point_mm: npt.ArrayLike,
        inward_dir: npt.ArrayLike,
        depths_mm: Sequence[float],
        radius_mm: float = 0.0,
        region: metrics.Region = "all",
    ) -> list[ResolvedTarget]:
        """ROIs at increasing depth along ``inward_dir`` from a cortical point."""
        p = _as_point(cortical_point_mm)
        d = _as_point(inward_dir)
        norm = float(np.linalg.norm(d))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("inward_dir must be a non-zero, finite 3-vector.")
        d = d / norm
        return [
            self.roi(p + depth * d, radius_mm=radius_mm, region=region) for depth in depths_mm
        ]

    def _conductivity_uq_precompute(
        self, config: ConductivityUQConfig
    ) -> ConductivityUQPrecompute:
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

    @property
    def _host_metric_inputs(self) -> _HostMetricInputs:
        """Host ``(vols, tet_tags, barycenters_mm)``, built once and shared by every result."""
        if self._host_metrics is None:
            ctx = self.context
            self._host_metrics = (
                cp.asnumpy(ctx.vols),
                cp.asnumpy(ctx.tet_tags),
                np.asarray(self._mesh.tet_barycenters_mm),
            )
        return self._host_metrics

    def _reduce_chunk(
        self,
        chunk: Sequence[Placement],
        outs: Sequence[PlacementResult],
        coil: Coil,
        didt: float,
        retain: _Retain,
    ) -> list[FieldResult]:
        """Turn one solved chunk into results, summarising on the device.

        Runs inside the scratch pool, so nothing it returns may be device-backed.
        ``compute_metrics`` satisfies that: it returns Python floats and host arrays even
        when every input is on the device.
        """
        ctx = self.context
        # Pool-owned: freed with the chunk, so it is re-uploaded rather than cached.
        barycenters_mm = cp.asarray(self._mesh.tet_barycenters_mm)
        host = self._host_metric_inputs if retain.magnitude else None
        return [
            FieldResult(
                summary=metrics.compute_metrics(
                    out["magnE"],
                    ctx.vols,
                    barycenters_mm,
                    ctx.tet_tags,
                    region=_DEFAULT_REGION,
                ),
                magnE=cp.asnumpy(out["magnE"]) if retain.magnitude else None,
                E=cp.asnumpy(out["E"]) if retain.vectors else None,
                v=cp.asnumpy(out["v"]) if retain.potential else None,
                transform=np.asarray(out["transform"]),
                placement=site,
                coil_name=coil.name,
                didt=didt,
                vols=host[0] if host else None,
                tet_tags=host[1] if host else None,
                barycenters_mm=host[2] if host else None,
            )
            for site, out in zip(chunk, outs, strict=False)
        ]

    def _iter_reduced(
        self,
        coil: Coil,
        sites: Sequence[Placement],
        didt: float,
        block_k: int | None,
        retain: _Retain,
    ) -> Iterator[FieldResult]:
        """Solve ``sites`` in chunks, reducing inside a scratch pool and yielding outside it.

        The reduction runs under the scratch allocator so every intermediate is bulk-freed
        with its chunk, but the yield happens after the allocator is popped: control passes
        to caller code between chunks, and its allocations must not land in our pool.
        """
        ctx = self.context
        # Held across chunks, so these are allocated outside the scratch pool.
        dip_pos_m = cp.asarray(coil.positions_m)
        dip_moment = cp.asarray(coil.moments)
        # Placements share the stiffness matrix, so chunks of up to MAX_BLOCK solve as
        # one lockstep block CG that reads the matrix/hierarchy once for the whole chunk.
        chunk_k = MAX_BLOCK if block_k is None else max(1, min(MAX_BLOCK, block_k))
        temp_pool = cp.cuda.MemoryPool()
        try:
            for start in range(0, len(sites), chunk_k):
                chunk = sites[start : start + chunk_k]
                site_args = [(s.center_mm, s.handle_mm, s.distance_mm) for s in chunk]
                with cp.cuda.using_allocator(temp_pool.malloc):
                    outs = solve_placements_block(ctx, dip_pos_m, dip_moment, site_args, didt)
                    batch = self._reduce_chunk(chunk, outs, coil, didt, retain)
                    del outs
                yield from batch
        finally:
            temp_pool.free_all_blocks()

    def iter_simulate(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        didt: float = 1e6,
        *,
        magnitude: bool = False,
        vectors: bool = False,
        potential: bool = False,
        block_k: int | None = None,
    ) -> Iterator[FieldResult]:
        """Stream one :class:`FieldResult` per placement, in the order given.

        ``didt`` is the coil current's rate of change in A/s; the field is linear in it.

        Every result carries its gray-matter metrics; the flags say which of the
        full-volume arrays to keep as well. All three default to off because they are what
        makes a result large -- on a 4M-tetrahedron mesh they total roughly 70 MB, against
        about a kilobyte for a result with none of them.

        Retaining ``magnitude`` also unlocks the metrics a precomputed summary cannot
        answer: :meth:`FieldResult.summary_for` for a non-default region, and
        :meth:`FieldResult.focality` at an arbitrary fraction.

        Placements are solved in blocks of ``block_k`` that share a single stiffness and
        hierarchy read. It defaults to ``MAX_BLOCK`` and is clamped to ``[1, MAX_BLOCK]``.
        A block is solved and reduced as a unit, so it also caps peak memory: only
        ``block_k`` results are live at once, however long the sweep.

        See :meth:`iter_simulate_conductivity_uq` to stream a conductivity Monte Carlo
        instead.
        """
        # The loop lives in a separate generator so this stays a plain function: a `yield`
        # here would defer the check below to the caller's first next(), surfacing the
        # error at the consumption site instead of where the bad argument was passed.
        sites = _sites(placements, "iter_simulate")
        retain = _Retain(magnitude, vectors, potential)
        return self._iter_reduced(coil, sites, didt, block_k, retain)

    def simulate(
        self,
        coil: Coil,
        placement: Placement,
        didt: float = 1e6,
        *,
        magnitude: bool = False,
        vectors: bool = False,
        potential: bool = False,
        block_k: int | None = None,
    ) -> FieldResult:
        """Solve a single placement, taking the same flags as :meth:`iter_simulate`.

        Sequences go through :meth:`iter_simulate`, which bounds peak memory by the block
        rather than by the number of placements.
        """
        if not isinstance(placement, Placement):
            raise TypeError(
                "simulate() takes a single Placement. Use Subject.iter_simulate(...) to "
                "stream a sequence, wrapping it in list() if you want them all at once."
            )
        # Unpacking drives the generator to exhaustion, so its scratch pool is released.
        (result,) = self.iter_simulate(
            coil,
            [placement],
            didt,
            magnitude=magnitude,
            vectors=vectors,
            potential=potential,
            block_k=block_k,
        )
        return result

    def _iter_uq_reduced(
        self,
        coil: Coil,
        sites: Sequence[Placement],
        config: ConductivityUQConfig,
        didt: float,
        moments: bool,
        record_rois: Mapping[str, ResolvedTarget],
    ) -> Iterator[ConductivityUQResult]:
        """One Monte Carlo per placement, reduced inside a scratch pool and yielded outside.

        Unlike the deterministic path there is no block solve, so the chunk is a single
        placement; the pool scoping is otherwise identical.
        """
        from cunibs.uq import run_conductivity_uq

        ctx = self.context
        pre = self._conductivity_uq_precompute(config)
        temp_pool = cp.cuda.MemoryPool()
        try:
            for site in sites:
                with cp.cuda.using_allocator(temp_pool.malloc):
                    result = run_conductivity_uq(
                        ctx, pre, coil, site, config, didt, record_rois
                    )
                    # Summarised on the device before anything is copied back.
                    host = self._host_metric_inputs if moments else None
                    reduced = replace(
                        result,
                        summary=result.compute_summary(),
                        mean_magnE=cp.asnumpy(result.mean_magnE) if moments else None,
                        std_magnE=cp.asnumpy(result.std_magnE) if moments else None,
                        cov_magnE=cp.asnumpy(result.cov_magnE) if moments else None,
                        vols=host[0] if host else None,
                        tet_tags=host[1] if host else None,
                        barycenters_mm=host[2] if host else None,
                    )
                    del result
                yield reduced
        finally:
            temp_pool.free_all_blocks()

    def iter_simulate_conductivity_uq(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        config: ConductivityUQConfig,
        didt: float = 1e6,
        *,
        moments: bool = False,
        record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
    ) -> Iterator[ConductivityUQResult]:
        """Stream a conductivity Monte Carlo per placement, in the order given.

        Every sampled conductivity vector is solved with the same finite-element model, and
        :class:`~cunibs.uq.ConductivityUQResult` reports per-tetrahedron moments of ``|E|``
        instead of the deterministic :class:`FieldResult` that :meth:`iter_simulate`
        returns. Every result carries its gray-matter summary.

        ``moments=True`` also keeps the per-tetrahedron ``mean``/``std``/``cov`` arrays.
        The three come as a set, since any two determine the third.

        Every result carries per-draw arrays of the gray-matter peak ``|E|``, focality, and
        peak location. They are kilobytes, so there is no flag to enable them.
        ``record_rois``, a ``{name: ResolvedTarget}`` mapping of ROIs from :meth:`roi` / ``resolve_target``,
        adds each draw's volume-weighted mean ``|E|`` per ROI (``result.roi_samples[name]``).
        """
        # The loop lives in a separate generator so this stays a plain function: a `yield`
        # here would defer the check below to the caller's first next(), surfacing the error
        # at the consumption site instead of where the bad argument was passed.
        sites = _sites(placements, "iter_simulate_conductivity_uq")
        return self._iter_uq_reduced(coil, sites, config, didt, moments, record_rois)

    def simulate_conductivity_uq(
        self,
        coil: Coil,
        placement: Placement,
        config: ConductivityUQConfig,
        didt: float = 1e6,
        *,
        moments: bool = False,
        record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
    ) -> ConductivityUQResult:
        """Run a conductivity Monte Carlo for a single placement.

        Takes the same options as :meth:`iter_simulate_conductivity_uq`; sequences go
        through that method, which holds only one placement's moments at a time.
        """
        if not isinstance(placement, Placement):
            raise TypeError(
                "simulate_conductivity_uq() takes a single Placement. Use "
                "Subject.iter_simulate_conductivity_uq(...) to stream a sequence, "
                "wrapping it in list() if you want them all at once."
            )
        # Unpacking drives the generator to exhaustion, so its scratch pool is released.
        (result,) = self.iter_simulate_conductivity_uq(
            coil, [placement], config, didt, moments=moments, record_rois=record_rois
        )
        return result


@dataclass
class FieldResult:
    """Metrics for one placement, plus whichever full-volume arrays were retained.

    ``summary`` holds the gray-matter metrics and is always present. ``magnE``, ``E`` and
    ``v`` are ``None`` unless the run asked for them with ``magnitude=``, ``vectors=`` or
    ``potential=``. All arrays are NumPy.

    ``vols``, ``tet_tags`` and ``barycenters_mm`` accompany ``magnE`` -- they are what the
    on-demand metrics need. Results from one subject share them, so treat them as
    read-only.
    """

    summary: metrics.FieldMetrics
    magnE: npt.NDArray[np.float32] | None
    E: npt.NDArray[np.float32] | None
    v: npt.NDArray[np.float64] | None
    transform: npt.NDArray[np.float64]
    placement: Placement
    coil_name: str
    didt: float
    vols: npt.NDArray[np.float32] | None
    tet_tags: npt.NDArray[np.int32] | None
    barycenters_mm: npt.NDArray[np.float64] | None
    _summaries: dict[str, metrics.FieldMetrics] = field(default_factory=dict, repr=False)

    def _require_magnitude(self, what: str) -> npt.NDArray[np.float32]:
        if self.magnE is None:
            raise ValueError(
                f"{what} needs the per-element |E|, which this result did not retain. "
                "Re-run with magnitude=True."
            )
        return self.magnE

    def peak_magnE(self, region: metrics.Region = _DEFAULT_REGION) -> float:
        return self.summary_for(region)["peak_magnE"]

    def peak_location_mm(
        self, region: metrics.Region = _DEFAULT_REGION
    ) -> npt.NDArray[np.float64]:
        return self.summary_for(region)["peak_location_mm"]

    def center_of_gravity_mm(
        self, region: metrics.Region = _DEFAULT_REGION
    ) -> npt.NDArray[np.float64]:
        return self.summary_for(region)["center_of_gravity_mm"]

    def focality(self, frac: float = 0.5, region: metrics.Region = _DEFAULT_REGION) -> float:
        """Volume with ``|E| >= frac *`` the 99.9th percentile of |E| in the region.

        The anchor is a high percentile rather than :meth:`peak_magnE` so that one outlier
        element cannot move the threshold; see :func:`cunibs.metrics.focality`.

        ``summary`` already carries the default fractions; any other one needs ``magnE``,
        so the run has to have passed ``magnitude=True``.
        """
        key = f"{frac:g}"
        summary = self.summary_for(region)
        if key in summary["focality_m3"]:
            return summary["focality_m3"][key]
        magnE = self._require_magnitude(f"focality at frac={key}")
        assert self.vols is not None and self.tet_tags is not None
        mask = metrics.region_mask(self.tet_tags, region)
        return metrics.focality(magnE, self.vols, mask, frac)

    def summary_for(self, region: metrics.Region) -> metrics.FieldMetrics:
        """Metrics for a tissue region, cached.

        ``summary``'s own region is free; any other needs ``magnE``, so the run has to have
        passed ``magnitude=True``.
        """
        if region == self.summary["region"]:
            return self.summary
        if region not in self._summaries:
            magnE = self._require_magnitude(f"metrics for region {region!r}")
            assert self.vols is not None and self.barycenters_mm is not None
            assert self.tet_tags is not None
            self._summaries[region] = metrics.compute_metrics(
                magnE, self.vols, self.barycenters_mm, self.tet_tags, region=region
            )
        return self._summaries[region]

    def save(self, path: str | Path) -> None:
        """Write the result to a self-contained HDF5 file.

        Arrays the run did not retain are absent from the file and load back as ``None``.
        """
        with h5py.File(Path(path), "w") as h5f:
            for name in ("magnE", "E", "v", "vols", "tet_tags", "barycenters_mm"):
                arr = getattr(self, name)
                if arr is None:
                    continue
                h5f.create_dataset(name, data=arr, compression="gzip")
            h5f.create_dataset("transform", data=self.transform)
            _write_metrics(h5f.create_group("summary"), self.summary)
            h5f.attrs["format_version"] = _FORMAT_VERSION
            h5f.attrs["coil_name"] = self.coil_name
            h5f.attrs["didt"] = self.didt
            h5f.attrs["placement_center_mm"] = self.placement.center_mm
            h5f.attrs["placement_handle_mm"] = self.placement.handle_mm
            h5f.attrs["placement_distance_mm"] = self.placement.distance_mm

    @classmethod
    def load(cls, path: str | Path) -> FieldResult:
        """Read a saved result into NumPy arrays."""
        with h5py.File(Path(path), "r") as h5f:
            _check_format_version(h5f, path, _FORMAT_VERSION)
            return cls(
                summary=_read_metrics(h5f["summary"]),
                magnE=_opt_array(h5f, "magnE"),
                E=_opt_array(h5f, "E"),
                v=_opt_array(h5f, "v"),
                transform=np.asarray(h5f["transform"]),
                placement=Placement(
                    center_mm=h5f.attrs["placement_center_mm"],
                    handle_mm=h5f.attrs["placement_handle_mm"],
                    distance_mm=float(h5f.attrs["placement_distance_mm"]),
                ),
                coil_name=str(h5f.attrs["coil_name"]),
                didt=float(h5f.attrs["didt"]),
                vols=_opt_array(h5f, "vols"),
                tet_tags=_opt_array(h5f, "tet_tags"),
                barycenters_mm=_opt_array(h5f, "barycenters_mm"),
            )
