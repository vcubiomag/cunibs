"""Run TMS simulations and store their results."""

from __future__ import annotations

import threading
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import version
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
    Recovery,
    RecoveryOperator,
    SolverContext,
    build_context,
    ensure_recovery,
    solve_placements_block,
    validate_recovery,
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

# Root attributes that describe the run rather than the result, read back as a unit. Additive, so
# a file written before they existed still loads and reports None.
_PROVENANCE_KEYS = (
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
)

# The region every result is summarised over up front. Others are computed on demand from
# a retained magnE.
_DEFAULT_REGION: metrics.Region = "gray_matter"

_NO_ROIS: Mapping[str, ResolvedTarget] = MappingProxyType({})

# Per-tetrahedron metric inputs on the host, shared by reference across every FieldResult.
type _HostMetricInputs = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float64]
]

# One ROI's name and the element indices and volume weights it reduces over.
type _RoiProbe = tuple[str, cp.ndarray, cp.ndarray]


@cache
def _environment() -> dict[str, str]:
    """The code, its dependencies and the hardware, probed once per process.

    Keys follow the openPMD base standard's root attributes, the convention for a simulation code
    stamping HDF5 output. ``machine`` carries the GPU and its compute capability because that is
    what a result depends on: the same inputs reproduce bitwise on the same hardware and software,
    and may differ across GPU architectures or CUDA versions.

    Provenance must never be the reason a save fails, so anything that cannot be probed is left
    out rather than fabricated.
    """
    info: dict[str, str] = {"software": "cuNIBS"}
    with suppress(Exception):
        info["softwareVersion"] = version("cunibs")

    deps: list[str] = []
    for module in ("numpy", "h5py"):
        with suppress(Exception):
            deps.append(f"{module}-{version(module)}")
    with suppress(Exception):
        deps.append(f"cupy-{cp.__version__}")
    with suppress(Exception):
        runtime = cp.cuda.runtime
        deps.append(f"cuda-runtime-{runtime.runtimeGetVersion()}")
        deps.append(f"cuda-driver-{runtime.driverGetVersion()}")
    with suppress(Exception):
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = props["name"]
        info["machine"] = (
            f"{name.decode() if isinstance(name, bytes) else name}; "
            f"sm_{props['major']}{props['minor']}"
        )
    if deps:
        info["softwareDependencies"] = "; ".join(deps)
    return info


def _write_provenance(h5f: h5py.File, provenance: Mapping[str, object]) -> None:
    """Stamp the provenance as root attributes, beside the inputs already written there."""
    for key, value in provenance.items():
        if value is not None:
            h5f.attrs[key] = value


def _plain(value: Any) -> Any:
    """An HDF5 attribute as a plain Python value."""
    if isinstance(value, bytes):
        return value.decode()
    return value.item() if hasattr(value, "item") else value


def _read_provenance(h5f: h5py.File) -> dict[str, Any] | None:
    """The provenance attributes, or ``None`` for a file that carries none."""
    return {k: _plain(h5f.attrs[k]) for k in _PROVENANCE_KEYS if k in h5f.attrs} or None


def _roi_means(magn: cp.ndarray, probes: Sequence[_RoiProbe]) -> dict[str, float] | None:
    """Volume-weighted mean ``|E|`` per ROI, reduced on the device against the solved field."""
    if not probes:
        return None
    return {name: float((magn[idx].astype(cp.float64) * w).sum()) for name, idx, w in probes}


def _roi_probes(record_rois: Mapping[str, ResolvedTarget]) -> list[_RoiProbe]:
    """The element indices and weights each ROI reduces over, as the UQ path prepares them."""
    return [
        (
            name,
            cp.ascontiguousarray(target.elem_idx.astype(cp.int64)),
            cp.ascontiguousarray(target.weights.astype(cp.float64)),
        )
        for name, target in record_rois.items()
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


def _scratch_pool() -> cp.cuda.MemoryPool:
    """A pool for chunk-scoped temporaries, with the default pool's free blocks handed back.

    A fresh pool cannot draw on blocks the default pool is holding free, so whatever the caller's
    setup churned through would stay stranded for as long as the scratch pool lives -- several GB
    on a full head mesh.
    """
    cp.get_default_memory_pool().free_all_blocks()
    return cp.cuda.MemoryPool()


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
    nodal: bool = False


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
        self._host_tets: npt.NDArray[np.int32] | None = None
        self._host_slots: dict[
            Recovery, tuple[npt.NDArray[np.int32], npt.NDArray[np.int32] | None]
        ] = {}
        self._region_slices: dict[metrics.Region, metrics.RegionSlice] = {}
        self._conductivity_uq_pre: dict[tuple[int, ...], ConductivityUQPrecompute] = {}
        # Guards every lazy field above: under the free-threaded build two threads sharing a
        # subject would otherwise each build a solver context, and free() could clear one out
        # from under a reader mid-sweep. Reentrant because _region_slice and
        # _host_metric_inputs both go through `context`.
        self._lock = threading.RLock()

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
        with self._lock:
            self._conductivity_uq_pre.clear()
            # The recovery operators hang off the context, so dropping it releases them too.
            self._ctx = None
            self._host_metrics = None
            self._host_tets = None
            self._host_slots.clear()
            self._region_slices.clear()
        cp.get_default_memory_pool().free_all_blocks()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.free()

    @property
    def mesh(self) -> HeadMesh:
        """The mesh in the order every result array is indexed by.

        :func:`~cunibs.fem.build_context` lays the loaded mesh out along a Morton curve, so this
        is a reordering of the file's nodes and tetrahedra rather than the arrays
        :func:`~cunibs.mesh.load_mesh` returned. Reading it builds the solver context.
        """
        return self.context.mesh

    @property
    def context(self) -> SolverContext:
        with self._lock:
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
        with self._lock:
            if tags not in self._conductivity_uq_pre:
                self._conductivity_uq_pre[tags] = build_conductivity_uq_precompute(ctx, tags)
            return self._conductivity_uq_pre[tags]

    def _provenance(self, coil: Coil) -> dict[str, Any]:
        """What produced this run: the environment, and the mesh, coil and tolerance it used.

        Hashes ``self._mesh``, the mesh as loaded. ``self.mesh`` is the Morton reordering
        ``build_context`` produced, which would force a context build and would digest arrays the
        input file does not contain; the identity worth recording is the file's.
        """
        return {
            **_environment(),
            "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "mesh_content_hash": self._mesh.content_hash,
            "mesh_n_nodes": self._mesh.n_nodes,
            "mesh_n_tets": self._mesh.n_tets,
            "coil_n_dipoles": coil.n_dipoles,
            "solver_tolerance": float(self.context.solver.tolerance),
        }

    def _region_slice(self, region: metrics.Region) -> metrics.RegionSlice:
        """The field-independent half of a summary, gathered once per region.

        Call it first outside any scratch allocator: filling the cache inside one would tie it to
        that pool's lifetime.
        """
        with self._lock:
            cached = self._region_slices.get(region)
            if cached is None:
                ctx = self.context
                cached = metrics.region_slice(
                    ctx.vols, ctx.tet_barycenters_mm, ctx.tet_tags, region
                )
                self._region_slices[region] = cached
            return cached

    @property
    def _host_metric_inputs(self) -> _HostMetricInputs:
        """Host ``(vols, tet_tags, barycenters_mm)``, built once and shared by every result."""
        with self._lock:
            if self._host_metrics is None:
                ctx = self.context
                self._host_metrics = (
                    cp.asnumpy(ctx.vols),
                    cp.asnumpy(ctx.tet_tags),
                    cp.asnumpy(ctx.tet_barycenters_mm),
                )
            return self._host_metrics

    def _recovery_operator(self, recovery: Recovery) -> RecoveryOperator | None:
        """Build (and cache on the context) the patch weights for ``recovery``.

        Held here rather than reached from inside a solve so the build is serialised against
        other threads sharing this subject, and so it lands outside the caller's scratch
        allocator -- the operator outlives the chunk that first asked for it.
        """
        if recovery == "raw":
            return None
        with self._lock:
            return ensure_recovery(self.context, recovery)

    def _reduce_chunk(
        self,
        chunk: Sequence[Placement],
        outs: Sequence[PlacementResult],
        coil: Coil,
        didt: float,
        retain: _Retain,
        region: metrics.RegionSlice,
        recovery: Recovery,
        op: RecoveryOperator | None,
        rois: Sequence[_RoiProbe] = (),
        provenance: dict[str, Any] | None = None,
    ) -> list[FieldResult]:
        """Turn one solved chunk into results, summarising on the device.

        Runs inside the scratch pool, so nothing it returns may be device-backed.
        ``RegionSlice.summarize`` satisfies that: it returns Python floats and host arrays even
        when every input is on the device.
        """
        # nodal_field resolves its region against tet_tags, so a nodal result needs the host
        # metric inputs even without magnitude.
        host = self._host_metric_inputs if (retain.magnitude or retain.nodal) else None
        slot_node, slot_tag = self._host_slot_maps(op) if retain.nodal else (None, None)
        # Only per-node slots need the connectivity to say which nodes a region touches; a
        # (node, tissue) slot carries its own tag.
        tet_nodes = self._host_tet_nodes if retain.nodal and slot_tag is None else None
        n_nodes = int(self.context.n_nodes) if retain.nodal else None
        return [
            FieldResult(
                summary=region.summarize(out["magnE"]),
                roi_means=_roi_means(out["magnE"], rois),
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
                recovery=recovery,
                iterations=int(out["iterations"]),
                relative_residual=float(out["relative_residual"]),
                provenance=provenance,
                n_nodes=n_nodes,
                E_slots=cp.asnumpy(out["E_slots"]) if retain.nodal else None,
                slot_node=slot_node,
                slot_tag=slot_tag,
                tet_nodes=tet_nodes,
            )
            for site, out in zip(chunk, outs, strict=False)
        ]

    @property
    def _host_tet_nodes(self) -> npt.NDArray[np.int32]:
        """Host connectivity, built once and shared by every result carrying a nodal field."""
        with self._lock:
            if self._host_tets is None:
                self._host_tets = np.asarray(self.context.mesh.tet_nodes)
            return self._host_tets

    def _host_slot_maps(
        self, op: RecoveryOperator | None
    ) -> tuple[npt.NDArray[np.int32] | None, npt.NDArray[np.int32] | None]:
        """Host ``(slot_node, slot_tag)`` for ``op``, shared by every result it produces.

        Cached like :attr:`_host_tet_nodes`: a sweep would otherwise copy an n_slots-length
        array off the device once per placement.
        """
        if op is None:
            return None, None
        with self._lock:
            cached = self._host_slots.get(op.mode)
            if cached is None:
                cached = (
                    cp.asnumpy(op.slot_node),
                    None if op.slot_tag is None else cp.asnumpy(op.slot_tag),
                )
                self._host_slots[op.mode] = cached
            return cached

    def _iter_reduced(
        self,
        coil: Coil,
        sites: Sequence[Placement],
        didt: float,
        block_k: int | None,
        retain: _Retain,
        recovery: Recovery,
        record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
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
        region = self._region_slice(_DEFAULT_REGION)
        # Same reason: the patch weights outlive the chunk that first needs them.
        op = self._recovery_operator(recovery)
        rois = _roi_probes(record_rois)
        provenance = self._provenance(coil)
        # Placements share the stiffness matrix, so chunks of up to MAX_BLOCK solve as
        # one lockstep block CG that reads the matrix/hierarchy once for the whole chunk.
        chunk_k = MAX_BLOCK if block_k is None else max(1, min(MAX_BLOCK, block_k))
        temp_pool = _scratch_pool()
        try:
            for start in range(0, len(sites), chunk_k):
                chunk = sites[start : start + chunk_k]
                site_args = [(s.center_mm, s.handle_mm, s.distance_mm) for s in chunk]
                with cp.cuda.using_allocator(temp_pool.malloc):
                    outs = solve_placements_block(
                        ctx,
                        dip_pos_m,
                        dip_moment,
                        site_args,
                        didt,
                        recovery=op,
                        nodal=retain.nodal,
                    )
                    batch = self._reduce_chunk(
                        chunk, outs, coil, didt, retain, region, recovery, op, rois, provenance
                    )
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
        nodal: bool = False,
        recovery: Recovery = "harmonic",
        record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
        block_k: int | None = None,
    ) -> Iterator[FieldResult]:
        """Stream one :class:`FieldResult` per placement, in the order given.

        ``didt`` is the coil current's rate of change in A/s; the field is linear in it.

        Every result carries its gray-matter metrics; the flags say which of the full-volume
        arrays to keep as well. All three default to off because they are what makes a result
        large -- tens of megabytes on a head mesh, against about a kilobyte without them.

        Retaining ``magnitude`` also unlocks the metrics a precomputed summary cannot
        answer: :meth:`FieldResult.summary_for` for a non-default region, and
        :meth:`FieldResult.focality` at an arbitrary fraction.

        ``recovery`` selects how the reported field is derived from the solve, defaulting to
        ``"harmonic"``, the only mode that converges at a tissue interface. ``E``, ``magnE`` and
        ``summary`` all describe whichever field it selected; the raw field is not additionally
        retained. See :data:`~cunibs.Recovery` for the modes and :mod:`cunibs.fem.recovery` for
        the derivations.

        ``nodal`` keeps the recovered field on its slots as well, which is what interpolating
        to a surface needs; read it back per tissue with :meth:`FieldResult.nodal_field`. It
        requires a recovery mode.

        ``record_rois``, a ``{name: ResolvedTarget}`` mapping of ROIs from :meth:`roi` /
        :meth:`depth_probes` / ``resolve_target``, adds each ROI's volume-weighted mean ``|E|``
        to ``result.roi_means``. The reduction runs on the device against the field the solve
        already holds, so a scalar per ROI needs no retained array. Same definition as the UQ
        path's ``record_rois``.

        Placements are solved in blocks of ``block_k`` that share a single stiffness and
        hierarchy read. It defaults to ``MAX_BLOCK`` and is clamped to ``[1, MAX_BLOCK]``.
        A block is solved and reduced as a unit, so it also caps peak memory: only
        ``block_k`` results are live at once, however long the sweep.

        See :meth:`iter_simulate_conductivity_uq` to stream a conductivity Monte Carlo
        instead.
        """
        # The loop lives in a separate generator so this stays a plain function: a `yield`
        # here would defer the checks below to the caller's first next(), surfacing the
        # error at the consumption site instead of where the bad argument was passed.
        sites = _sites(placements, "iter_simulate")
        recovery = validate_recovery(recovery)
        if nodal and recovery == "raw":
            raise ValueError(
                "nodal=True needs a recovery mode: the raw per-tetrahedron field has no nodal "
                "form. Drop recovery='raw' to take the default, or name another mode from "
                "cunibs.RECOVERY_MODES."
            )
        retain = _Retain(magnitude, vectors, potential, nodal)
        return self._iter_reduced(coil, sites, didt, block_k, retain, recovery, record_rois)

    def simulate(
        self,
        coil: Coil,
        placement: Placement,
        didt: float = 1e6,
        *,
        magnitude: bool = False,
        vectors: bool = False,
        potential: bool = False,
        nodal: bool = False,
        recovery: Recovery = "harmonic",
        record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
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
            nodal=nodal,
            recovery=recovery,
            record_rois=record_rois,
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
        recovery: Recovery,
    ) -> Iterator[ConductivityUQResult]:
        """One Monte Carlo per placement, reduced inside a scratch pool and yielded outside.

        Unlike the deterministic path there is no block solve, so the chunk is a single
        placement; the pool scoping is otherwise identical.
        """
        from cunibs.uq import run_conductivity_uq

        ctx = self.context
        pre = self._conductivity_uq_precompute(config)
        # Outside the pool below, for the same reason the precompute is: the patch weights are
        # cached on the context and outlive the placement that first asked for them.
        op = self._recovery_operator(recovery)
        provenance = self._provenance(coil)
        temp_pool = _scratch_pool()
        try:
            for site in sites:
                with cp.cuda.using_allocator(temp_pool.malloc):
                    result = run_conductivity_uq(
                        ctx, pre, coil, site, config, didt, record_rois, recovery=op
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
                        provenance=provenance,
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
        recovery: Recovery = "harmonic",
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

        ``recovery`` takes the same values as :meth:`iter_simulate` and is applied to every
        draw before any statistic is taken, so the moment arrays and the per-draw scalars all
        describe the same field. There is no ``nodal`` flag: a result holds moments over draws
        rather than a field. Use :meth:`simulate` for a nodal field.
        """
        # The loop lives in a separate generator so this stays a plain function: a `yield`
        # here would defer the check below to the caller's first next(), surfacing the error
        # at the consumption site instead of where the bad argument was passed.
        sites = _sites(placements, "iter_simulate_conductivity_uq")
        recovery = validate_recovery(recovery)
        return self._iter_uq_reduced(coil, sites, config, didt, moments, record_rois, recovery)

    def simulate_conductivity_uq(
        self,
        coil: Coil,
        placement: Placement,
        config: ConductivityUQConfig,
        didt: float = 1e6,
        *,
        moments: bool = False,
        record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
        recovery: Recovery = "harmonic",
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
            coil,
            [placement],
            config,
            didt,
            moments=moments,
            record_rois=record_rois,
            recovery=recovery,
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

    ``recovery`` records which post-processing produced ``E``/``magnE``/``summary``. It is
    provenance, not a switch: a result carries one field, not several.

    ``iterations`` and ``relative_residual`` are this placement's own linear solve, not its
    block's: in a k-RHS solve every column freezes at its own residual.

    ``provenance`` records what produced the result, so a saved file can be tied back to its
    inputs: cuNIBS and dependency versions, the GPU, and the identity of the mesh and coil. It is
    ``None`` on a result loaded from a file that carries none.

    ``roi_means`` is the volume-weighted mean ``|E|`` over each ROI the run named in
    ``record_rois``, and is ``None`` when it named none. It describes the same recovered field as
    ``summary``.

    ``E_slots`` and the arrays that map it back to the mesh are present only when the run
    passed ``nodal=True``. A *slot* is whatever the recovery fitted a patch around: a plain
    node for ``"spr_global"``, a (node, tissue) pair for the tissue-restricted modes, where the
    field is genuinely two-valued at a boundary. ``slot_node`` gives each slot's node and
    ``slot_tag`` its tissue, or ``None`` when slots are nodes. Prefer :meth:`nodal_field`,
    which resolves all of that.
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
    recovery: Recovery = "raw"
    roi_means: dict[str, float] | None = None
    iterations: int | None = None
    relative_residual: float | None = None
    provenance: dict[str, Any] | None = None
    n_nodes: int | None = None
    E_slots: npt.NDArray[np.float32] | None = None
    slot_node: npt.NDArray[np.int32] | None = None
    slot_tag: npt.NDArray[np.int32] | None = None
    tet_nodes: npt.NDArray[np.int32] | None = None
    _summaries: dict[str, metrics.FieldMetrics] = field(default_factory=dict, repr=False)

    def nodal_field(self, region: metrics.Region = _DEFAULT_REGION) -> npt.NDArray[np.float32]:
        """The recovered field at the mesh nodes, on ``region``'s side of any boundary.

        Returns ``(n_nodes, 3)`` with NaN at every node ``region`` does not reach, which is the
        form an interpolation onto a cortical surface consumes.

        For the tissue-restricted modes the recovered field lives on (node, tissue) slots
        because ``E`` really is two-valued at a conductivity jump, and this picks ``region``'s
        value. ``region="all"`` is rejected for those: a boundary node has no single value to
        report, and silently choosing one of the two would be a quiet wrong answer.
        """
        if self.E_slots is None or self.slot_node is None or self.n_nodes is None:
            raise ValueError(
                "nodal_field needs the recovered nodal field, which this result did not "
                "retain. Re-run with nodal=True."
            )
        if self.slot_tag is not None and region == "all":
            raise ValueError(
                f"nodal_field(region='all') is ambiguous for recovery={self.recovery!r}: its "
                "field is two-valued at a tissue boundary, so a node there has one value per "
                "tissue. Name a tissue label instead."
            )
        keep: npt.NDArray[np.bool_] | slice
        if self.slot_tag is not None:
            keep = self.slot_tag == metrics.region_tag(region)
        elif region == "all":
            keep = slice(None)
        else:
            if self.tet_nodes is None or self.tet_tags is None:
                raise ValueError(
                    "nodal_field for a specific region needs the mesh connectivity, which this "
                    "result did not retain. Re-run with nodal=True and magnitude=True."
                )
            touched = np.zeros(self.n_nodes, dtype=bool)
            touched[self.tet_nodes[metrics.region_mask(self.tet_tags, region)]] = True
            keep = touched[self.slot_node]
        out = np.full((self.n_nodes, 3), np.nan, dtype=np.float32)
        out[self.slot_node[keep]] = self.E_slots[keep]
        return out

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
            for name in (
                "magnE",
                "E",
                "v",
                "vols",
                "tet_tags",
                "barycenters_mm",
                "E_slots",
                "slot_node",
                "slot_tag",
                "tet_nodes",
            ):
                arr = getattr(self, name)
                if arr is None:
                    continue
                h5f.create_dataset(name, data=arr, compression="gzip")
            h5f.create_dataset("transform", data=self.transform)
            _write_metrics(h5f.create_group("summary"), self.summary)
            h5f.attrs["format_version"] = _FORMAT_VERSION
            h5f.attrs["recovery"] = self.recovery
            if self.n_nodes is not None:
                h5f.attrs["n_nodes"] = self.n_nodes
            h5f.attrs["coil_name"] = self.coil_name
            h5f.attrs["didt"] = self.didt
            h5f.attrs["placement_center_mm"] = self.placement.center_mm
            h5f.attrs["placement_handle_mm"] = self.placement.handle_mm
            h5f.attrs["placement_distance_mm"] = self.placement.distance_mm
            if self.iterations is not None:
                h5f.attrs["iterations"] = self.iterations
            if self.relative_residual is not None:
                h5f.attrs["relative_residual"] = self.relative_residual
            if self.provenance is not None:
                _write_provenance(h5f, self.provenance)
            if self.roi_means is not None:
                grp = h5f.create_group("roi_means")
                for name, value in self.roi_means.items():
                    grp.attrs[name] = float(value)

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
                recovery=cast("Recovery", str(h5f.attrs.get("recovery", "raw"))),
                iterations=int(h5f.attrs["iterations"]) if "iterations" in h5f.attrs else None,
                relative_residual=(
                    float(h5f.attrs["relative_residual"])
                    if "relative_residual" in h5f.attrs
                    else None
                ),
                provenance=_read_provenance(h5f),
                roi_means=(
                    {k: float(v) for k, v in h5f["roi_means"].attrs.items()}
                    if "roi_means" in h5f
                    else None
                ),
                n_nodes=int(h5f.attrs["n_nodes"]) if "n_nodes" in h5f.attrs else None,
                E_slots=_opt_array(h5f, "E_slots"),
                slot_node=_opt_array(h5f, "slot_node"),
                slot_tag=_opt_array(h5f, "slot_tag"),
                tet_nodes=_opt_array(h5f, "tet_nodes"),
            )
