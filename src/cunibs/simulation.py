"""Run TMS simulations and store their results."""

from __future__ import annotations

import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cupy as cp
import h5py
import numpy as np
import numpy.typing as npt

from cunibs import metrics
from cunibs._common import (
    DEFAULT_REGION,
    NO_ROIS,
    as_point,
    check_format_version,
    opt_array,
    read_metrics,
    read_provenance,
    write_metrics,
    write_provenance,
)
from cunibs._transfer import AsyncCopier, Fence
from cunibs.fem import (
    MAX_BLOCK,
    PlacementResult,
    Recovery,
    RecoveryOperator,
    SolverContext,
    build_context,
    compute_coil_transforms,
    ensure_recovery,
    ensure_sweep_basis,
    solve_placements_block,
    validate_recovery,
)
from cunibs.mesh import HeadMesh, load_mesh
from cunibs.solver import clear_cuda_error

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
    from typing import Self

    from cunibs.adm.target import ResolvedTarget
    from cunibs.coil import Coil
    from cunibs.uq import (
        ConductivityUQConfig,
        ConductivityUQPrecompute,
        ConductivityUQResult,
    )

_FORMAT_VERSION = 1

# The widest chunk a sweep starts at. Only _block_width changes it.
_widest_block = MAX_BLOCK


@contextmanager
def _block_width(k: int) -> Iterator[None]:
    """Cap the chunk width of sweeps started inside the block, for checks that it moves nothing.

    Process-wide and not thread-safe; it exists for tests and benchmarks, not for callers.
    """
    global _widest_block
    if not 1 <= k <= MAX_BLOCK:
        raise ValueError(f"block width must be in [1, {MAX_BLOCK}], got {k}")
    previous, _widest_block = _widest_block, k
    try:
        yield
    finally:
        _widest_block = previous


# Per-tetrahedron metric inputs on the host, shared by reference across every FieldResult.
type _HostMetricInputs = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float64]
]


def _published[T](value: T) -> T:
    """``value``, once the stream that built it has finished.

    The subject caches device state built on whichever stream first asked for it, and a later
    call may read it from another; waiting here orders every stream after the build.
    """
    cp.cuda.get_current_stream().synchronize()
    return value


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


def _roi_means(
    fields: Sequence[cp.ndarray], record_rois: Mapping[str, ResolvedTarget]
) -> list[dict[str, float] | None]:
    """Each field's volume-weighted mean ``|E|`` per ROI, reduced on the device and copied back
    together."""
    if not record_rois:
        return [None] * len(fields)
    means = cp.asnumpy(
        cp.stack(
            [
                metrics.roi_mean(magn, roi.elem_idx, roi.weights)
                for magn in fields
                for roi in record_rois.values()
            ]
        )
    ).reshape(len(fields), len(record_rois))
    return [dict(zip(record_rois, row.tolist(), strict=True)) for row in means]


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
        center = as_point(center_mm)
        handle = as_point(handle_mm)
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
        self._resident: cp.cuda.MemoryPool | None = None
        self._scratch: cp.cuda.MemoryPool | None = None
        self._copier: AsyncCopier | None = None
        # Guards every lazy field above: under the free-threaded build two threads sharing a
        # subject would otherwise each build a solver context, and free() could clear one out
        # from under a reader mid-sweep. Reentrant because _region_slice and
        # _host_metric_inputs both go through `context`.
        self._lock = threading.RLock()

    @classmethod
    def from_mesh(cls, mesh_file: str | Path) -> Subject:
        return cls(load_mesh(mesh_file))

    def free(self) -> None:
        """Release cached GPU state: solver context, AMG hierarchies, UQ precompute and scratch.

        The subject keeps its device memory in pools of its own, which this returns to the device,
        so a loop over many subjects can reclaim memory as it goes. Arrays a caller still holds
        from the cached state, such as ``subject.context``'s, stay valid and are released when
        dropped. The subject stays usable afterwards; cached state is rebuilt lazily on the next
        call. Also available via the context manager:
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
            if self._copier is not None:
                self._copier.drain()
                self._copier = None
            for pool in (self._scratch, self._resident):
                if pool is not None:
                    pool.free_all_blocks()
            self._scratch = None
            self._resident = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.free()

    def _async_copier(self) -> AsyncCopier:
        """The pinned ring a multi-chunk call copies retained arrays through, kept until free()."""
        with self._lock:
            if self._copier is None:
                self._copier = AsyncCopier()
            return self._copier

    def _scratch_pool(self) -> cp.cuda.MemoryPool:
        """The pool a call's temporaries come from, kept until :meth:`free`.

        Keeping it lets repeated calls reuse its blocks rather than allocate and release them each
        time.
        """
        with self._lock:
            if self._scratch is None:
                self._scratch = cp.cuda.MemoryPool()
            return self._scratch

    def _release_free_blocks(self) -> None:
        """Return every free block a pool holds to the device, after an allocation failed.

        Each pool only reclaims its own blocks when it runs out, so memory cached by one is out of
        reach of the others until handed back. The default pool is the caller's, but its free
        blocks are unused memory, and a failed allocation is the moment they are needed.
        """
        # A failed allocation leaves the thread's last CUDA error set, and the next launch check
        # would report it as its own.
        clear_cuda_error()
        for pool in (self._scratch, self._resident, cp.get_default_memory_pool()):
            if pool is not None:
                pool.free_all_blocks()

    def _build[T](self, build: Callable[[], T]) -> T:
        """Run a build of cached state, allocating what it keeps from the subject's resident pool.

        Its temporaries come from the build's own arena (see :mod:`cunibs._memory`), so the
        resident pool holds nothing but cached state. A build that runs out of memory is retried
        once after :meth:`_release_free_blocks`. Callers hold the subject's lock.
        """
        if self._resident is None:
            self._resident = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(self._resident.malloc):
            try:
                return _published(build())
            except MemoryError:
                self._release_free_blocks()
                return _published(build())

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
                self._ctx = self._build(lambda: build_context(self._mesh))
            return self._ctx

    def roi(
        self,
        position_mm: npt.ArrayLike,
        radius_mm: float = 0.0,
        region: metrics.Region = "gray_matter",
    ) -> ResolvedTarget:
        """Volume-weighted ROI of ``region`` elements around ``position_mm`` (nearest if radius 0).

        Returns a :class:`~cunibs.adm.target.ResolvedTarget` usable as a ``record_rois`` probe or as
        the target of :func:`cunibs.adm.build_reciprocity` and :func:`cunibs.adm.optimize`.
        """
        from cunibs.adm.target import Target, resolve_target

        return resolve_target(
            self.context, Target(position_mm, region=region, radius_mm=radius_mm)
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
        p = as_point(cortical_point_mm)
        d = as_point(inward_dir)
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

        Cached by the set of perturbed tissues (see :meth:`ConductivityUQConfig.tags_for`) so
        repeated UQ runs on the same subject reuse the components and the nominal-σ AMG hierarchy.
        """
        from cunibs.uq import build_conductivity_uq_precompute

        ctx = self.context
        with cp.cuda.using_allocator(self._scratch_pool().malloc):
            tags = config.tags_for(cp.asnumpy(cp.unique(ctx.tet_tags)))
        with self._lock:
            if tags not in self._conductivity_uq_pre:
                self._conductivity_uq_pre[tags] = self._build(
                    lambda: build_conductivity_uq_precompute(ctx, tags)
                )
            return self._conductivity_uq_pre[tags]

    def _provenance(self, coil: Coil, *, seeded: bool = True) -> dict[str, Any]:
        """What produced this run: the environment, and the mesh, coil and tolerance it used.

        Hashes ``self._mesh``, the mesh as loaded. ``self.mesh`` is the Morton reordering
        ``build_context`` produced, which would force a context build and would digest arrays the
        input file does not contain; the identity worth recording is the file's.

        A sweep basis is recorded for the runs it seeds (``seeded``), since their solves land at a
        different point within the tolerance.
        """
        basis = self.context.sweep_basis if seeded else None
        return {
            **_environment(),
            "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "mesh_content_hash": self._mesh.content_hash,
            "mesh_n_nodes": self._mesh.n_nodes,
            "mesh_n_tets": self._mesh.n_tets,
            "coil_n_dipoles": coil.n_dipoles,
            "solver_tolerance": float(self.context.solver.tolerance),
            "sweep_basis_coil": None if basis is None else basis.coil_name,
            "sweep_basis_size": None if basis is None else basis.requested_size,
        }

    def _region_slice(self, region: metrics.Region) -> metrics.RegionSlice:
        """The field-independent half of a summary, gathered once per region."""
        with self._lock:
            cached = self._region_slices.get(region)
            if cached is None:
                ctx = self.context
                cached = self._build(
                    lambda: metrics.region_slice(
                        ctx.vols, ctx.tet_barycenters_mm, ctx.tet_tags, region
                    )
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

    def _recovery_operator(
        self, recovery: Recovery, tissues: Collection[int] | None = None
    ) -> RecoveryOperator | None:
        """Build (and cache on the context) the patch weights for ``recovery``.

        Held here rather than reached from inside a solve so the build is serialised against
        other threads sharing this subject, and so it lands in the resident pool, since the
        operator outlives the chunk that first asked for it. ``tissues`` is what the call reads;
        see :meth:`_needed_tissues`.
        """
        if recovery == "raw":
            return None
        with self._lock:
            ctx = self.context
            return self._build(lambda: ensure_recovery(ctx, recovery, tissues))

    def _needed_tissues(
        self, full_volume: bool, record_rois: Mapping[str, ResolvedTarget]
    ) -> frozenset[int] | None:
        """The tissues a call reads the recovered field on, or ``None`` for all of them.

        A call that keeps no full-volume array reads only its summary region and its ROIs, so the
        recovery can skip the rest of the head.
        """
        if full_volume:
            return None
        tags = {metrics.region_tag(DEFAULT_REGION)}
        tet_tags = self.context.tet_tags
        for target in record_rois.values():
            tags.update(int(t) for t in cp.asnumpy(cp.unique(tet_tags[target.elem_idx])))
        return frozenset(tags)

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
        record_rois: Mapping[str, ResolvedTarget] = NO_ROIS,
        provenance: dict[str, Any] | None = None,
        copier: AsyncCopier | None = None,
    ) -> list[FieldResult]:
        """Turn one solved chunk into results, summarising on the device.

        Runs inside the scratch pool, so nothing it returns may be device-backed.
        ``RegionSlice.summarize`` satisfies that: it returns Python floats and host arrays even
        when every input is on the device.

        With a ``copier`` the retained arrays are queued behind the chunk's GPU work rather than
        copied now, and are complete only once ``copier.fence()`` passes.
        """
        if copier is None:
            to_host = cp.asnumpy
        else:
            produced = cp.cuda.Event(disable_timing=True)
            produced.record()

            def to_host(a: cp.ndarray) -> npt.NDArray[Any]:
                return copier.copy(a, produced)

        # nodal_field resolves its region against tet_tags, so a nodal result needs the host
        # metric inputs even without magnitude.
        host = self._host_metric_inputs if (retain.magnitude or retain.nodal) else None
        slot_node, slot_tag = self._host_slot_maps(op) if retain.nodal else (None, None)
        # Only per-node slots need the connectivity to say which nodes a region touches; a
        # (node, tissue) slot carries its own tag.
        tet_nodes = self._host_tet_nodes if retain.nodal and slot_tag is None else None
        n_nodes = int(self.context.n_nodes) if retain.nodal else None
        fields = [out["magnE"] for out in outs]
        summaries = region.summarize_block(fields)
        roi_means = _roi_means(fields, record_rois)
        return [
            FieldResult(
                summary=summary,
                roi_means=rois,
                magnE=to_host(out["magnE"]) if retain.magnitude else None,
                E=to_host(out["E"]) if retain.vectors else None,
                v=to_host(out["v"]) if retain.potential else None,
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
                E_slots=to_host(out["E_slots"]) if retain.nodal else None,
                slot_node=slot_node,
                slot_tag=slot_tag,
                tet_nodes=tet_nodes,
            )
            for site, out, summary, rois in zip(chunk, outs, summaries, roi_means, strict=True)
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
        retain: _Retain,
        recovery: Recovery,
        record_rois: Mapping[str, ResolvedTarget] = NO_ROIS,
    ) -> Iterator[FieldResult]:
        """Solve ``sites`` in chunks, reducing inside a scratch pool and yielding outside it.

        The reduction runs under the subject's scratch allocator, whose blocks every chunk and
        every later call reuse, but the yield happens after the allocator is popped: control
        passes to caller code between chunks, and its allocations must not land in our pool.
        """
        if not sites:
            return
        ctx = self.context
        region = self._region_slice(DEFAULT_REGION)
        temp_pool = self._scratch_pool()
        with cp.cuda.using_allocator(temp_pool.malloc):
            dip_pos_m = cp.asarray(coil.positions_m)
            dip_moment = cp.asarray(coil.moments)
            tissues = self._needed_tissues(
                retain.magnitude or retain.vectors or retain.nodal, record_rois
            )
            # Every site's frame in one launch, which also rejects a degenerate handle before any
            # result is yielded.
            transforms = cp.asnumpy(
                compute_coil_transforms(
                    ctx,
                    np.stack([s.center_mm for s in sites]),
                    np.stack([s.handle_mm for s in sites]),
                    np.array([s.distance_mm for s in sites]),
                )
            )
        op = self._recovery_operator(recovery, tissues)
        provenance = self._provenance(coil)
        # Placements share the stiffness matrix, so a chunk solves as one lockstep block CG that
        # reads the matrix and hierarchy once for all of it. A chunk starts as wide as the block
        # kernels go and halves when the device runs out of memory for it, which never moves a
        # result: every placement's arithmetic is the same at any width.
        width = _widest_block
        # An allocation that failed before this call, in CuPy or the caller's own code, leaves the
        # thread's last CUDA error set, which the first launch check here would report as its own.
        clear_cuda_error()
        # With arrays to keep and more than one chunk, a chunk's copies to the host run while the
        # next chunk computes, so its results are yielded one chunk late. The pending chunk's
        # device outputs are held until its copies land.
        keeps = retain.magnitude or retain.vectors or retain.potential or retain.nodal
        copier = self._async_copier() if keeps and len(sites) > width else None
        pending: tuple[list[FieldResult], Fence] | None = None
        start = 0
        try:
            while start < len(sites):
                chunk = sites[start : start + width]
                site_args = [(s.center_mm, s.handle_mm, s.distance_mm) for s in chunk]
                try:
                    with cp.cuda.using_allocator(temp_pool.malloc):
                        outs = solve_placements_block(
                            ctx,
                            dip_pos_m,
                            dip_moment,
                            site_args,
                            didt,
                            recovery=op,
                            nodal=retain.nodal,
                            tissues=tissues,
                            transforms=transforms[start : start + len(chunk)],
                            vectors=retain.vectors,
                            potential=retain.potential,
                        )
                        batch = self._reduce_chunk(
                            chunk,
                            outs,
                            coil,
                            didt,
                            retain,
                            region,
                            recovery,
                            op,
                            record_rois,
                            provenance,
                            copier,
                        )
                        del outs
                except MemoryError:
                    if width == 1:
                        raise
                    width //= 2
                    if copier is not None:
                        copier.drain()
                    self._release_free_blocks()
                    continue
                start += len(chunk)
                if copier is None:
                    yield from batch
                    continue
                if pending is not None:
                    copier.settle(pending[1])
                    yield from pending[0]
                pending = (batch, copier.fence())
            if pending is not None and copier is not None:
                copier.settle(pending[1])
                yield from pending[0]
        finally:
            # An abandoned sweep still has copies writing into its arrays.
            if copier is not None:
                copier.drain()

    def prepare_sweep(self, coil: Coil, size: int = 128) -> None:
        """Seed every later solve on this subject from a basis of ``size`` scalp placements.

        Each solve then starts from the Galerkin solution over those placements' solutions
        rather than from zero, which on a head mesh cuts the CG iterations of a dense sweep about
        one target by about a third. Building the basis
        costs ``size`` solves (about 0.9 s) and ``size`` fp32 vectors of device memory (about
        360 MB at the default on a 700k-node mesh), so it pays from several hundred placements
        and is never built implicitly.

        A placement's result does not depend on the other placements in its sweep, with or
        without a basis. It does change, within the solver tolerance, once a basis is present,
        so prepare before the first solve of a sweep whose results will be compared. Results record
        the basis in their provenance. A second call for the same coil and size keeps the basis;
        any other raises until :meth:`free` discards it.
        """
        with self._lock:
            ctx = self.context
            self._build(lambda: ensure_sweep_basis(ctx, coil, size))

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
        record_rois: Mapping[str, ResolvedTarget] = NO_ROIS,
    ) -> Iterator[FieldResult]:
        """Stream one :class:`FieldResult` per placement, in the order given.

        ``didt`` is the coil current's rate of change in A/s; the field is linear in it.

        Every result carries its gray-matter metrics; the flags say which of the full-volume
        arrays to keep as well. All three default to off because they are what makes a result
        large: tens of megabytes on a head mesh, against about a kilobyte without them.

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

        Placements are solved in blocks that share a single stiffness and hierarchy read, as
        many at once as the device has memory for. No result depends on how they were grouped.

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
        return self._iter_reduced(coil, sites, didt, retain, recovery, record_rois)

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
        record_rois: Mapping[str, ResolvedTarget] = NO_ROIS,
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
        temp_pool = self._scratch_pool()
        with cp.cuda.using_allocator(temp_pool.malloc):
            tissues = self._needed_tissues(moments, record_rois)
        op = self._recovery_operator(recovery, tissues)
        gray_matter = self._region_slice(DEFAULT_REGION)
        provenance = self._provenance(coil, seeded=False)
        host = self._host_metric_inputs if moments else None

        def reduce(site: Placement) -> ConductivityUQResult:
            with cp.cuda.using_allocator(temp_pool.malloc):
                result = run_conductivity_uq(
                    ctx,
                    pre,
                    coil,
                    site,
                    config,
                    didt=didt,
                    record_rois=record_rois,
                    recovery=op,
                    tissues=tissues,
                    region=gray_matter,
                )
                # Summarised on the device before anything is copied back.
                return replace(
                    result,
                    summary=result.compute_summary(region_slice=gray_matter),
                    mean_magnE=cp.asnumpy(result.mean_magnE) if moments else None,
                    std_magnE=cp.asnumpy(result.std_magnE) if moments else None,
                    cov_magnE=cp.asnumpy(result.cov_magnE) if moments else None,
                    vols=host[0] if host else None,
                    tet_tags=host[1] if host else None,
                    barycenters_mm=host[2] if host else None,
                    provenance=provenance,
                )

        for site in sites:
            # A placement's draws do not depend on anything a failed attempt left behind, so a
            # rerun after the pools hand back their free blocks gives the same result.
            try:
                reduced = reduce(site)
            except MemoryError:
                self._release_free_blocks()
                reduced = reduce(site)
            yield reduced

    def iter_simulate_conductivity_uq(
        self,
        coil: Coil,
        placements: Sequence[Placement],
        config: ConductivityUQConfig,
        didt: float = 1e6,
        *,
        moments: bool = False,
        record_rois: Mapping[str, ResolvedTarget] = NO_ROIS,
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
        record_rois: Mapping[str, ResolvedTarget] = NO_ROIS,
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

    ``vols``, ``tet_tags`` and ``barycenters_mm`` accompany ``magnE``, since they are what the
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

    def nodal_field(self, region: metrics.Region = DEFAULT_REGION) -> npt.NDArray[np.float32]:
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

    def peak_magnE(self, region: metrics.Region = DEFAULT_REGION) -> float:
        return self.summary_for(region)["peak_magnE"]

    def peak_location_mm(
        self, region: metrics.Region = DEFAULT_REGION
    ) -> npt.NDArray[np.float64]:
        return self.summary_for(region)["peak_location_mm"]

    def center_of_gravity_mm(
        self, region: metrics.Region = DEFAULT_REGION
    ) -> npt.NDArray[np.float64]:
        return self.summary_for(region)["center_of_gravity_mm"]

    def focality(self, frac: float = 0.5, region: metrics.Region = DEFAULT_REGION) -> float:
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
            write_metrics(h5f.create_group("summary"), self.summary)
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
                write_provenance(h5f, self.provenance)
            if self.roi_means is not None:
                grp = h5f.create_group("roi_means")
                for name, value in self.roi_means.items():
                    grp.attrs[name] = float(value)

    @classmethod
    def load(cls, path: str | Path) -> FieldResult:
        """Read a saved result into NumPy arrays."""
        with h5py.File(Path(path), "r") as h5f:
            check_format_version(h5f, path, _FORMAT_VERSION)
            return cls(
                summary=read_metrics(h5f["summary"]),
                magnE=opt_array(h5f, "magnE"),
                E=opt_array(h5f, "E"),
                v=opt_array(h5f, "v"),
                transform=np.asarray(h5f["transform"]),
                placement=Placement(
                    center_mm=h5f.attrs["placement_center_mm"],
                    handle_mm=h5f.attrs["placement_handle_mm"],
                    distance_mm=float(h5f.attrs["placement_distance_mm"]),
                ),
                coil_name=str(h5f.attrs["coil_name"]),
                didt=float(h5f.attrs["didt"]),
                vols=opt_array(h5f, "vols"),
                tet_tags=opt_array(h5f, "tet_tags"),
                barycenters_mm=opt_array(h5f, "barycenters_mm"),
                recovery=cast("Recovery", str(h5f.attrs.get("recovery", "raw"))),
                iterations=int(h5f.attrs["iterations"]) if "iterations" in h5f.attrs else None,
                relative_residual=(
                    float(h5f.attrs["relative_residual"])
                    if "relative_residual" in h5f.attrs
                    else None
                ),
                provenance=read_provenance(h5f),
                roi_means=(
                    {k: float(v) for k, v in h5f["roi_means"].attrs.items()}
                    if "roi_means" in h5f
                    else None
                ),
                n_nodes=int(h5f.attrs["n_nodes"]) if "n_nodes" in h5f.attrs else None,
                E_slots=opt_array(h5f, "E_slots"),
                slot_node=opt_array(h5f, "slot_node"),
                slot_tag=opt_array(h5f, "slot_tag"),
                tet_nodes=opt_array(h5f, "tet_nodes"),
            )
