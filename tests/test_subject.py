"""Subject lifecycle: context caching, ``free()``, the context manager, and ROI selection.

``free()`` exists so a loop over many subjects can reclaim VRAM as it goes, and the
streaming API exists so a long placement sweep does not accumulate memory on either side.
Both are memory contracts that nothing else in the suite checks.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from cunibs import (
    RECOVERY_MODES,
    ConductivityUQConfig,
    ConductivityUQResult,
    FieldResult,
    HeadMesh,
    Placement,
    ResolvedTarget,
    Subject,
    resolve_target,
)
from cunibs._memory import build_arena, release_scratch, scratch_pool, unpooled
from cunibs.fem.solve import SolverConvergenceError
from cunibs.mesh import load_mesh

pytestmark = pytest.mark.gpu

PLACEMENT = Placement([50, 50, 100], [50, 100, 100])


@pytest.mark.realmesh
def test_from_mesh_reorders_the_loaded_mesh(patch_mesh_path, patch_mesh):
    """The subject's mesh is a spatial reordering of the file's, not a different mesh.

    Every comparison here is permutation invariant, which is the point: the node and element
    order moves, the geometry does not.
    """
    subj = Subject.from_mesh(patch_mesh_path)
    try:
        mesh = subj.mesh
        assert mesh.n_nodes == patch_mesh.n_nodes
        assert mesh.tet_nodes.shape == patch_mesh.tet_nodes.shape
        np.testing.assert_array_equal(np.sort(mesh.tet_tags), np.sort(patch_mesh.tet_tags))
        np.testing.assert_allclose(
            np.sort(mesh.nodes_mm, axis=0), np.sort(patch_mesh.nodes_mm, axis=0)
        )
        np.testing.assert_allclose(
            np.sort(mesh.tet_barycenters_mm, axis=0),
            np.sort(patch_mesh.tet_barycenters_mm, axis=0),
        )
    finally:
        subj.free()


def test_context_is_cached(cube_subject):
    assert cube_subject.context is cube_subject.context


def test_free_drops_and_rebuilds_context(fresh_subject, cube_mesh, synthetic_coil):
    subj = fresh_subject(cube_mesh)
    before = subj.simulate(synthetic_coil, PLACEMENT).peak_magnE()
    first_ctx = subj.context

    subj.free()
    assert subj._ctx is None

    second_ctx = subj.context
    assert second_ctx is not first_ctx
    after = subj.simulate(synthetic_coil, PLACEMENT).peak_magnE()
    assert after == pytest.approx(before, rel=1e-6)


def test_free_clears_uq_precompute(fresh_subject, two_tissue_cube_mesh, synthetic_coil):
    from cunibs import ConductivityUQConfig

    subj = fresh_subject(two_tissue_cube_mesh)
    subj.simulate_conductivity_uq(
        synthetic_coil, PLACEMENT, config=ConductivityUQConfig(n_samples=8, seed=0)
    )
    assert subj._conductivity_uq_pre

    subj.free()
    assert subj._conductivity_uq_pre == {}


def test_context_manager_frees_on_exit(cube_mesh, synthetic_coil):
    with Subject(cube_mesh) as subj:
        subj.simulate(synthetic_coil, PLACEMENT)
        assert subj._ctx is not None
    assert subj._ctx is None
    # Still usable afterwards: state is rebuilt lazily.
    assert subj.simulate(synthetic_coil, PLACEMENT).peak_magnE() > 0
    subj.free()


def test_mesh_property_is_the_context_mesh(cube_mesh):
    """Results index against the context's ordering, so that is what the property must expose."""
    subj = Subject(cube_mesh)
    try:
        assert subj.mesh is subj.context.mesh
    finally:
        subj.free()


def _held(cp) -> int:
    """Bytes in use across the pools a call's device arrays can come from."""
    return scratch_pool().used_bytes() + cp.get_default_memory_pool().used_bytes()


@pytest.mark.realmesh
def test_free_releases_device_memory(cp, patch_mesh):
    """The cached state stays out of the caller's pool, and free() lets go of all of it."""
    caller = cp.get_default_memory_pool()
    caller.free_all_blocks()
    baseline = caller.used_bytes()

    subj = Subject(patch_mesh)
    held = weakref.ref(subj.context)
    assert caller.used_bytes() - baseline < 1_000_000

    subj.free()
    gc.collect()
    assert held() is None
    assert scratch_pool().total_bytes() == 0
    assert caller.used_bytes() - baseline < 1_000_000


@pytest.mark.realmesh
def test_cached_arrays_do_not_share_a_pool_chunk(cp, patch_mesh):
    """An array sharing a chunk with a dropped pool's other blocks would strand that chunk."""
    subj = Subject(patch_mesh)
    assert type(subj.context.vols.data.mem) is cp.cuda.Memory
    assert type(subj.context.rhs_indptr.data.mem) is cp.cuda.Memory
    subj.free()


@pytest.mark.realmesh
def test_summary_sweep_does_not_grow_pool(
    cp, fresh_subject, patch_mesh, d70_coil, patch_placement
):
    """The documented scratch-pool contract: a long sweep must not accumulate device memory."""
    subj = fresh_subject(patch_mesh)

    subj.simulate(d70_coil, patch_placement)
    subj.simulate(d70_coil, patch_placement)
    settled = _held(cp)
    for _ in range(8):
        subj.simulate(d70_coil, patch_placement)
    assert _held(cp) - settled < 1_000_000


@pytest.mark.realmesh
def test_retained_fields_are_host_backed_and_leave_the_pool_flat(
    cp, fresh_subject, patch_mesh, d70_coil, patch_placement
):
    """The counterpart: retaining fields moves them to the host, off the scarcer resource."""
    subj = fresh_subject(patch_mesh)

    subj.simulate(d70_coil, patch_placement)  # warm the context and its caches
    settled = _held(cp)
    results = list(subj.iter_simulate(d70_coil, [patch_placement] * 4, magnitude=True))

    for r in results:
        assert isinstance(r.magnE, np.ndarray)
        assert r.magnE.shape == (patch_mesh.tet_nodes.shape[0],)
    # Retained results share the metric inputs rather than copying them per result.
    assert results[0].vols is results[1].vols
    # Holding every result on the host must not have grown the device pools.
    assert _held(cp) - settled < 1_000_000


def test_roi_radius_zero_is_nearest_gm_element(cp, cube_subject, cube_mesh):
    point = np.array([80.0, 40.0, 20.0])
    roi = cube_subject.roi(point)
    expected = int(np.argmin(np.linalg.norm(cube_mesh.tet_barycenters_mm - point, axis=1)))
    assert cp.asnumpy(roi.elem_idx).tolist() == [expected]
    np.testing.assert_array_equal(cp.asnumpy(roi.weights), [1.0])


def test_roi_radius_selects_ball_with_volume_weights(cp, cube_subject, cube_mesh):
    ctx = cube_subject.context
    point = np.array([50.0, 50.0, 50.0])
    roi = cube_subject.roi(point, radius_mm=40.0)

    idx = cp.asnumpy(roi.elem_idx)
    assert idx.size > 1
    assert np.linalg.norm(cube_mesh.tet_barycenters_mm[idx] - point, axis=1).max() <= 40.0
    vols = cp.asnumpy(ctx.vols).astype(np.float64)[idx]
    np.testing.assert_allclose(cp.asnumpy(roi.weights), vols / vols.sum(), rtol=1e-14)


@pytest.mark.realmesh
def test_roi_region_filter(cp, patch_subject):
    """On a multi-tissue mesh the region argument actually restricts the selection."""
    mesh = patch_subject.mesh
    csf = mesh.tet_barycenters_mm[mesh.tet_tags == 3]
    point = csf.mean(axis=0)

    roi = patch_subject.roi(point, radius_mm=6.0, region="csf")
    idx = cp.asnumpy(roi.elem_idx)
    np.testing.assert_array_equal(mesh.tet_tags[idx], 3)

    everything = patch_subject.roi(point, radius_mm=6.0, region="all")
    all_idx = cp.asnumpy(everything.elem_idx)
    assert all_idx.size > idx.size
    assert set(mesh.tet_tags[all_idx].tolist()) > {3}


def test_depth_probes_normalizes_the_direction(cp, cube_subject):
    """Only the direction of ``inward_dir`` matters; its magnitude must not scale the depths."""
    depths = [0.0, 10.0, 30.0]
    unit = cube_subject.depth_probes([50, 50, 100], [0, 0, -1], depths, radius_mm=40.0)
    scaled = cube_subject.depth_probes([50, 50, 100], [0, 0, -12.5], depths, radius_mm=40.0)
    for a, b in zip(unit, scaled, strict=False):
        np.testing.assert_array_equal(cp.asnumpy(a.elem_idx), cp.asnumpy(b.elem_idx))
        np.testing.assert_allclose(cp.asnumpy(a.barycenter_mm), cp.asnumpy(b.barycenter_mm))


def test_depth_probes_matches_roi_at_each_depth(cp, cube_subject):
    depths = [0.0, 15.0, 35.0]
    probes = cube_subject.depth_probes([50, 50, 100], [0, 0, -1], depths, radius_mm=40.0)
    for depth, probe in zip(depths, probes, strict=False):
        expected = cube_subject.roi([50, 50, 100 - depth], radius_mm=40.0, region="gray_matter")
        np.testing.assert_array_equal(cp.asnumpy(probe.elem_idx), cp.asnumpy(expected.elem_idx))


def test_depth_probes_rejects_bad_vectors(cube_subject):
    with pytest.raises(ValueError):
        cube_subject.depth_probes([50, 50], [0, 0, -1], [0.0])
    with pytest.raises(ValueError):
        cube_subject.depth_probes([50, 50, 100], [0, 0], [0.0])


@pytest.mark.parametrize("bad", [[0, 0, 0], [np.nan, 0, -1]])
def test_depth_probes_rejects_an_undefined_direction(cube_subject, bad):
    """Otherwise the probe points are NaN and every depth silently resolves to element 0."""
    with pytest.raises(ValueError, match="inward_dir must be a non-zero, finite 3-vector"):
        cube_subject.depth_probes([50, 50, 100], bad, [0.0, 5.0], radius_mm=40.0)


@pytest.mark.realmesh
def test_two_subjects_are_independent(fresh_subject, patch_mesh, cube_mesh, synthetic_coil):
    """Freeing one subject must not disturb another's cached state."""
    a = fresh_subject(cube_mesh)
    b = fresh_subject(patch_mesh)
    a_ctx, b_ctx = a.context, b.context
    assert a_ctx is not b_ctx

    a.free()
    assert b._ctx is b_ctx
    assert b.simulate(synthetic_coil, PLACEMENT) is not None


@pytest.mark.realmesh
def test_load_mesh_is_deterministic(patch_mesh_path):
    """Two loads of the same file must produce byte-identical arrays."""
    first, second = load_mesh(patch_mesh_path), load_mesh(patch_mesh_path)
    np.testing.assert_array_equal(first.nodes_mm, second.nodes_mm)
    np.testing.assert_array_equal(first.tet_nodes, second.tet_nodes)
    np.testing.assert_array_equal(first.tet_tags, second.tet_tags)
    np.testing.assert_array_equal(first.skin_tris, second.skin_tris)


def _cortex_under(subject, placement) -> np.ndarray:
    """The gray-matter barycentre nearest the placement's centre."""
    mesh = subject.mesh
    gm = np.asarray(mesh.tet_barycenters_mm)[np.asarray(mesh.tet_tags) == 2]
    return gm[np.argmin(np.linalg.norm(gm - placement.center_mm, axis=1))]


@pytest.mark.realmesh
def test_record_rois_matches_a_reduction_of_the_retained_field(
    cp, patch_subject, d70_coil, patch_placement
):
    """An ROI scalar must be the same number a retained array would have given.

    Recording ROIs on the forward path avoids keeping a per-tetrahedron array to extract a few
    numbers from; it does not approximate the reduction, so the two must agree exactly. A host
    sum associates differently, so it can only agree to rounding.
    """
    from cunibs import metrics

    cortex = _cortex_under(patch_subject, patch_placement)
    rois = {
        "near": patch_subject.roi(cortex, radius_mm=5.0, region="gray_matter"),
        "point": patch_subject.roi(cortex, region="gray_matter"),
    }
    result = patch_subject.simulate(d70_coil, patch_placement, magnitude=True, record_rois=rois)

    assert set(result.roi_means) == {"near", "point"}
    for name, target in rois.items():
        retained = metrics.roi_mean(cp.asarray(result.magnE), target.elem_idx, target.weights)
        assert result.roi_means[name] == float(retained)
        idx = cp.asnumpy(target.elem_idx)
        w = cp.asnumpy(target.weights)
        host = float((result.magnE[idx].astype(np.float64) * w).sum())
        assert result.roi_means[name] == pytest.approx(host, rel=1e-14)

    # Recording an ROI must not require retaining anything.
    lean = patch_subject.simulate(d70_coil, patch_placement, record_rois=rois)
    assert lean.magnE is None and lean.E is None
    assert lean.roi_means == result.roi_means


@pytest.mark.realmesh
def test_roi_means_round_trip_through_hdf5(patch_subject, d70_coil, patch_placement, tmp_path):
    from cunibs import FieldResult

    rois = {
        "m1": patch_subject.roi(_cortex_under(patch_subject, patch_placement), radius_mm=5.0)
    }
    result = patch_subject.simulate(d70_coil, patch_placement, record_rois=rois)
    path = tmp_path / "roi.h5"
    result.save(path)
    assert FieldResult.load(path).roi_means == result.roi_means

    # A run that named no ROI carries no group, and loads back as None rather than {}.
    plain = patch_subject.simulate(d70_coil, patch_placement)
    plain.save(path)
    assert plain.roi_means is None
    assert FieldResult.load(path).roi_means is None


def test_out_of_memory_is_a_memory_error_from_every_allocator(cp):
    """The retries catch MemoryError, and CUDA's allocators report through a runtime error."""
    huge = 1 << 46
    with build_arena(), pytest.raises(MemoryError):
        cp.empty(huge, dtype=cp.uint8)
    with pytest.raises(MemoryError):
        unpooled(huge)
    with pytest.raises(MemoryError), cp.cuda.using_allocator(scratch_pool().malloc):
        cp.empty(huge, dtype=cp.uint8)


def test_another_allocator_failure_is_raised_as_it_is(cp, monkeypatch):
    def fails(_size):
        raise cp.cuda.runtime.CUDARuntimeError(1)

    monkeypatch.setattr(cp.cuda, "Memory", fails)
    with pytest.raises(cp.cuda.runtime.CUDARuntimeError) as raised:
        unpooled(8)
    assert raised.value.__cause__ is None


@pytest.mark.realmesh
def test_a_field_outside_fp32_range_raises(
    fresh_subject, patch_mesh, d70_coil, patch_placement
):
    """fp32 CG breaks down with a NaN residual, which must not pass for convergence."""
    subj = fresh_subject(patch_mesh)
    with pytest.raises(SolverConvergenceError):
        subj.simulate(d70_coil, patch_placement, 1e-12)
    assert np.isfinite(subj.simulate(d70_coil, patch_placement, 1e6).peak_magnE())


@pytest.mark.realmesh
@pytest.mark.parametrize("didt", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_didt_is_rejected(
    fresh_subject, patch_mesh, d70_coil, patch_placement, didt
):
    subj = fresh_subject(patch_mesh)
    with pytest.raises(ValueError, match="didt must be finite"):
        subj.simulate(d70_coil, patch_placement, didt)
    with pytest.raises(ValueError, match="didt must be finite"):
        subj.simulate_conductivity_uq(
            d70_coil, patch_placement, ConductivityUQConfig(n_samples=2), didt
        )


@pytest.mark.realmesh
@pytest.mark.parametrize("element", [-5, 10**8])
def test_a_roi_outside_the_mesh_is_rejected(
    cp, fresh_subject, patch_mesh, d70_coil, patch_placement, element
):
    """Its elements index device arrays with no bounds check, and another subject's can overrun."""
    roi = ResolvedTarget(
        cp.asarray(np.array([element], dtype=np.int32)),
        cp.ones(1),
        cp.eye(3),
        True,
        cp.zeros(3),
    )
    subj = fresh_subject(patch_mesh)
    with pytest.raises(ValueError, match="outside this mesh"):
        subj.simulate(d70_coil, patch_placement, 1e6, record_rois={"a": roi})
    with pytest.raises(ValueError, match="outside this mesh"):
        resolve_target(subj, roi)


@pytest.fixture
def no_gray_matter(patch_mesh):
    """The patch with its gray matter relabelled, so the default summary region is absent."""
    tags = np.where(patch_mesh.tet_tags == 2, 1, patch_mesh.tet_tags)
    return HeadMesh(
        patch_mesh.nodes_mm,
        patch_mesh.tet_nodes,
        tags,
        patch_mesh.surf_tris,
        patch_mesh.surf_tags,
    )


@pytest.mark.realmesh
@pytest.mark.parametrize("recovery", RECOVERY_MODES)
def test_a_mesh_with_no_gray_matter_skips_the_summary(
    no_gray_matter, d70_coil, patch_placement, recovery
):
    with Subject(no_gray_matter) as subj:
        bare = subj.simulate(d70_coil, patch_placement, 1e6, recovery=recovery)
        assert bare.summary is None
        kept = subj.simulate(d70_coil, patch_placement, 1e6, recovery=recovery, magnitude=True)
        assert kept.summary is None
        assert np.isfinite(kept.magnE).all()
        assert kept.peak_magnE("all") == float(kept.magnE.max())
        with pytest.raises(ValueError, match="holds no elements"):
            kept.peak_magnE()
        swept = list(
            subj.iter_simulate(d70_coil, [patch_placement] * 9, 1e6, recovery=recovery)
        )
        assert all(r.summary is None for r in swept)


@pytest.mark.realmesh
def test_a_mesh_with_no_gray_matter_still_reads_rois_and_saves(
    tmp_path, no_gray_matter, d70_coil, patch_placement
):
    with Subject(no_gray_matter) as subj:
        roi = subj.roi(patch_placement.center_mm - [0.0, 0.0, 6.0], 4.0, "white_matter")
        result = subj.simulate(d70_coil, patch_placement, 1e6, record_rois={"a": roi})
        assert result.summary is None
        assert result.roi_means["a"] > 0.0
        result.save(tmp_path / "r.h5")
        loaded = FieldResult.load(tmp_path / "r.h5")
        assert loaded.summary is None
        assert loaded.roi_means["a"] == result.roi_means["a"]


@pytest.mark.realmesh
def test_uq_on_a_mesh_with_no_gray_matter_skips_its_scalars(
    tmp_path, no_gray_matter, d70_coil, patch_placement
):
    config = ConductivityUQConfig(n_samples=9, seed=0)
    with Subject(no_gray_matter) as subj:
        roi = subj.roi(patch_placement.center_mm - [0.0, 0.0, 6.0], 4.0, "white_matter")
        uq = subj.simulate_conductivity_uq(
            d70_coil, patch_placement, config, 1e6, moments=True, record_rois={"a": roi}
        )
    assert uq.summary is None
    assert uq.peak_samples is None
    assert np.isfinite(uq.mean_magnE).all()
    assert uq.roi_samples["a"].shape == (9,)
    assert set(uq.tissue_sensitivity("a")) == set(uq.perturbed_tags)
    with pytest.raises(ValueError, match="no gray matter"):
        uq.tissue_sensitivity("peak")
    uq.save(tmp_path / "u.h5")
    loaded = ConductivityUQResult.load(tmp_path / "u.h5")
    assert loaded.summary is None
    assert loaded.peak_samples is None
    np.testing.assert_array_equal(loaded.mean_magnE, uq.mean_magnE)


@pytest.mark.realmesh
def test_uq_skips_a_named_tissue_the_mesh_does_not_hold(
    fresh_subject, patch_mesh, d70_coil, patch_placement
):
    absent = next(t for t in (6, 7, 8, 9, 10) if t not in patch_mesh.tet_tags)
    subj = fresh_subject(patch_mesh)
    with_absent = ConductivityUQConfig(n_samples=4, seed=0, tissue_cov={2: 0.2, absent: 0.1})
    with pytest.warns(UserWarning, match="Skipping tissues"):
        skipped = subj.simulate_conductivity_uq(d70_coil, patch_placement, with_absent, 1e6)
    alone = subj.simulate_conductivity_uq(
        d70_coil,
        patch_placement,
        ConductivityUQConfig(n_samples=4, seed=0, tissue_cov={2: 0.2}),
        1e6,
    )
    assert skipped.perturbed_tags == (2,)
    np.testing.assert_array_equal(skipped.peak_samples, alone.peak_samples)


def _fails_once_holding(cp, size):
    """A build whose first attempt dies holding a temporary, and reports what the retry saw."""
    seen = []

    def build():
        if not seen:
            seen.append(None)
            with cp.cuda.using_allocator(scratch_pool().malloc):
                held = cp.empty(size, dtype=cp.uint8)
                raise MemoryError(held.size)
        seen.append(scratch_pool().used_bytes())
        return "built"

    return build, seen


def test_a_retried_build_does_not_hold_the_failed_attempts_memory(cp, cube_mesh):
    """The handler's traceback pins the failed frames, so the retry must run outside it."""
    release_scratch()
    build, seen = _fails_once_holding(cp, 1 << 24)
    with Subject(cube_mesh) as subj:
        assert subj._build(build) == "built"
    assert seen[1] == 0


@pytest.mark.parametrize(
    "status",
    [
        "CUBLAS_STATUS_ALLOC_FAILED",
        "CUSOLVER_STATUS_ALLOC_FAILED",
        "CUSOLVER_STATUS_INTERNAL_ERROR",
    ],
)
def test_a_library_out_of_memory_is_a_memory_error(status):
    """A handle's first allocation fails this way on a nearly full device."""
    with pytest.raises(MemoryError), build_arena():
        raise RuntimeError(status)


def test_other_library_errors_pass_through():
    with pytest.raises(RuntimeError, match="INVALID_VALUE"), build_arena():
        raise RuntimeError("CUBLAS_STATUS_INVALID_VALUE")


@pytest.mark.realmesh
def test_element_and_node_order_map_results_to_the_loaded_mesh(
    patch_mesh, d70_coil, patch_placement
):
    with Subject(patch_mesh) as subj:
        result = subj.simulate(d70_coil, patch_placement, 1e6, magnitude=True, potential=True)
        elements, nodes = subj.element_order, subj.node_order

        assert sorted(elements.tolist()) == list(range(patch_mesh.n_tets))
        assert sorted(nodes.tolist()) == list(range(patch_mesh.n_nodes))
        np.testing.assert_array_equal(patch_mesh.tet_tags[elements], result.tet_tags)
        np.testing.assert_array_equal(patch_mesh.nodes_mm[nodes], subj.mesh.nodes_mm)
        np.testing.assert_allclose(
            patch_mesh.nodes_mm[patch_mesh.tet_nodes].mean(axis=1)[elements],
            result.barycenters_mm,
            atol=1e-9,
        )

        in_file_order = np.empty_like(result.magnE)
        in_file_order[elements] = result.magnE
        np.testing.assert_array_equal(in_file_order[elements], result.magnE)
        gray = patch_mesh.tet_tags == 2
        assert in_file_order[gray].sum() == pytest.approx(
            result.magnE[result.tet_tags == 2].sum(), rel=1e-6
        )


@pytest.mark.realmesh
def test_order_does_not_depend_on_when_it_is_asked_for(patch_mesh):
    with Subject(patch_mesh) as subj:
        before = subj.element_order.copy(), subj.node_order.copy()
        assert subj.context is not None
        np.testing.assert_array_equal(subj.element_order, before[0])
        np.testing.assert_array_equal(subj.node_order, before[1])
        with pytest.raises(ValueError, match="read-only"):
            subj.element_order[0] = 0
        subj.free()
        np.testing.assert_array_equal(subj.element_order, before[0])
