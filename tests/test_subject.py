"""Subject lifecycle: context caching, ``free()``, the context manager, and ROI selection.

``free()`` exists so a loop over many subjects can reclaim VRAM as it goes, and the
streaming API exists so a long placement sweep does not accumulate memory on either side.
Both are memory contracts that nothing else in the suite checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from cunibs import Placement, Subject
from cunibs.mesh import load_mesh

pytestmark = pytest.mark.gpu

PLACEMENT = Placement([50, 50, 100], [50, 100, 100])


@pytest.mark.realmesh
def test_from_mesh_matches_load_mesh(fresh_subject, patch_mesh_path, patch_mesh):
    subj = Subject.from_mesh(patch_mesh_path)
    try:
        assert subj.mesh.n_nodes == patch_mesh.n_nodes
        np.testing.assert_array_equal(subj.mesh.tet_tags, patch_mesh.tet_tags)
        np.testing.assert_array_equal(subj.mesh.tet_nodes, patch_mesh.tet_nodes)
        np.testing.assert_allclose(subj.mesh.nodes_mm, patch_mesh.nodes_mm)
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


def test_mesh_property_is_the_input_mesh(cube_mesh):
    subj = Subject(cube_mesh)
    assert subj.mesh is cube_mesh


@pytest.mark.realmesh
def test_free_releases_device_memory(cp, patch_mesh):
    """Dropping the cached context must return the pool to roughly its starting size."""
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline = pool.used_bytes()

    subj = Subject(patch_mesh)
    assert subj.context is not None
    grown = pool.used_bytes()
    assert grown - baseline > 1_000_000  # the patch's resident arrays are several MB

    subj.free()
    assert pool.used_bytes() - baseline < 1_000_000


@pytest.mark.realmesh
def test_summary_sweep_does_not_grow_pool(
    cp, fresh_subject, patch_mesh, d70_coil, patch_placement
):
    """The documented scratch-pool contract: a long sweep must not accumulate device memory."""
    subj = fresh_subject(patch_mesh)
    pool = cp.get_default_memory_pool()

    subj.simulate(d70_coil, patch_placement)
    subj.simulate(d70_coil, patch_placement)
    settled = pool.used_bytes()
    for _ in range(8):
        subj.simulate(d70_coil, patch_placement)
    assert pool.used_bytes() - settled < 1_000_000


@pytest.mark.realmesh
def test_retained_fields_are_host_backed_and_leave_the_pool_flat(
    cp, fresh_subject, patch_mesh, d70_coil, patch_placement
):
    """The counterpart: retaining fields moves them to the host, off the scarcer resource."""
    subj = fresh_subject(patch_mesh)
    pool = cp.get_default_memory_pool()

    subj.simulate(d70_coil, patch_placement)  # warm the context and its caches
    settled = pool.used_bytes()
    results = list(subj.iter_simulate(d70_coil, [patch_placement] * 4, magnitude=True))

    for r in results:
        assert isinstance(r.magnE, np.ndarray)
        assert r.magnE.shape == (patch_mesh.tet_nodes.shape[0],)
    # Retained results share the metric inputs rather than copying them per result.
    assert results[0].vols is results[1].vols
    # Holding every result on the host must not have grown the device pool.
    assert pool.used_bytes() - settled < 1_000_000


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
    for a, b in zip(unit, scaled):
        np.testing.assert_array_equal(cp.asnumpy(a.elem_idx), cp.asnumpy(b.elem_idx))
        np.testing.assert_allclose(cp.asnumpy(a.barycenter_mm), cp.asnumpy(b.barycenter_mm))


def test_depth_probes_matches_roi_at_each_depth(cp, cube_subject):
    depths = [0.0, 15.0, 35.0]
    probes = cube_subject.depth_probes([50, 50, 100], [0, 0, -1], depths, radius_mm=40.0)
    for depth, probe in zip(depths, probes):
        expected = cube_subject.roi([50, 50, 100 - depth], radius_mm=40.0, region="gray_matter")
        np.testing.assert_array_equal(cp.asnumpy(probe.elem_idx), cp.asnumpy(expected.elem_idx))


def test_depth_probes_rejects_bad_vectors(cube_subject):
    with pytest.raises(ValueError):
        cube_subject.depth_probes([50, 50], [0, 0, -1], [0.0])
    with pytest.raises(ValueError):
        cube_subject.depth_probes([50, 50, 100], [0, 0], [0.0])


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
    np.testing.assert_array_equal(first.skin_triangle_normals, second.skin_triangle_normals)
