"""Robustness on a real head model, and end-to-end regression pins.

Everything else in the GPU suite runs on a 6-tetrahedron single-tissue cube, which by
construction cannot exercise seven-tissue conductivity contrasts, a curved non-convex scalp,
sliver tetrahedra, or an AMG hierarchy deeper than one level. These tests run on
``tests/data/head_patch_r25.msh.gz`` — a 25 mm ball cropped out of a SimNIBS CHARM head mesh
around the vertex (see ``tools/make_test_patch.py``).

Marked ``reference`` tests need the full 184 MB mesh via ``CUNIBS_REFERENCE_MESH``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cunibs.fem.assembly import TISSUE_CONDUCTIVITY
from cunibs.mesh import VOLUME_KEY_TO_LABEL, load_mesh

pytestmark = pytest.mark.gpu

# Pinned from a run of the committed fixture. Changing these means the pipeline changed.
PATCH_PEAK_MAGNE = 0.556405
PATCH_PEAK_MAGNE_HARMONIC = 0.484461
PATCH_PEAK_LOCATION_MM = np.array([11.794, -1.125, 77.349])

# Pinned from sub-001 at the same canonical placement.
FULL_PEAK_MAGNE = 1.4368
FULL_NODES = 881_456
FULL_TETS = 4_943_304
FULL_SKIN_TRIS = 75_058


@pytest.mark.realmesh
def test_patch_counts_match_manifest(patch_mesh, patch_manifest):
    """Guards the crop tool and the parser together: any drift in either shows up here."""
    assert patch_mesh.n_nodes == patch_manifest["n_nodes"] == 8404
    assert patch_mesh.tet_nodes.shape == (patch_manifest["n_tets"], 4)
    assert patch_mesh.tet_nodes.shape[0] == 41_139
    assert patch_mesh.skin_tris.shape == (patch_manifest["n_skin_tris"], 3)

    tags, counts = np.unique(patch_mesh.tet_tags, return_counts=True)
    assert {str(int(t)): int(c) for t, c in zip(tags, counts, strict=False)} == patch_manifest[
        "tet_tags"
    ]
    assert set(tags.tolist()) == {1, 2, 3, 5, 7, 8, 9}


@pytest.mark.realmesh
def test_patch_indices_are_in_range(patch_mesh):
    assert patch_mesh.tet_nodes.min() >= 0
    assert patch_mesh.tet_nodes.max() < patch_mesh.n_nodes
    assert patch_mesh.skin_tris.min() >= 0
    assert patch_mesh.skin_tris.max() < patch_mesh.n_nodes
    # Every node is referenced by some tet — the crop reindexes orphans away.
    assert np.unique(patch_mesh.tet_nodes).size == patch_mesh.n_nodes


@pytest.mark.realmesh
def test_patch_multi_tissue_conductivity_mapping(cp, patch_subject):
    """First exercise of the multi-tissue conductivity LUT: the cube only ever has one tag."""
    from cunibs.fem.assembly import conductivity_per_tet

    ctx = patch_subject.context
    cond = cp.asnumpy(conductivity_per_tet(ctx.tet_tags))
    tags = cp.asnumpy(ctx.tet_tags)

    assert np.isfinite(cond).all() and (cond > 0).all()
    assert np.unique(cond).size == 7
    for tag in np.unique(tags):
        expected = TISSUE_CONDUCTIVITY[int(tag)]
        np.testing.assert_allclose(cond[tags == tag], expected)


@pytest.mark.realmesh
def test_patch_skin_normals_point_outward(patch_mesh, patch_manifest, patch_skin_normals):
    """All 1125 smoothed normals face away from the crop centre — no inverted triangles."""
    normals = patch_skin_normals
    centroids = patch_mesh.nodes_mm[patch_mesh.skin_tris].mean(axis=1)
    outward = centroids - np.asarray(patch_manifest["center_mm"])
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    assert np.all((outward * normals).sum(axis=1) > 0)


@pytest.mark.realmesh
def test_patch_no_degenerate_tets(cp, patch_subject):
    """Real meshes carry slivers; every element must still have strictly positive volume."""
    vols = cp.asnumpy(patch_subject.context.vols).astype(np.float64)
    assert (vols > 0).all()
    assert vols.max() / vols.min() < 1e6  # measured ≈ 1.0e4


@pytest.mark.realmesh
def test_patch_stiffness_is_spd_after_grounding(cp, patch_subject):
    """Grounding one DOF must leave the reduced stiffness positive definite."""
    import cupyx.scipy.sparse as csp

    solver = patch_subject.context.solver
    n = int(solver.idx.shape[0])
    a = csp.csr_matrix((solver.values, solver.col_idx, solver.row_ptr), shape=(n, n))

    rng = np.random.default_rng(0)
    for _ in range(20):
        x = cp.asarray(rng.standard_normal(n))
        assert float(x @ a.dot(x)) > 0

    # Symmetry, checked on a random projection rather than densifying 8403².
    y = cp.asarray(rng.standard_normal(n))
    x = cp.asarray(rng.standard_normal(n))
    assert float(x @ a.dot(y)) == pytest.approx(float(y @ a.dot(x)), rel=1e-10)


@pytest.mark.realmesh
def test_patch_solve_peak_regression(patch_subject, patch_placement, d70_coil):
    """End-to-end pin: real mesh, bundled MagStim D70, 1e6 A/s.

    Pinned on ``recovery="raw"`` so it stays a pin on the *solve*; recovery has its own coverage
    in test_recovery.py.
    """
    r = patch_subject.simulate(
        d70_coil,
        patch_placement,
        1e6,
        magnitude=True,
        vectors=True,
        potential=True,
        recovery="raw",
    )
    assert r.peak_magnE() == pytest.approx(PATCH_PEAK_MAGNE, rel=2e-3)
    np.testing.assert_allclose(r.peak_location_mm(), PATCH_PEAK_LOCATION_MM, atol=2.0)
    assert r.coil_name == d70_coil.name


@pytest.mark.realmesh
def test_patch_default_recovery_peak_regression(patch_subject, patch_placement, d70_coil):
    """The same pin on the default path, which is harmonic recovery."""
    r = patch_subject.simulate(d70_coil, patch_placement, 1e6, magnitude=True)
    assert r.recovery == "harmonic"
    assert r.peak_magnE() == pytest.approx(PATCH_PEAK_MAGNE_HARMONIC, rel=2e-3)


@pytest.mark.realmesh
def test_patch_solve_converged(patch_subject, patch_placement, d70_coil):
    from cunibs.fem import solve_placement

    ctx = patch_subject.context
    solve_placement(
        ctx,
        d70_coil.positions_m,
        d70_coil.moments,
        patch_placement.center_mm,
        patch_placement.handle_mm,
        patch_placement.distance_mm,
        1e6,
    )
    assert ctx.solver.last_relative_residual <= ctx.solver.tolerance
    assert 0 < ctx.solver.last_iterations < ctx.solver.max_iters


@pytest.mark.realmesh
def test_patch_multilevel_solve_true_residual(
    cp, fresh_subject, patch_mesh, patch_placement, d70_coil
):
    """A multi-level V-cycle solve, checked against a freshly computed ``b - Ax``.

    ``solve_mixed`` returns the residual carried by the CG recurrence, not one recomputed from
    the operator, so a converged-looking result is not self-evidently converged. The cube
    fixture in ``test_solver.py`` already compares against a dense solve, but it is small
    enough that the hierarchy has no coarse levels at all; this is the same guarantee at a
    size where the V-cycle actually runs.
    """
    import cupyx.scipy.sparse as csp

    from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform
    from cunibs.fem.solve import (
        _assemble_rhs_kernel,
        _dadt_node_to_elm,
        solve_grounded,
    )

    ctx = fresh_subject(patch_mesh).context
    tf = compute_coil_transform(
        ctx, patch_placement.center_mm, patch_placement.handle_mm, patch_placement.distance_mm
    )
    dadt_elm = _dadt_node_to_elm(
        coil_dadt_at_nodes(d70_coil.positions_m, d70_coil.moments, tf, 1e6, ctx.nodes_mm),
        ctx.tet_nodes,
    )
    b = _assemble_rhs_kernel(ctx, dadt_elm)

    solver = ctx.solver
    v = solve_grounded(solver, b)

    n_red = int(solver.idx.shape[0])
    a_red = csp.csr_matrix(
        (solver.values, solver.col_idx, solver.row_ptr), shape=(n_red, n_red)
    )
    b_red = cp.ascontiguousarray(b[solver.idx], dtype=cp.float64)
    x_red = cp.ascontiguousarray(v[solver.idx])
    true_residual = float(cp.linalg.norm(b_red - a_red @ x_red) / cp.linalg.norm(b_red))
    assert true_residual <= solver.tolerance

    # Guard the premise: at zero coarsening levels the V-cycle is just the dense coarse inverse,
    # an exact solve, and this would silently become a re-test of the trivial case.
    assert solver.precond.n_levels() >= 1, "the patch hierarchy should have coarse levels"


@pytest.mark.realmesh
def test_patch_roi_on_cortex(cp, patch_subject):
    """A gray-matter ROI on real cortex: correct region, volume weights, and centroid."""
    gm = patch_subject.mesh.tet_barycenters_mm[patch_subject.mesh.tet_tags == 2]
    point = gm.mean(axis=0)
    roi = patch_subject.roi(point, radius_mm=5.0, region="gray_matter")

    idx = cp.asnumpy(roi.elem_idx)
    weights = cp.asnumpy(roi.weights)
    assert idx.size > 1
    np.testing.assert_array_equal(patch_subject.mesh.tet_tags[idx], 2)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(weights > 0)

    barys = patch_subject.mesh.tet_barycenters_mm[idx]
    assert np.linalg.norm(barys - point, axis=1).max() <= 5.0 + 1e-9
    np.testing.assert_allclose(cp.asnumpy(roi.barycenter_mm), weights @ barys, atol=1e-9)


@pytest.mark.realmesh
def test_patch_depth_probes_walk_into_the_brain(cp, patch_subject, patch_top_mm):
    """Probes stepping inward from the scalp must advance monotonically along the direction."""
    depths = [0.0, 5.0, 10.0, 15.0, 20.0]
    inward = np.array([0.0, 0.0, -1.0])
    probes = patch_subject.depth_probes(patch_top_mm, inward, depths, radius_mm=3.0)

    assert len(probes) == len(depths)
    projected = [float((cp.asnumpy(p.barycenter_mm) - patch_top_mm) @ inward) for p in probes]
    assert all(a < b for a, b in itertools.pairwise(projected))
    for p in probes:
        assert abs(float(p.weights.sum()) - 1.0) < 1e-12


@pytest.mark.realmesh
def test_patch_metrics_all_present_regions(patch_subject, patch_placement, d70_coil):
    """Every tissue present must produce finite, ordered metrics."""
    r = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, vectors=True, potential=True
    )
    present = {VOLUME_KEY_TO_LABEL[int(t)] for t in np.unique(patch_subject.mesh.tet_tags)}
    assert len(present) == 7

    for region in [*sorted(present), "all"]:
        m = r.summary_for(region)
        assert np.isfinite(m["peak_magnE"]) and m["peak_magnE"] > 0, region
        assert m["region_volume_m3"] > 0, region
        d = m["distribution"]
        assert d["p50"] <= d["p95"] <= d["p99"] <= d["p99.9"], region
        assert np.isfinite(m["center_of_gravity_mm"]).all(), region


@pytest.mark.realmesh
def test_patch_field_result_roundtrip(tmp_path, patch_subject, patch_placement, d70_coil):
    """Save/load at 41k elements, with compression actually applied."""
    from cunibs.simulation import FieldResult

    r = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, vectors=True, potential=True
    )
    path = tmp_path / "patch.h5"
    r.save(path)
    loaded = FieldResult.load(path)

    for name in ("E", "magnE", "v", "transform", "vols", "tet_tags", "barycenters_mm"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(r, name), err_msg=name)
    assert loaded.summary["peak_magnE"] == r.summary["peak_magnE"]
    # 41k tets of raw float32/float64 would be well over 3 MB uncompressed.
    assert path.stat().st_size < 3_000_000


@pytest.mark.realmesh
def test_patch_uq_multi_tissue(patch_subject, patch_placement, d70_coil):
    """Conductivity UQ across three real tissues produces finite, non-trivial local variance."""
    from cunibs import ConductivityUQConfig

    r = patch_subject.simulate_conductivity_uq(
        d70_coil,
        patch_placement,
        ConductivityUQConfig(n_samples=64, perturbed_tags=(1, 2, 3), seed=0),
        1e6,
        moments=True,
    )
    cov = np.asarray(r.cov_magnE)
    assert np.isfinite(cov).all()
    assert cov.max() > 1e-3
    assert r.sigma_samples.shape == (64, 3)


@pytest.mark.reference
def test_full_mesh_load_counts(reference_mesh_path):
    """The full head model parses to the documented element counts and tag histogram."""
    mesh = load_mesh(reference_mesh_path)
    assert mesh.n_nodes == FULL_NODES
    assert mesh.tet_nodes.shape[0] == FULL_TETS
    assert mesh.skin_tris.shape[0] == FULL_SKIN_TRIS
    tags, counts = np.unique(mesh.tet_tags, return_counts=True)
    assert set(tags.tolist()) == {1, 2, 3, 5, 6, 7, 8, 9, 10}
    assert counts.sum() == FULL_TETS


@pytest.mark.reference
@pytest.mark.slow
def test_full_mesh_forward(cp, fresh_subject, reference_mesh_path, d70_coil):
    """4.9M tetrahedra end to end: block-vs-serial parity and the peak field at scale.

    Pinned on ``recovery="raw"`` so it stays a pin on the *solve*; recovery has its own coverage
    in test_recovery.py.
    """
    from cunibs.simulation import Placement

    mesh = load_mesh(reference_mesh_path)
    subj = fresh_subject(mesh)

    skin_nodes = np.unique(mesh.skin_tris)
    coords = mesh.nodes_mm[skin_nodes]
    top = coords[np.argmax(coords[:, 2])]
    placements = [Placement(top, top + [0.0, 50.0, 0.0], 4.0)] * 8

    blocked = list(subj.iter_simulate(d70_coil, placements, 1e6, recovery="raw"))
    assert blocked[0].peak_magnE() == pytest.approx(FULL_PEAK_MAGNE, rel=5e-3)

    serialized = list(
        subj.iter_simulate(d70_coil, placements[:1], 1e6, block_k=1, recovery="raw")
    )
    assert blocked[0].peak_magnE() == pytest.approx(serialized[0].peak_magnE(), rel=2e-5)
    assert cp.get_default_memory_pool().used_bytes() < 4e9
