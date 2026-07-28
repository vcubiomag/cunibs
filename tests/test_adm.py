"""ADM reciprocity backend: the self-consistent validation ladder plus optimizer checks.

Every layer is checked against the forward solve (or the layer below), so the tests need no external
oracle. Run on the small synthetic ``cube_mesh``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def coil(figure8_coil):
    return figure8_coil


CENTER = [50.0, 50.0, 100.0]
HANDLE = [50.0, 150.0, 100.0]
DIST = 4.0
DIDT = 1e6


def _setup(cube_mesh, coil):
    from cunibs.fem import build_context, solve_placement
    from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform

    ctx = build_context(cube_mesh)
    res = solve_placement(ctx, coil.positions_m, coil.moments, CENTER, HANDLE, DIST, DIDT)
    tf = compute_coil_transform(ctx, CENTER, HANDLE, DIST)
    dadt_nodes = coil_dadt_at_nodes(coil.positions_m, coil.moments, tf, DIDT, ctx.nodes_mm)
    return ctx, res, dadt_nodes


def test_reciprocity_matches_forward_directional(cp, cube_mesh, coil):
    """Step 2: exact reciprocity functional equals ê·E from the forward solve."""
    from cunibs.adm.reciprocity import exact_functional, solve_adjoint
    from cunibs.adm.target import Target, resolve_target

    ctx, res, dadt_nodes = _setup(cube_mesh, coil)
    tgt_elm = int(cp.argmax(res["magnE"]))
    tgt_pt = cp.asnumpy(cp.asarray(cube_mesh.tet_barycenters_mm)[tgt_elm])

    for direction in ([1, 0, 0], [0, 0, 1], [0.3, -0.7, 0.6]):
        rt = resolve_target(ctx, Target(tgt_pt, direction, region="gray_matter"))
        assert int(rt.elem_idx[0]) == tgt_elm
        adj = solve_adjoint(ctx, rt)
        j = float(exact_functional(adj.node_weights, dadt_nodes)[0])
        ehat = np.asarray(direction, float)
        ehat /= np.linalg.norm(ehat)
        fwd = float(cp.asnumpy(res["E"][tgt_elm]) @ ehat)
        assert j == pytest.approx(fwd, rel=1e-3, abs=1e-3)


def test_reciprocity_magnitude(cp, cube_mesh, coil):
    """Magnitude mode: three adjoint solves reconstruct the full target E-vector."""
    from cunibs.adm.reciprocity import exact_functional, solve_adjoint
    from cunibs.adm.target import Target, resolve_target

    ctx, res, dadt_nodes = _setup(cube_mesh, coil)
    tgt_elm = int(cp.argmax(res["magnE"]))
    tgt_pt = cp.asnumpy(cp.asarray(cube_mesh.tet_barycenters_mm)[tgt_elm])

    rt = resolve_target(ctx, Target(tgt_pt, region="gray_matter"))  # direction=None
    assert rt.magnitude and rt.directions.shape[0] == 3
    adj = solve_adjoint(ctx, rt)
    j = cp.asnumpy(exact_functional(adj.node_weights, dadt_nodes))
    np.testing.assert_allclose(j, cp.asnumpy(res["E"][tgt_elm]), rtol=1e-3, atol=1e-3)


def test_qfield_reformulation_and_grid_interp(cp, cube_mesh, coil):
    """Steps 3-4: the Q reformulation (exact) and its grid interpolation agree with the functional."""
    from cunibs.adm.evaluate import evaluate, evaluate_exact, grid_for_placements
    from cunibs.adm.reciprocity import exact_functional, sample_qfield, solve_adjoint
    from cunibs.adm.target import Target, resolve_target
    from cunibs.simulation import Placement

    ctx, res, dadt_nodes = _setup(cube_mesh, coil)
    tgt_elm = int(cp.argmax(res["magnE"]))
    tgt_pt = cp.asnumpy(cp.asarray(cube_mesh.tet_barycenters_mm)[tgt_elm])
    rt = resolve_target(ctx, Target(tgt_pt, region="gray_matter"))
    adj = solve_adjoint(ctx, rt)

    j = cp.asnumpy(exact_functional(adj.node_weights, dadt_nodes))  # step 2 reference
    pl = Placement(CENTER, HANDLE, DIST)

    grid = grid_for_placements(ctx, coil, [pl], spacing_mm=2.0, margin_mm=8.0)
    e_exact = cp.asnumpy(evaluate_exact(adj, coil, pl, DIDT, center_m=grid.center_m))
    np.testing.assert_allclose(e_exact, j, rtol=1e-3, atol=1e-3)

    recip = sample_qfield(adj, grid)
    e_grid = cp.asnumpy(evaluate(recip, coil, pl, DIDT))
    np.testing.assert_allclose(e_grid, j, rtol=5e-3, atol=5e-3)


def test_optimize_consistent_with_forward(cp, cube_mesh, coil):
    """The optimizer's reported optimum matches a forward solve at that placement."""
    from cunibs import adm
    from cunibs.adm.target import Target
    from cunibs.fem import build_context, solve_placement
    from cunibs.fem.placement import compute_coil_transform

    ctx = build_context(cube_mesh)
    res0 = solve_placement(ctx, coil.positions_m, coil.moments, CENTER, HANDLE, DIST, DIDT)
    tgt_elm = int(cp.argmax(res0["magnE"]))
    tgt_pt = cp.asnumpy(cp.asarray(cube_mesh.tet_barycenters_mm)[tgt_elm])

    tf = compute_coil_transform(ctx, CENTER, HANDLE, DIST)
    x0, y0 = cp.asnumpy(tf[:3, 0]), cp.asnumpy(tf[:3, 1])
    centers = np.array(
        [CENTER + dx * x0 + dy * y0 for dx in (-10, 0, 10) for dy in (-10, 0, 10)]
    )

    result = adm.optimize(ctx, coil, Target(tgt_pt, region="gray_matter"), centers)
    fwd = solve_placement(
        ctx,
        coil.positions_m,
        coil.moments,
        result.best_center_mm,
        result.best_handle_mm,
        DIST,
        DIDT,
    )
    fwd_mag = float(cp.linalg.norm(fwd["E"][tgt_elm]))
    assert result.best_objective == pytest.approx(fwd_mag, rel=5e-3, abs=5e-3)


def test_fourier_matches_brute_angle_sweep(cp, cube_mesh, coil):
    """The closed-form (Fourier) rotation optimum agrees with a brute-force angle scan."""
    from cunibs.adm import build_reciprocity, evaluate
    from cunibs.adm.optimize import optimize_placement
    from cunibs.adm.target import Target
    from cunibs.fem import build_context, solve_placement
    from cunibs.simulation import Placement

    ctx = build_context(cube_mesh)
    res0 = solve_placement(ctx, coil.positions_m, coil.moments, CENTER, HANDLE, DIST, DIDT)
    tgt_elm = int(cp.argmax(res0["magnE"]))
    tgt_pt = cp.asnumpy(cp.asarray(cube_mesh.tet_barycenters_mm)[tgt_elm])
    center = np.array([50.0, 50.0, 100.0])
    recip = build_reciprocity(ctx, coil, Target(tgt_pt, region="gray_matter"), center[None])

    # Closed-form optimum for this single position (cube is near-field -> use more samples).
    result = optimize_placement(recip, coil, center[None], n_samples=17)

    # Brute-force reference: scan the in-plane angle with explicit placements. Build the tangent
    # basis the optimizer uses (x, y from the coil frame) and rotate the handle.
    from cunibs.fem.placement import compute_coil_transform

    tf = compute_coil_transform(ctx, center, [50.0, 150.0, 100.0], DIST)
    x0, y0 = cp.asnumpy(tf[:3, 0]), cp.asnumpy(tf[:3, 1])
    proj = result.best_center_mm
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    pls = [Placement(proj, proj + np.cos(a) * y0 + np.sin(a) * x0, DIST) for a in angles]
    e = evaluate(recip, coil, pls, DIDT)
    brute_max = float(cp.linalg.norm(e, axis=1).max())

    assert result.best_objective == pytest.approx(brute_max, rel=5e-3)


# --- Target resolution -------------------------------------------------------------------


def test_resolve_target_empty_region_raises(cube_subject):
    """The cube is gray matter only, so any other region has no elements."""
    from cunibs.adm.target import Target, resolve_target

    with pytest.raises(ValueError, match="contains no elements"):
        resolve_target(cube_subject.context, Target([50, 50, 50], region="white_matter"))


def test_resolve_target_radius_zero_picks_nearest(cp, cube_subject, cube_mesh):
    from cunibs.adm.target import Target, resolve_target

    point = np.array([10.0, 10.0, 10.0])
    rt = resolve_target(cube_subject.context, Target(point, radius_mm=0.0))
    expected = int(np.argmin(np.linalg.norm(cube_mesh.tet_barycenters_mm - point, axis=1)))
    assert cp.asnumpy(rt.elem_idx).tolist() == [expected]
    np.testing.assert_array_equal(cp.asnumpy(rt.weights), [1.0])


def test_resolve_target_weights_are_volume_fractions(cp, cube_subject, cube_mesh):
    from cunibs.adm.target import Target, resolve_target

    ctx = cube_subject.context
    rt = resolve_target(ctx, Target([50, 50, 50], radius_mm=60.0))
    idx = cp.asnumpy(rt.elem_idx)
    assert idx.size > 1

    vols = cp.asnumpy(ctx.vols).astype(np.float64)[idx]
    np.testing.assert_allclose(cp.asnumpy(rt.weights), vols / vols.sum(), rtol=1e-14)
    assert float(rt.weights.sum()) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        cp.asnumpy(rt.barycenter_mm),
        cp.asnumpy(rt.weights) @ cube_mesh.tet_barycenters_mm[idx],
        rtol=1e-12,
    )


def test_resolve_target_tiny_radius_falls_back_to_nearest(cp, cube_subject, cube_mesh):
    """A ball containing no barycentre degrades to the nearest element(s).

    The cube's six tetrahedra are images of each other under its symmetry group, so a point on
    any symmetry element ties several of them: the main diagonal ties all six, and ``[1e4,0,0]``
    still ties two across the y=z mirror. This point lies on no symmetry element.
    """
    from cunibs.adm.target import Target, resolve_target

    point = np.array([1e4, 3e3, -7e3])
    rt = resolve_target(cube_subject.context, Target(point, radius_mm=1e-6))
    expected = int(np.argmin(np.linalg.norm(cube_mesh.tet_barycenters_mm - point, axis=1)))
    assert cp.asnumpy(rt.elem_idx).tolist() == [expected]
    assert float(rt.weights.sum()) == pytest.approx(1.0, abs=1e-12)


def test_resolve_target_equidistant_fallback_keeps_all_ties(cp, cube_subject):
    """On the diagonal every tetrahedron ties, and the fallback keeps them all, volume-weighted."""
    from cunibs.adm.target import Target, resolve_target

    rt = resolve_target(cube_subject.context, Target([1e4, 1e4, 1e4], radius_mm=1e-6))
    assert cp.asnumpy(rt.elem_idx).size == 6
    assert float(rt.weights.sum()) == pytest.approx(1.0, abs=1e-12)


def test_target_direction_is_normalized():
    from cunibs.adm.target import Target

    t = Target([0, 0, 0], [0.0, 0.0, -7.5])
    np.testing.assert_allclose(t.direction, [0.0, 0.0, -1.0])
    assert not np.shares_memory(t.direction, np.zeros(3))


def test_target_zero_direction_raises():
    from cunibs.adm.target import Target

    with pytest.raises(ValueError, match="non-zero"):
        Target([0, 0, 0], [0.0, 0.0, 0.0])


def test_target_magnitude_mode_uses_identity_basis(cp, cube_subject):
    from cunibs.adm.target import Target, resolve_target

    rt = resolve_target(cube_subject.context, Target([50, 50, 50]))
    assert rt.magnitude
    np.testing.assert_allclose(cp.asnumpy(rt.directions), np.eye(3))


def test_optimize_placement_rejects_even_n_samples(cube_subject, coil):
    from cunibs.adm import build_reciprocity
    from cunibs.adm.optimize import optimize_placement
    from cunibs.adm.target import Target

    ctx = cube_subject.context
    center = np.array([[50.0, 50.0, 100.0]])
    recip = build_reciprocity(ctx, coil, Target([50, 50, 60]), center)
    with pytest.raises(ValueError, match="odd"):
        optimize_placement(recip, coil, center, n_samples=8)


# --- evaluate() contracts ----------------------------------------------------------------


@pytest.fixture
def cube_reciprocity(cube_subject, coil):
    from cunibs.adm import build_reciprocity
    from cunibs.adm.target import Target

    return build_reciprocity(
        cube_subject.context, coil, Target([50, 50, 60]), np.array([[50.0, 50.0, 100.0]])
    )


def test_evaluate_batched_matches_single(cp, cube_reciprocity, coil):
    from cunibs.adm import evaluate
    from cunibs.simulation import Placement

    pls = [
        Placement([50, 50, 100], [50, 150, 100], 4.0),
        Placement([45, 55, 100], [90, 150, 100], 4.0),
        Placement([55, 45, 100], [10, 150, 100], 6.0),
    ]
    batched = cp.asnumpy(evaluate(cube_reciprocity, coil, pls, DIDT))
    assert batched.shape == (3, 3)
    for i, pl in enumerate(pls):
        one = cp.asnumpy(evaluate(cube_reciprocity, coil, pl, DIDT))
        # Not bitwise: the batch reduces over a (P, N, 3) block, the single over (1, N, 3).
        np.testing.assert_allclose(batched[i], one, rtol=1e-12)


def test_evaluate_outside_the_grid_raises(cube_reciprocity, coil):
    """The grid is sized for distance_mm=4; a far larger stand-off leaves it entirely."""
    from cunibs.adm import evaluate
    from cunibs.simulation import Placement

    far = Placement([50, 50, 100], [50, 150, 100], 200.0)
    with pytest.raises(ValueError, match="outside the reciprocity grid"):
        evaluate(cube_reciprocity, coil, far, DIDT)


def test_evaluate_reports_the_offending_sample_in_a_batch(cube_reciprocity, coil):
    from cunibs.adm import evaluate
    from cunibs.simulation import Placement

    good = Placement([50, 50, 100], [50, 150, 100], 4.0)
    far = Placement([50, 50, 100], [50, 150, 100], 200.0)
    with pytest.raises(ValueError, match=r"sample 2 of 3"):
        evaluate(cube_reciprocity, coil, [good, good, far], DIDT)


def test_evaluate_accepts_a_dipole_on_the_grid_corner(cp, cube_reciprocity):
    """The bounds check has to admit the last sample, not just the interior."""
    from cunibs.adm.evaluate import _require_in_grid

    grid = cube_reciprocity.grid
    corners = cp.stack([grid.origin_m, grid.upper_m])
    _require_in_grid(grid, grid.world_to_index(corners), offset=0, n_dip=1, n_total=2)


def test_evaluate_is_linear_in_didt(cp, cube_reciprocity, coil):
    from cunibs.adm import evaluate
    from cunibs.simulation import Placement

    pl = Placement([50, 50, 100], [50, 150, 100], 4.0)
    one = cp.asnumpy(evaluate(cube_reciprocity, coil, pl, 1e6))
    two = cp.asnumpy(evaluate(cube_reciprocity, coil, pl, 2e6))
    np.testing.assert_allclose(two, 2.0 * one, rtol=1e-9)


def test_optimize_result_fields_consistent(cp, cube_subject, coil):
    from cunibs import adm
    from cunibs.adm.target import Target

    ctx = cube_subject.context
    centers = np.array(
        [[50.0 + dx, 50.0 + dy, 100.0] for dx in (-8, 0, 8) for dy in (-8, 0, 8)]
    )
    result = adm.optimize(ctx, coil, Target([50, 50, 60]), centers)

    objectives = cp.asnumpy(cp.asarray(result.center_objective))
    assert objectives.shape == (len(centers),)
    assert result.best_objective == pytest.approx(float(objectives.max()), rel=1e-9)
    assert np.linalg.norm(result.best_handle_mm - result.best_center_mm) > 0


# --- The real-mesh ladder ----------------------------------------------------------------


@pytest.fixture(scope="session")
def patch_target(patch_mesh):
    """A gray-matter point a little below the vertex, well inside the crop."""
    gm = patch_mesh.tet_barycenters_mm[patch_mesh.tet_tags == 2]
    return gm[np.argmax(gm[:, 2])] - np.array([0.0, 0.0, 4.0])


@pytest.mark.realmesh
def test_reciprocity_matches_forward_on_patch(
    cp, patch_subject, d70_coil, patch_target, patch_placement
):
    """The adjoint functional against ê·E from a forward solve, on multi-tissue geometry.

    The σ-weighted adjoint node weights only really matter once conductivity varies between
    elements, which the single-tissue cube cannot express.
    """
    from cunibs.adm.reciprocity import exact_functional, solve_adjoint
    from cunibs.adm.target import Target, resolve_target
    from cunibs.fem import solve_placement
    from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform

    ctx = patch_subject.context
    res = solve_placement(
        ctx,
        d70_coil.positions_m,
        d70_coil.moments,
        patch_placement.center_mm,
        patch_placement.handle_mm,
        patch_placement.distance_mm,
        DIDT,
    )
    tf = compute_coil_transform(
        ctx, patch_placement.center_mm, patch_placement.handle_mm, patch_placement.distance_mm
    )
    dadt_nodes = coil_dadt_at_nodes(
        d70_coil.positions_m, d70_coil.moments, tf, DIDT, ctx.nodes_mm
    )

    rt = resolve_target(ctx, Target(patch_target, region="gray_matter"))
    elem = int(cp.asnumpy(rt.elem_idx)[0])
    adj = solve_adjoint(ctx, rt)
    got = cp.asnumpy(exact_functional(adj.node_weights, dadt_nodes))
    expected = cp.asnumpy(res["E"][elem]).astype(np.float64)

    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) <= 1e-2


@pytest.mark.realmesh
def test_grid_interpolation_error_converges(
    cp, patch_subject, d70_coil, patch_target, patch_placement
):
    """Refining the Q-grid must monotonically reduce the interpolation error. Self-validating."""
    from cunibs.adm.evaluate import evaluate, evaluate_exact, grid_for_placements
    from cunibs.adm.reciprocity import sample_qfield, solve_adjoint
    from cunibs.adm.target import Target, resolve_target

    ctx = patch_subject.context
    rt = resolve_target(ctx, Target(patch_target, region="gray_matter"))
    adj = solve_adjoint(ctx, rt)

    errors = []
    for spacing in (4.0, 2.0, 1.0):
        grid = grid_for_placements(
            ctx, d70_coil, [patch_placement], spacing_mm=spacing, margin_mm=8.0
        )
        exact = cp.asnumpy(
            evaluate_exact(adj, d70_coil, patch_placement, DIDT, center_m=grid.center_m)
        )
        approx = cp.asnumpy(evaluate(sample_qfield(adj, grid), d70_coil, patch_placement, DIDT))
        errors.append(float(np.linalg.norm(approx - exact) / np.linalg.norm(exact)))

    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] <= 3e-3, errors


@pytest.mark.realmesh
def test_optimize_matches_brute_force_on_patch(
    cp, patch_subject, d70_coil, patch_target, patch_top_mm
):
    """The closed-form rotation optimum against a 360-angle brute-force sweep on real scalp."""
    from cunibs.adm import build_reciprocity, evaluate
    from cunibs.adm.optimize import optimize_placement
    from cunibs.adm.target import Target
    from cunibs.fem.placement import compute_coil_transform
    from cunibs.simulation import Placement

    ctx = patch_subject.context
    centers = patch_top_mm[None]
    recip = build_reciprocity(
        ctx, d70_coil, Target(patch_target, region="gray_matter"), centers, spacing_mm=2.0
    )
    result = optimize_placement(recip, d70_coil, centers, n_samples=17)

    tf = compute_coil_transform(ctx, patch_top_mm, patch_top_mm + [0.0, 50.0, 0.0], 4.0)
    x0, y0 = tf[:3, 0], tf[:3, 1]
    proj = result.best_center_mm
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    sweep = [
        Placement(proj, proj + 50 * (np.cos(a) * y0 + np.sin(a) * x0), 4.0) for a in angles
    ]
    brute = cp.asnumpy(cp.linalg.norm(evaluate(recip, d70_coil, sweep, DIDT), axis=1))

    assert result.best_objective == pytest.approx(float(brute.max()), rel=5e-3)
    best_angle = angles[int(brute.argmax())]
    delta = abs((result.best_angle_rad - best_angle + np.pi) % (2 * np.pi) - np.pi)
    assert delta <= 2 * np.pi / 360 * 1.5
