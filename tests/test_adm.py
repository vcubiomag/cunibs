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
    e_exact = cp.asnumpy(evaluate_exact(adj, coil, pl, DIDT))
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
    brute_max = float(np.linalg.norm(e, axis=1).max())

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
    with pytest.warns(UserWarning, match="outside gray_matter"):
        rt = resolve_target(cube_subject.context, Target(point, radius_mm=1e-6))
    expected = int(np.argmin(np.linalg.norm(cube_mesh.tet_barycenters_mm - point, axis=1)))
    assert cp.asnumpy(rt.elem_idx).tolist() == [expected]
    assert float(rt.weights.sum()) == pytest.approx(1.0, abs=1e-12)


def test_resolve_target_reports_how_far_outside_the_region_it_landed(cube_subject):
    """Inside the cube the offset is zero although every barycentre is tens of mm away."""
    import warnings

    from cunibs.adm.target import Target, resolve_target

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inside = resolve_target(cube_subject.context, Target([50, 50, 50]))
    assert inside.offset_mm == 0.0
    with pytest.warns(UserWarning, match="outside gray_matter"):
        outside = resolve_target(cube_subject.context, Target([50, 50, 400]))
    assert outside.offset_mm > 200.0


def test_a_resolved_target_is_accepted_wherever_a_target_is(cp, cube_subject, coil):
    from cunibs.adm import build_reciprocity, evaluate
    from cunibs.adm.target import Target
    from cunibs.simulation import Placement

    center = np.array([[50.0, 50.0, 100.0]])
    pl = Placement([50, 50, 100], [50, 150, 100], 4.0)
    by_target = build_reciprocity(cube_subject.context, coil, Target([50, 50, 60]), center)
    by_roi = build_reciprocity(
        cube_subject.context, coil, cube_subject.roi([50, 50, 60]), center
    )
    np.testing.assert_array_equal(
        cp.asnumpy(evaluate(by_roi, coil, pl, DIDT)),
        cp.asnumpy(evaluate(by_target, coil, pl, DIDT)),
    )


def test_resolve_target_equidistant_fallback_keeps_all_ties(cp, cube_subject):
    """On the diagonal every tetrahedron ties, and the fallback keeps them all, volume-weighted."""
    from cunibs.adm.target import Target, resolve_target

    with pytest.warns(UserWarning, match="outside gray_matter"):
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


def test_optimize_placement_defaults_to_what_the_field_was_built_for(cube_subject, coil):
    from cunibs.adm import build_reciprocity, optimize_placement
    from cunibs.adm.target import Target

    centers = np.array([[50.0, 50.0, 100.0], [45.0, 52.0, 100.0]])
    recip = build_reciprocity(
        cube_subject, coil, Target([50, 50, 60]), centers, distance_mm=6.0
    )
    implicit = optimize_placement(recip, coil)
    explicit = optimize_placement(recip, coil, centers, distance_mm=6.0)
    assert implicit.best_placement.distance_mm == 6.0
    np.testing.assert_array_equal(implicit.center_objective, explicit.center_objective)


def test_optimize_rejects_bad_sampling_before_building_the_field(
    cube_subject, coil, monkeypatch
):
    import sys

    from cunibs import adm
    from cunibs.adm.target import Target

    opt = sys.modules["cunibs.adm.optimize"]

    def unreachable(*_args, **_kwargs):
        raise AssertionError("built the reciprocity field for an invalid search")

    monkeypatch.setattr(opt, "build_reciprocity", unreachable)
    center = np.array([[50.0, 50.0, 100.0]])
    with pytest.raises(ValueError, match="odd"):
        adm.optimize(cube_subject, coil, Target([50, 50, 60]), center, n_samples=8)
    with pytest.raises(ValueError, match="resolution"):
        adm.optimize(cube_subject, coil, Target([50, 50, 60]), center, resolution=4)


@pytest.mark.parametrize(
    "kwargs", [{"distance_mm": -1.0}, {"spacing_mm": 0.0}, {"margin_mm": float("nan")}]
)
def test_build_reciprocity_rejects_bad_geometry(cube_subject, coil, kwargs):
    from cunibs.adm import build_reciprocity
    from cunibs.adm.target import Target

    with pytest.raises(ValueError, match="must be"):
        build_reciprocity(
            cube_subject, coil, Target([50, 50, 60]), np.array([[50.0, 50.0, 100.0]]), **kwargs
        )


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


def test_evaluate_frames_matches_the_placement_path(cp, cube_reciprocity, coil):
    """A frame taken verbatim must give what the same frame derived from a Placement gives.

    This is the contract a cross-code comparison rests on: adopting another code's coil-to-head
    affine has to be equivalent to re-deriving it here, or the two are evaluating different
    placements while appearing to agree on the parameters.
    """
    from cunibs.adm import evaluate, evaluate_frames
    from cunibs.fem.placement import compute_coil_transforms
    from cunibs.simulation import Placement

    pls = [
        Placement([50, 50, 100], [50, 150, 100], 4.0),
        Placement([45, 55, 100], [90, 150, 100], 4.0),
        Placement([55, 45, 100], [10, 150, 100], 6.0),
    ]
    ctx = cube_reciprocity.ctx
    frames = compute_coil_transforms(
        ctx,
        cp.asarray([pl.center_mm for pl in pls], dtype=cp.float64),
        cp.asarray([pl.handle_mm for pl in pls], dtype=cp.float64),
        cp.asarray([pl.distance_mm for pl in pls], dtype=cp.float64),
    )
    np.testing.assert_array_equal(
        cp.asnumpy(evaluate_frames(cube_reciprocity, coil, frames, DIDT)),
        cp.asnumpy(evaluate(cube_reciprocity, coil, pls, DIDT)),
    )


def test_evaluate_frames_ignores_the_scalp(cp, cube_reciprocity, coil):
    """The frame is used as given: translating it must move the coil, not re-project it."""
    from cunibs.adm import evaluate_frames

    base = cp.eye(4, dtype=cp.float64)[None]
    base[0, :3, 3] = cp.asarray([50.0, 50.0, 100.0])
    lifted = base.copy()
    lifted[0, 2, 3] += 6.0

    near = cp.asnumpy(evaluate_frames(cube_reciprocity, coil, base, DIDT))
    far = cp.asnumpy(evaluate_frames(cube_reciprocity, coil, lifted, DIDT))
    assert np.linalg.norm(far) < np.linalg.norm(near)


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


def test_optimize_result_fields_consistent(cube_subject, coil):
    from cunibs import adm
    from cunibs.adm.target import Target

    ctx = cube_subject.context
    centers = np.array(
        [[50.0 + dx, 50.0 + dy, 100.0] for dx in (-8, 0, 8) for dy in (-8, 0, 8)]
    )
    result = adm.optimize(ctx, coil, Target([50, 50, 60]), centers)

    objectives = result.center_objective
    assert isinstance(objectives, np.ndarray)
    assert isinstance(result.centers_mm, np.ndarray)
    assert objectives.shape == (len(centers),)
    assert result.best_objective == pytest.approx(float(objectives.max()), rel=1e-9)
    assert np.linalg.norm(result.best_handle_mm - result.best_center_mm) == pytest.approx(50.0)

    best = result.best_placement
    np.testing.assert_array_equal(best.center_mm, result.best_center_mm)
    np.testing.assert_array_equal(best.handle_mm, result.best_handle_mm)
    assert best.distance_mm == 4.0


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
        exact = cp.asnumpy(evaluate_exact(adj, d70_coil, patch_placement, DIDT))
        approx = cp.asnumpy(evaluate(sample_qfield(adj, grid), d70_coil, patch_placement, DIDT))
        errors.append(float(np.linalg.norm(approx - exact) / np.linalg.norm(exact)))

    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] <= 3e-3, errors


@pytest.mark.realmesh
def test_dipoles_beside_the_scalp_with_a_grid_point_on_a_node(
    cp, patch_subject, patch_target, patch_top_mm
):
    """A grid point 10 nm from a scalp node and a dipole pair 1 mm above it, against exact.

    The field is singular at every node, so that point's full sample would be ~1e10 times the
    field at the dipoles, and 3 mm cells interpolate a 1 mm neighbourhood badly even without it.
    The pair's opposed moments cancel the smooth far field, as an auxiliary-dipole coil's do, so
    what is left is mostly the near field; the error is taken against one dipole's own.
    """
    from cunibs.adm.evaluate import evaluate_frames
    from cunibs.adm.grid import build_grid
    from cunibs.adm.reciprocity import exact_functional, sample_qfield, solve_adjoint
    from cunibs.adm.target import Target, resolve_target
    from cunibs.coil import Coil
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = patch_subject.context
    adj = solve_adjoint(ctx, resolve_target(ctx, Target(patch_target, region="gray_matter")))
    node_m = cp.asarray(patch_top_mm * 1e-3)
    grid = build_grid(node_m[None], spacing_mm=3.0, margin_mm=8.0)
    nearest = (
        grid.origin_m + cp.rint((node_m - grid.origin_m) / grid.spacing_m) * grid.spacing_m
    )
    grid.origin_m = grid.origin_m + (node_m - nearest) + 1e-8
    recip = sample_qfield(adj, grid)

    coil = Coil(
        np.array([[-1e-3, 0.0, 0.0], [1e-3, 0.0, 0.0]]),
        np.array([[0.6, -0.3, 0.74], [-0.6, 0.3, -0.74]]),
    )
    offsets = [(dx, dy, h) for dx in (-0.7, 0.0, 0.9) for dy in (-0.4, 0.5) for h in (0.6, 1.0)]
    frames = np.repeat(np.eye(4)[None], len(offsets), axis=0)
    frames[:, :3, 3] = patch_top_mm + np.asarray(offsets)

    def exact(c):
        return np.stack(
            [
                cp.asnumpy(
                    exact_functional(
                        adj.node_weights,
                        coil_dadt_at_nodes(c.positions_m, c.moments, f, DIDT, ctx.nodes_mm),
                    )
                )
                for f in frames
            ]
        )

    approx = evaluate_frames(recip, coil, frames, DIDT)
    one = Coil(coil.positions_m[:1], coil.moments[:1])
    err = np.linalg.norm(approx - exact(coil), axis=1) / np.linalg.norm(exact(one), axis=1)
    assert err.max() <= 1.5e-2, err


@pytest.mark.realmesh
def test_optimize_matches_brute_force_on_patch(
    patch_subject, d70_coil, patch_target, patch_top_mm
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
    brute = np.linalg.norm(evaluate(recip, d70_coil, sweep, DIDT), axis=1)

    assert result.best_objective == pytest.approx(float(brute.max()), rel=5e-3)
    best_angle = angles[int(brute.argmax())]
    delta = abs((result.best_angle_rad - best_angle + np.pi) % (2 * np.pi) - np.pi)
    assert delta <= 2 * np.pi / 360 * 1.5


def test_adjoint_solver_is_cached_on_the_context(
    patch_subject, d70_coil, patch_target, patch_top_mm
):
    """A second search on the same subject must not rebuild the float64 stiffness and hierarchy.

    build_adjoint_solver reassembles the stiffness in float64 and builds its own tight-tolerance
    AMG hierarchy. Neither depends on the target, so a sweep has to share one setup.
    """
    from cunibs.adm import Target, build_reciprocity
    from cunibs.adm.reciprocity import ensure_adjoint_solver

    ctx = patch_subject.context
    ctx.adjoint.clear()

    first = ensure_adjoint_solver(ctx)
    assert len(ctx.adjoint) == 1
    assert ensure_adjoint_solver(ctx) is first

    # build_reciprocity must go through the cache too, or optimize pays per call.
    build_reciprocity(
        ctx,
        d70_coil,
        Target(patch_target, region="gray_matter"),
        patch_top_mm[None],
        spacing_mm=4.0,
    )
    assert len(ctx.adjoint) == 1
    assert ctx.adjoint[next(iter(ctx.adjoint))] is first

    # A different tolerance is a different solver, not a silent reuse of the cached one.
    assert ensure_adjoint_solver(ctx, tolerance=1e-7) is not first
    assert len(ctx.adjoint) == 2
    ctx.adjoint.clear()


@pytest.mark.realmesh
def test_masked_sampling_matches_the_full_box_and_fills_off_coverage_frames(
    cp, patch_subject, d70_coil, patch_target, patch_top_mm
):
    """Sampling only the reachable region must not change any value a frame reads.

    The coverage decides which points are sampled up front. A frame tilted off it lands in cells
    the coverage skipped; those are sampled on demand, and the result has to equal the one from a
    grid sampled everywhere, bit for bit, since every point is the same N-body either way.
    """
    import dataclasses

    from cunibs.adm import Target, build_reciprocity
    from cunibs.adm.evaluate import evaluate_frames
    from cunibs.adm.optimize import _frames, _project
    from cunibs.adm.reciprocity import sample_qfield, solve_adjoint
    from cunibs.adm.target import resolve_target

    ctx = patch_subject.context
    target = Target(patch_target, region="gray_matter")
    centers = patch_top_mm[None]
    masked = build_reciprocity(ctx, d70_coil, target, centers, spacing_mm=4.0)
    assert masked.grid.coverage is not None
    assert int(masked.sampled.sum()) < masked.grid.n_points

    adj = solve_adjoint(ctx, resolve_target(ctx, target))
    full = sample_qfield(adj, dataclasses.replace(masked.grid, coverage=None))

    proj, normal, u, v = _project(ctx, cp.asarray(centers), 4.0)
    frames = _frames(proj, normal, u, v, cp.linspace(0.0, 6.0, 5), 4.0)
    tilt = np.deg2rad(12.0)
    rot = cp.asarray(
        [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
    )
    tilted = frames.copy()
    tilted[:, :3, :3] = frames[:, :3, :3] @ rot
    before = int(masked.sampled.sum())
    for f in (frames, tilted):
        np.testing.assert_array_equal(
            cp.asnumpy(evaluate_frames(masked, d70_coil, f, DIDT)),
            cp.asnumpy(evaluate_frames(full, d70_coil, f, DIDT)),
        )
    assert int(masked.sampled.sum()) > before, (
        "the tilted frames needed points the coverage skipped"
    )


def test_binned_sources_on_a_fine_grid_match_numpy(cp):
    """The node CSR and near-cell mask over 12M extended cells at 1.5 mm, against numpy.

    That is the cell count of a 1.5 mm grid over a head, enough for the CUB histogram behind
    cp.bincount to write out of bounds on an A100. Some of the nodes fall outside the extended
    cells.
    """
    from types import SimpleNamespace

    from cunibs.adm.grid import Grid
    from cunibs.adm.reciprocity import _binned_sources

    rng = np.random.default_rng(3)
    shape = (226, 231, 228)
    spacing_m = 1.5e-3
    origin = np.array([-0.171, -0.118, -0.093])
    span = (np.array(shape) - 1) * spacing_m
    nodes_mm = (origin + rng.uniform(-0.03, 1.03, size=(900_000, 3)) * span) * 1e3
    weights = rng.standard_normal((3, nodes_mm.shape[0], 3))
    adjoint = SimpleNamespace(
        ctx=SimpleNamespace(nodes_mm=nodes_mm), node_weights=cp.asarray(weights)
    )
    grid = Grid(origin_m=cp.asarray(origin), spacing_m=cp.full(3, spacing_m), shape=shape)
    src, src_w, cell_ptr, _, cell_near = _binned_sources(adjoint, grid)

    idx = ((nodes_mm * 1e-3 - origin) / spacing_m).astype(np.float32)
    ext = np.floor(idx).astype(np.int64) + 1
    n_ext = np.array(shape) + 1
    inside = ((ext >= 0) & (ext < n_ext)).all(axis=1)
    assert 0.05 < 1.0 - inside.mean() < 0.2
    n_cells = int(np.prod(n_ext))
    flat = np.where(inside, (ext[:, 0] * n_ext[1] + ext[:, 1]) * n_ext[2] + ext[:, 2], n_cells)
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n_cells + 1)[:n_cells]
    np.testing.assert_array_equal(
        cp.asnumpy(cell_ptr), np.concatenate([[0], np.cumsum(counts)])
    )
    np.testing.assert_array_equal(cp.asnumpy(src), idx[order])
    np.testing.assert_array_equal(cp.asnumpy(src_w), weights.astype(np.float32)[:, order])

    occupied = (counts > 0).reshape(tuple(n_ext))
    near = occupied[:-2] | occupied[1:-1] | occupied[2:]
    near = near[:, :-2] | near[:, 1:-1] | near[:, 2:]
    near = near[:, :, :-2] | near[:, :, 1:-1] | near[:, :, 2:]
    np.testing.assert_array_equal(cp.asnumpy(cell_near), near.astype(np.uint8).ravel())


def test_reach_profile_matches_the_broadcast_distance(cp):
    """The raster kernel against the (rho, h, dipole) distance block it stands in for.

    Every cell's verdict has to agree with the array form exactly, including cells whose centre
    sits on the margin.
    """
    from cunibs.adm.reciprocity import _reach_profile

    rng = np.random.default_rng(5)
    base = np.stack([rng.uniform(0.0, 0.05, 300), rng.uniform(-0.01, 0.01, 300)], axis=1)
    prof = cp.asarray(np.concatenate([base, base[::3]]))
    cell = 0.75e-3
    rc = (cp.arange(90, dtype=cp.float64) + 0.5) * cell
    hc = -0.02 + (cp.arange(60, dtype=cp.float64) + 0.5) * cell
    for margin in (0.0, 2e-3, 8e-3):
        reach_sq = (margin + cell * 0.5 * 2.0**0.5) ** 2
        d2 = (rc[:, None, None] - prof[None, None, :, 0]) ** 2 + (
            hc[None, :, None] - prof[None, None, :, 1]
        ) ** 2
        expected = (d2.min(axis=2) <= reach_sq).astype(cp.uint8)
        got = cp.empty((rc.size, hc.size), dtype=cp.uint8)
        _reach_profile(rc, hc, prof, int(prof.shape[0]), int(hc.size), reach_sq, got)
        assert 0 < int(expected.sum()) < expected.size
        np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(expected))
