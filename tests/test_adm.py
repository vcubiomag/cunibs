"""ADM reciprocity backend: the self-consistent validation ladder plus optimizer checks.

Every layer is checked against the forward solve (or the layer below), so the tests need no external
oracle. Run on the small synthetic ``cube_mesh``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpu import requires_gpu

pytestmark = requires_gpu


@pytest.fixture
def cp():
    import cupy

    return cupy


@pytest.fixture
def coil():
    from cunibs.coil import Coil

    # A figure-8-like pair placed a little wider so the field over the cube is non-trivial.
    positions_m = np.array([[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]], dtype=np.float64)
    moments = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    return Coil(positions_m=positions_m, moments=moments, name="test", didt_max=1e6)


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
    from cunibs.fem import build_context, solve_placement
    from cunibs.fem.placement import compute_coil_transform
    from cunibs.adm.target import Target

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
