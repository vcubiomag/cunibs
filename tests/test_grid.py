"""The ADM sampling grid and batched dipole placement."""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.adm.evaluate import place_coil_dipoles_batch
from cunibs.adm.grid import Grid, build_grid

pytestmark = pytest.mark.gpu


def _points_m(grid):
    """All grid points as a device ``(n_points, 3)`` in C order, the order the kernels use."""
    ijk = np.indices(grid.shape).reshape(3, -1).T
    import cupy as cp

    return cp.asarray(np.asarray(grid.origin_m) + np.asarray(grid.spacing_m) * ijk)


@pytest.fixture
def dipole_cloud(cp):
    rng = np.random.default_rng(0)
    return cp.asarray(rng.uniform(-0.05, 0.05, size=(200, 3)))


def test_build_grid_covers_points_with_margin(cp, dipole_cloud):
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    lo = np.asarray(grid.origin_m)
    hi = lo + np.asarray(grid.spacing_m) * (np.asarray(grid.shape) - 1)
    pts = cp.asnumpy(dipole_cloud)
    assert np.all(lo <= pts.min(axis=0) - 8e-3 + 1e-15)
    assert np.all(hi >= pts.max(axis=0) + 8e-3 - 1e-15)


def test_grid_shape_and_n_points_consistent(dipole_cloud):
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    assert grid.n_points == int(np.prod(grid.shape))
    np.testing.assert_allclose(grid.spacing_m, 3e-3)


def test_build_grid_keeps_a_cell_on_an_axis_with_no_extent(cp):
    grid = build_grid(cp.asarray([[0.01, 0.02, 0.03], [0.05, 0.02, 0.03]]), 3.0, 0.0)
    assert grid.shape[1:] == (2, 2)


def test_world_to_index_roundtrip_is_integral(cp, dipole_cloud):
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    idx = cp.asnumpy(grid.world_to_index(_points_m(grid)))
    assert idx.shape == (3, grid.n_points)

    expected = np.stack(np.unravel_index(np.arange(grid.n_points), grid.shape))
    np.testing.assert_allclose(idx, expected, atol=1e-9)


def test_world_to_index_uses_each_axis_own_origin_and_spacing(cp):
    grid = Grid(
        origin_m=(-0.02, 0.05, 0.11),
        spacing_m=(2e-3, 3e-3, 5e-3),
        shape=(9, 7, 5),
    )
    query = np.array([[-0.02, 0.05, 0.11], [-0.018, 0.056, 0.125], [-0.004, 0.068, 0.13]])
    np.testing.assert_allclose(
        cp.asnumpy(grid.world_to_index(cp.asarray(query))),
        [[0.0, 1.0, 8.0], [0.0, 2.0, 6.0], [0.0, 3.0, 4.0]],
        atol=1e-9,
    )


def test_upper_m_is_the_last_sample_on_each_axis(cp, dipole_cloud):
    """The last index is the bound the evaluate check compares against, so it must be a real sample."""
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    idx = grid.world_to_index(cp.asarray([grid.upper_m]))
    np.testing.assert_allclose(cp.asnumpy(idx).ravel(), np.asarray(grid.shape) - 1.0, atol=1e-9)


def test_place_coil_dipoles_batch_matches_numpy(cp, synthetic_coil):
    rng = np.random.default_rng(3)
    transforms = np.zeros((4, 4, 4))
    for p in range(4):
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        transforms[p, :3, :3] = q * np.sign(np.linalg.det(q))
        transforms[p, :3, 3] = rng.uniform(-100, 100, 3)
        transforms[p, 3, 3] = 1.0

    s, m = place_coil_dipoles_batch(
        cp.asarray(transforms), synthetic_coil.positions_m, synthetic_coil.moments
    )
    for p in range(4):
        rot, trans = transforms[p, :3, :3], transforms[p, :3, 3]
        expected_s = (synthetic_coil.positions_m * 1e3 @ rot.T + trans) * 1e-3
        np.testing.assert_allclose(cp.asnumpy(s[p]), expected_s, atol=1e-12)
        np.testing.assert_allclose(cp.asnumpy(m[p]), synthetic_coil.moments @ rot.T, atol=1e-12)


def test_place_batch_leaves_moments_alone_under_pure_translation(cp, synthetic_coil):
    tf = cp.asarray(np.eye(4)[None].repeat(2, axis=0))
    tf[:, :3, 3] = cp.asarray([[10.0, 20.0, 30.0], [-5.0, 0.0, 1.0]])
    s, m = place_coil_dipoles_batch(tf, synthetic_coil.positions_m, synthetic_coil.moments)
    for p in range(2):
        np.testing.assert_array_equal(cp.asnumpy(m[p]), synthetic_coil.moments)
    np.testing.assert_allclose(
        cp.asnumpy(s[0]) - cp.asnumpy(s[1]), np.full((2, 3), [15e-3, 20e-3, 29e-3]), atol=1e-15
    )


def test_place_batch_agrees_with_the_dadt_placement(cp, cube_subject, figure8_coil):
    """Dipoles placed here and then translated only give the dA/dt the kernel's own placement does."""
    from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform

    ctx = cube_subject.context
    tf = compute_coil_transform(ctx, [50, 50, 120], [90, 150, 100], 4.0)
    s, m = place_coil_dipoles_batch(
        cp.asarray(tf[None]), figure8_coil.positions_m, figure8_coil.moments
    )
    trans = np.eye(4)
    trans[:3, 3] = tf[:3, 3]
    local = cp.asnumpy(s[0]) - tf[:3, 3] * 1e-3

    placed = cp.asnumpy(coil_dadt_at_nodes(local, m[0], trans, 1e6, ctx.nodes_mm))
    direct = cp.asnumpy(
        coil_dadt_at_nodes(
            figure8_coil.positions_m, figure8_coil.moments, tf, 1e6, ctx.nodes_mm
        )
    )
    assert np.linalg.norm(direct) > 0.0
    assert np.linalg.norm(placed - direct) / np.linalg.norm(direct) < 1e-6


def test_grid_for_placements_contains_every_dipole(cp, cube_subject, figure8_coil):
    """The invariant that keeps ``evaluate`` off the silent nearest-neighbour clamp."""
    from cunibs.adm.evaluate import _placed_dipoles, grid_for_placements
    from cunibs.simulation import Placement

    ctx = cube_subject.context
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    placements = [
        Placement(
            [50 + 10 * np.cos(a), 50 + 10 * np.sin(a), 120],
            [50 + 60 * np.cos(a), 50 + 60 * np.sin(a), 120],
            4.0,
        )
        for a in angles
    ]
    grid = grid_for_placements(ctx, figure8_coil, placements, spacing_mm=3.0, margin_mm=8.0)
    s, _ = _placed_dipoles(ctx, figure8_coil, placements)

    pts = cp.asnumpy(s).reshape(-1, 3)
    assert np.all(pts >= grid.origin_m) and np.all(pts <= grid.upper_m)


def test_coverage_grid_contains_all_rotations(cp, cube_subject, figure8_coil):
    """``_coverage_grid`` dilates by max|position| so any in-plane rotation stays inside."""
    from cunibs.adm.evaluate import _placed_dipoles
    from cunibs.adm.reciprocity import _coverage_grid
    from cunibs.fem.placement import compute_coil_transform
    from cunibs.simulation import Placement

    ctx = cube_subject.context
    center = np.array([50.0, 50.0, 120.0])
    grid = _coverage_grid(ctx, figure8_coil, center[None], 4.0, 3.0, 8.0)

    tf = compute_coil_transform(ctx, center, center + [0, 50, 0], 4.0)
    x0, y0 = tf[:3, 0], tf[:3, 1]
    proj = tf[:3, 3]
    sweep = [
        Placement(center, proj + 50 * (np.cos(a) * y0 + np.sin(a) * x0), 4.0)
        for a in np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ]
    s, _ = _placed_dipoles(ctx, figure8_coil, sweep)

    pts = cp.asnumpy(s).reshape(-1, 3)
    assert np.all(pts >= grid.origin_m) and np.all(pts <= grid.upper_m)
