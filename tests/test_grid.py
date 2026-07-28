"""The ADM sampling grid and batched dipole placement — previously untested end to end.

``adm/grid.py`` decides where the Q-field is sampled and how world coordinates map to
``map_coordinates`` index space. An axis transposition between ``points_m`` and
``world_to_index`` would silently corrupt every ADM evaluation while leaving shapes intact,
so the load-bearing test here is interpolating a known linear field.
"""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.adm.grid import Grid, build_grid
from cunibs.adm.place import place_coil_dipoles_batch

pytestmark = pytest.mark.gpu


@pytest.fixture
def dipole_cloud(cp):
    rng = np.random.default_rng(0)
    return cp.asarray(rng.uniform(-0.05, 0.05, size=(200, 3)))


def test_build_grid_covers_points_with_margin(cp, dipole_cloud):
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    lo = cp.asnumpy(grid.origin_m)
    hi = lo + cp.asnumpy(grid.spacing_m) * (np.asarray(grid.shape) - 1)
    pts = cp.asnumpy(dipole_cloud)
    assert np.all(lo <= pts.min(axis=0) - 8e-3 + 1e-15)
    assert np.all(hi >= pts.max(axis=0) + 8e-3 - 1e-15)


def test_grid_shape_and_n_points_consistent(cp, dipole_cloud):
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    assert grid.n_points == int(np.prod(grid.shape))
    assert grid.points_m().shape == (grid.n_points, 3)
    np.testing.assert_allclose(cp.asnumpy(grid.spacing_m), 3e-3)


def test_points_m_matches_c_order_reshape(cp, dipole_cloud):
    """points_m() must enumerate in C order so it can be reshaped to ``shape + (3,)``."""
    grid = build_grid(dipole_cloud, spacing_mm=6.0, margin_mm=4.0)
    pts = cp.asnumpy(grid.points_m())
    origin = cp.asnumpy(grid.origin_m)
    spacing = cp.asnumpy(grid.spacing_m)
    rng = np.random.default_rng(1)
    for k in rng.integers(0, grid.n_points, size=200):
        ijk = np.unravel_index(int(k), grid.shape)
        np.testing.assert_allclose(pts[k], origin + spacing * np.asarray(ijk), atol=1e-15)


def test_center_m_is_the_geometric_center(cp, dipole_cloud):
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    pts = grid.points_m()
    np.testing.assert_allclose(
        cp.asnumpy(grid.center_m),
        0.5 * (cp.asnumpy(pts[0]) + cp.asnumpy(pts[-1])),
        atol=1e-15,
    )


def test_world_to_index_roundtrip_is_integral(cp, dipole_cloud):
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    idx = cp.asnumpy(grid.world_to_index(grid.points_m()))
    assert idx.shape == (3, grid.n_points)
    assert cp.ascontiguousarray(grid.world_to_index(grid.points_m())).flags.c_contiguous

    expected = np.stack(np.unravel_index(np.arange(grid.n_points), grid.shape))
    np.testing.assert_allclose(idx, expected, atol=1e-9)


def test_map_coordinates_on_linear_field_is_exact(cp):
    """Trilinear interpolation is exact for a·x + b, so any axis mix-up shows up immediately."""
    from cupyx.scipy.ndimage import map_coordinates

    grid = Grid(
        origin_m=cp.asarray([-0.02, 0.05, 0.11]),
        spacing_m=cp.asarray([2e-3, 3e-3, 5e-3]),
        shape=(9, 7, 5),
    )
    a = np.array([3.0, -7.0, 11.0])
    b = 0.5

    pts = cp.asnumpy(grid.points_m())
    field = cp.asarray((pts @ a + b).reshape(grid.shape))

    rng = np.random.default_rng(2)
    lo = cp.asnumpy(grid.origin_m)
    hi = cp.asnumpy(grid.upper_m)
    query = rng.uniform(lo, hi, size=(500, 3))

    got = cp.asnumpy(
        map_coordinates(field, grid.world_to_index(cp.asarray(query)), order=1, mode="nearest")
    )
    np.testing.assert_allclose(got, query @ a + b, rtol=1e-6)


def test_upper_m_is_the_last_sample_on_each_axis(cp, dipole_cloud):
    """``max_index`` is the bound the evaluate check compares against, so it must be a real sample."""
    grid = build_grid(dipole_cloud, spacing_mm=3.0, margin_mm=8.0)
    idx = grid.world_to_index(grid.upper_m[None, :])
    np.testing.assert_allclose(cp.asnumpy(idx).ravel(), cp.asnumpy(grid.max_index), atol=1e-9)


def test_outside_grid_clamps_to_the_edge(cp):
    """The hazard ``_require_in_grid`` exists to catch: clamping, not an error, past the edge."""
    from cupyx.scipy.ndimage import map_coordinates

    grid = Grid(
        origin_m=cp.asarray([0.0, 0.0, 0.0]),
        spacing_m=cp.asarray([1e-3, 1e-3, 1e-3]),
        shape=(4, 4, 4),
    )
    field = cp.asarray(
        (cp.asnumpy(grid.points_m()) @ np.array([1.0, 0.0, 0.0])).reshape(grid.shape)
    )
    inside = cp.asarray([[3e-3, 0.0, 0.0]])
    outside = cp.asarray([[3e-3 + 3 * 1e-3, 0.0, 0.0]])
    edge = float(
        map_coordinates(field, grid.world_to_index(inside), order=1, mode="nearest")[0]
    )
    clamped = float(
        map_coordinates(field, grid.world_to_index(outside), order=1, mode="nearest")[0]
    )
    assert clamped == edge


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


def test_place_batch_agrees_with_the_dadt_placement_math(cp, cube_subject, synthetic_coil):
    """``adm/place.py`` and the inline placement in ``coil_dadt_at_nodes`` must not drift apart."""
    from cunibs.fem.placement import compute_coil_transform

    tf = compute_coil_transform(cube_subject.context, [50, 50, 120], [50, 150, 100], 4.0)
    s, m = place_coil_dipoles_batch(
        cp.asarray(tf[None]), synthetic_coil.positions_m, synthetic_coil.moments
    )
    rot, trans = tf[:3, :3], tf[:3, 3]
    inline_s = (synthetic_coil.positions_m * 1e3 @ rot.T + trans) * 1e-3
    inline_m = synthetic_coil.moments @ rot.T
    np.testing.assert_allclose(cp.asnumpy(s[0]), inline_s, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(m[0]), inline_m, atol=1e-12)


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

    lo = cp.asnumpy(grid.origin_m)
    hi = cp.asnumpy(grid.upper_m)
    pts = cp.asnumpy(s).reshape(-1, 3)
    assert np.all(pts >= lo) and np.all(pts <= hi)


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

    lo = cp.asnumpy(grid.origin_m)
    hi = cp.asnumpy(grid.upper_m)
    pts = cp.asnumpy(s).reshape(-1, 3)
    assert np.all(pts >= lo) and np.all(pts <= hi)
