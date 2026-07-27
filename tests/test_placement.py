"""Coil placement: the scalp projection, the coil-to-head frame, and the dA/dt N-body.

The frame convention under test (``fem/placement.py``) is columns ``[x | y | z | c]`` with
``y`` along the handle, ``z`` pointing inward (``-normal``), ``x = y × z``, and ``c`` offset
from the projected scalp point by ``distance_mm`` along the outward normal.

``solver/place.cu`` is validated against a NumPy transcription of the same Ericson
closest-point-on-triangle ladder; the N-body against the closed-form dipole vector potential.
"""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.fem.placement import MU0_OVER_4PI, compute_coil_transform, compute_coil_transforms

pytestmark = pytest.mark.gpu


def closest_on_tri(p, a, b, c):
    """Ericson's closest point on a triangle — the reference for ``place.cu``'s branch ladder."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp_ = p - c
    d5, d6 = ab @ cp_, ac @ cp_
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + (vb * denom) * ab + (vc * denom) * ac


def project_to_skin(mesh, point):
    """Nearest point on the skin surface, ties broken by lowest triangle index."""
    tris = mesh.nodes_mm[mesh.skin_tris]
    best_d2, best_tri, best_q = np.inf, -1, None
    for j, (a, b, c) in enumerate(tris):
        q = closest_on_tri(np.asarray(point, float), a, b, c)
        d2 = float((q - point) @ (q - point))
        if d2 < best_d2:
            best_d2, best_tri, best_q = d2, j, q
    return best_q, best_tri


def dadt_reference(positions_m, moments, transform, didt, targets_mm):
    """A(r) = (μ₀/4π) Σⱼ mⱼ × (r − sⱼ)/|r − sⱼ|³, times dI/dt — all in float64."""
    rot, trans = transform[:3, :3], transform[:3, 3]
    s = (np.asarray(positions_m) * 1e3 @ rot.T + trans) * 1e-3
    m = np.asarray(moments) @ rot.T
    r = np.asarray(targets_mm, dtype=np.float64) * 1e-3
    diff = r[:, None, :] - s[None, :, :]
    dist3 = np.linalg.norm(diff, axis=2)[:, :, None] ** 3
    return didt * MU0_OVER_4PI * (np.cross(m[None, :, :], diff) / dist3).sum(axis=1)


def rel_l2(got, ref) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def test_transform_frame_is_orthonormal_right_handed(cube_subject):
    tf = compute_coil_transform(cube_subject.context, [50, 50, 100], [50, 150, 100], 4.0)
    r = tf[:3, :3]
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(tf[3], [0, 0, 0, 1])


def test_transform_columns_follow_the_documented_convention(cube_subject, cube_mesh):
    """[x | y | z | c] with y = in-plane handle, z = −normal, x = y × z, c = proj + d·normal.

    The reference normal is read back from the mesh rather than assumed: skin normals are
    smoothed across the whole closed surface, so even a cube's face normals are not axis
    aligned.
    """
    center, handle, dist = np.array([50.0, 50.0, 120.0]), np.array([50.0, 150.0, 100.0]), 4.0
    tf = compute_coil_transform(cube_subject.context, center, handle, dist)

    proj, tri = project_to_skin(cube_mesh, center)
    normal = cube_mesh.skin_triangle_normals[tri]

    x, y, z, c = tf[:3, 0], tf[:3, 1], tf[:3, 2], tf[:3, 3]
    np.testing.assert_allclose(z, -normal, atol=1e-12)
    np.testing.assert_allclose(c, proj + dist * normal, atol=1e-11)

    expected_y = handle - proj
    expected_y -= z * (expected_y @ z)
    expected_y /= np.linalg.norm(expected_y)
    np.testing.assert_allclose(y, expected_y, atol=1e-12)
    np.testing.assert_allclose(x, np.cross(y, z), atol=1e-12)


def test_transform_orthogonalizes_the_handle(cube_subject):
    """A handle far out of the tangent plane is projected onto it, keeping its in-plane sign."""
    tf = compute_coil_transform(cube_subject.context, [50, 50, 120], [50, 150, 900], 4.0)
    y, z = tf[:3, 1], tf[:3, 2]
    assert abs(float(y @ z)) < 1e-14
    assert np.linalg.norm(y) == pytest.approx(1.0, abs=1e-14)
    assert y[1] > 0  # still points along +y, the in-plane part of the handle


def test_transform_distance_offsets_along_the_outward_normal(cube_subject, cube_mesh):
    """distance_mm slides the origin along the outward normal and rotates nothing."""
    ctx = cube_subject.context
    center = np.array([50.0, 50.0, 120.0])
    near = compute_coil_transform(ctx, center, [50, 150, 100], 0.0)
    far = compute_coil_transform(ctx, center, [50, 150, 100], 10.0)

    proj, tri = project_to_skin(cube_mesh, center)
    normal = cube_mesh.skin_triangle_normals[tri]
    np.testing.assert_allclose(near[:3, 3], proj, atol=1e-11)
    np.testing.assert_allclose(far[:3, 3] - near[:3, 3], 10.0 * normal, atol=1e-11)
    np.testing.assert_allclose(near[:3, :3], far[:3, :3], atol=1e-14)


@pytest.mark.parametrize(
    "center",
    [
        [50.0, 50.0, 130.0],  # above a face interior
        [50.0, 130.0, 130.0],  # nearest point on an edge
        [130.0, 130.0, 130.0],  # nearest point at a vertex
        [10.0, 90.0, 140.0],  # off-centre, still above the top face
    ],
    ids=["face", "edge", "vertex", "offcentre"],
)
def test_transform_projection_matches_numpy_closest_point(cube_subject, cube_mesh, center):
    """Every branch of the closest-point ladder, against an independent NumPy transcription."""
    tf = compute_coil_transform(cube_subject.context, center, np.add(center, [0, 50, 0]), 4.0)
    expected_q, tri = project_to_skin(cube_mesh, np.asarray(center, float))
    normal = cube_mesh.skin_triangle_normals[tri]
    np.testing.assert_allclose(tf[:3, 3] - 4.0 * normal, expected_q, atol=1e-11)
    np.testing.assert_allclose(tf[:3, 2], -normal, atol=1e-12)


def test_transforms_batched_equals_serial(cube_subject):
    import cupy as cp

    ctx = cube_subject.context
    centers = np.array(
        [[50, 50, 120], [30, 70, 120], [70, 30, 120], [10, 10, 120], [90, 90, 120]]
    )
    handles = centers + np.array([0.0, 50.0, 0.0])
    dists = np.array([4.0, 1.0, 8.0, 0.0, 12.0])

    batched = cp.asnumpy(compute_coil_transforms(ctx, centers, handles, dists))
    for i in range(len(centers)):
        one = compute_coil_transform(ctx, centers[i], handles[i], float(dists[i]))
        np.testing.assert_array_equal(batched[i], one, err_msg=f"placement {i}")


@pytest.mark.realmesh
def test_transform_from_inside_the_head_projects_outward(patch_subject, patch_mesh):
    """A centre buried in gray matter still projects onto the real, non-convex scalp."""
    gm = patch_mesh.tet_barycenters_mm[patch_mesh.tet_tags == 2]
    inside = gm[np.argmin(gm[:, 2])]
    tf = compute_coil_transform(patch_subject.context, inside, inside + [0, 50, 0], 4.0)

    r = tf[:3, :3]
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    proj = tf[:3, 3] + 4.0 * tf[:3, 2]  # z is the inward normal, so this undoes the offset
    expected, _ = project_to_skin(patch_mesh, inside)
    np.testing.assert_allclose(proj, expected, atol=1e-9)


@pytest.mark.realmesh
def test_transform_curved_scalp_offset(patch_subject, patch_mesh):
    """Across 20 real scalp sites the standoff is exactly distance_mm along the outward normal."""
    import cupy as cp

    centroids = patch_mesh.nodes_mm[patch_mesh.skin_tris].mean(axis=1)
    centers = centroids[:: max(len(centroids) // 20, 1)][:20]
    tfs = cp.asnumpy(
        compute_coil_transforms(
            patch_subject.context,
            centers,
            centers + [0.0, 50.0, 0.0],
            np.full(len(centers), 4.0),
        )
    )
    for i, tf in enumerate(tfs):
        proj = tf[:3, 3] + 4.0 * tf[:3, 2]
        offset = tf[:3, 3] - proj
        assert np.linalg.norm(offset) == pytest.approx(4.0, rel=1e-12), f"site {i}"
        assert float(offset @ -tf[:3, 2]) > 0, f"site {i}"  # outward, not inward
        np.testing.assert_allclose(tf[:3, :3].T @ tf[:3, :3], np.eye(3), atol=1e-12)


def _identity_transform(translation=(0.0, 0.0, 0.0)):
    tf = np.eye(4)
    tf[:3, 3] = translation
    return tf


def test_dadt_matches_analytic_dipole(cp, cube_subject):
    """The fp32 N-body against the closed-form vector potential, over the cube's nodes."""
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = cube_subject.context
    positions = np.array([[0.0, 0.0, 0.0]])
    moments = np.array([[0.0, 0.0, 1.0]])
    tf = _identity_transform((50.0, 50.0, 150.0))

    got = cp.asnumpy(coil_dadt_at_nodes(positions, moments, tf, 1e6, ctx.nodes_mm))
    ref = dadt_reference(positions, moments, tf, 1e6, cp.asnumpy(ctx.nodes_mm))
    assert rel_l2(got.astype(np.float64), ref) <= 1e-5


def test_dadt_two_dipole_superposition(cp, cube_subject, synthetic_coil):
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = cube_subject.context
    tf = _identity_transform((50.0, 50.0, 150.0))
    both = cp.asnumpy(
        coil_dadt_at_nodes(
            synthetic_coil.positions_m, synthetic_coil.moments, tf, 1e6, ctx.nodes_mm
        )
    )
    parts = sum(
        cp.asnumpy(
            coil_dadt_at_nodes(
                synthetic_coil.positions_m[i : i + 1],
                synthetic_coil.moments[i : i + 1],
                tf,
                1e6,
                ctx.nodes_mm,
            )
        ).astype(np.float64)
        for i in range(2)
    )
    assert rel_l2(both.astype(np.float64), parts) <= 1e-6


def test_dadt_is_linear_in_didt(cp, cube_subject, synthetic_coil):
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = cube_subject.context
    tf = _identity_transform((50.0, 50.0, 150.0))
    args = (synthetic_coil.positions_m, synthetic_coil.moments, tf)
    one = cp.asnumpy(coil_dadt_at_nodes(*args, 1e6, ctx.nodes_mm))
    two = cp.asnumpy(coil_dadt_at_nodes(*args, 2e6, ctx.nodes_mm))
    np.testing.assert_allclose(two, 2.0 * one, rtol=1e-6)


def test_dadt_zero_didt_and_zero_moment_are_zero(cp, cube_subject, synthetic_coil):
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = cube_subject.context
    tf = _identity_transform((50.0, 50.0, 150.0))
    zero_didt = coil_dadt_at_nodes(
        synthetic_coil.positions_m, synthetic_coil.moments, tf, 0.0, ctx.nodes_mm
    )
    zero_moment = coil_dadt_at_nodes(
        synthetic_coil.positions_m, np.zeros_like(synthetic_coil.moments), tf, 1e6, ctx.nodes_mm
    )
    assert float(cp.abs(zero_didt).max()) == 0.0
    assert float(cp.abs(zero_moment).max()) == 0.0


def test_dadt_applies_rotation_to_moments_not_translation(cp, cube_subject):
    """A pure translation must move the sources and leave the moment vectors untouched."""
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = cube_subject.context
    positions = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    moments = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    tf = _identity_transform((50.0, 50.0, 150.0))

    got = cp.asnumpy(coil_dadt_at_nodes(positions, moments, tf, 1e6, ctx.nodes_mm))
    # Reference built by hand: sources shifted, moments verbatim.
    shifted = positions + np.array([0.05, 0.05, 0.15])
    ref = dadt_reference(shifted, moments, np.eye(4), 1e6, cp.asnumpy(ctx.nodes_mm))
    assert rel_l2(got.astype(np.float64), ref) <= 1e-5


def test_dadt_rotation_is_applied_to_moments(cp, cube_subject):
    """A 90° rotation about x must carry m = ẑ to ŷ, which the analytic oracle checks directly."""
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = cube_subject.context
    tf = np.eye(4)
    tf[:3, :3] = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    tf[:3, 3] = [50.0, 50.0, 150.0]
    positions = np.array([[0.02, 0.0, 0.0]])
    moments = np.array([[0.0, 0.0, 1.0]])

    got = cp.asnumpy(coil_dadt_at_nodes(positions, moments, tf, 1e6, ctx.nodes_mm))
    ref = dadt_reference(positions, moments, tf, 1e6, cp.asnumpy(ctx.nodes_mm))
    np.testing.assert_allclose(
        np.asarray(moments) @ tf[:3, :3].T, [[0.0, -1.0, 0.0]], atol=1e-15
    )
    assert rel_l2(got.astype(np.float64), ref) <= 1e-5


@pytest.mark.realmesh
def test_dadt_matches_analytic_on_real_nodes(cp, patch_subject, patch_placement, d70_coil):
    """8.4k real nodes and a 3000-dipole coil: the fp64-centre / fp32-N-body split at scale."""
    from cunibs.fem.placement import coil_dadt_at_nodes

    ctx = patch_subject.context
    tf = compute_coil_transform(
        ctx, patch_placement.center_mm, patch_placement.handle_mm, patch_placement.distance_mm
    )
    got = cp.asnumpy(
        coil_dadt_at_nodes(d70_coil.positions_m, d70_coil.moments, tf, 1e6, ctx.nodes_mm)
    )
    ref = dadt_reference(
        d70_coil.positions_m, d70_coil.moments, tf, 1e6, cp.asnumpy(ctx.nodes_mm)
    )
    assert rel_l2(got.astype(np.float64), ref) <= 1e-5
