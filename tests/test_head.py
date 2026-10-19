"""Derived head geometry and the CHARM companion files, in cunibs.head."""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.head import eeg

# --- geometry ---------------------------------------------------------------------------------


def _mesh_with_tagged_interface(two_region_mesh, z0):
    """The two-region cube, with its flat interface exposed as a gray-matter surface.

    The fixture's interface is conforming and exactly planar, so the distance from the top face
    down to it is known to the bit: that is what makes this a ground truth rather than a
    self-consistency check.
    """
    from cunibs.mesh import HeadMesh

    nodes = two_region_mesh.nodes_mm
    tets = two_region_mesh.tet_nodes
    faces = tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    on_plane = np.isclose(nodes[faces][:, :, 2], z0).all(axis=1)
    interface = np.unique(np.sort(faces[on_plane], axis=1), axis=0).astype(np.int32)
    assert interface.shape[0] > 0

    skin = two_region_mesh.skin_tris
    surf_tris = np.vstack([skin, interface]).astype(np.int32)
    surf_tags = np.concatenate(
        [np.full(len(skin), 1005), np.full(len(interface), 1002)]
    ).astype(np.int32)
    return HeadMesh(nodes, tets, two_region_mesh.tet_tags, surf_tris, surf_tags)


@pytest.mark.gpu
def test_depth_to_surface_is_exact_on_a_flat_interface(two_region_mesh, two_region_z0):
    """A planar interface at a known depth is the one case with an answer to check against."""
    from cunibs.fem import build_context
    from cunibs.head import depth_to_surface_mm

    ctx = build_context(_mesh_with_tagged_interface(two_region_mesh, two_region_z0))
    # Points above the top face; each projects straight down onto it.
    xy = np.array([[30.0, 30.0], [50.0, 50.0], [70.0, 40.0]])
    pts = np.column_stack([xy, np.full(len(xy), 130.0)])

    depth = depth_to_surface_mm(ctx, pts, "gray_matter")
    np.testing.assert_allclose(depth, 100.0 - two_region_z0, atol=1e-9)


@pytest.mark.gpu
def test_depth_is_measured_from_the_scalp_not_from_the_query_point(
    two_region_mesh, two_region_z0
):
    """Moving the query point along its own normal must not change the distance.

    The quantity is a property of the head, not of where the caller happened to stand, so the
    projection has to be what the ray starts from.
    """
    from cunibs.fem import build_context
    from cunibs.head import depth_to_surface_mm

    ctx = build_context(_mesh_with_tagged_interface(two_region_mesh, two_region_z0))
    near = np.array([[50.0, 50.0, 105.0]])
    far = np.array([[50.0, 50.0, 400.0]])
    assert depth_to_surface_mm(ctx, near) == pytest.approx(depth_to_surface_mm(ctx, far))


@pytest.mark.gpu
def test_thickness_differences_two_depths_along_one_ray(two_region_mesh, two_region_z0):
    from cunibs.fem import build_context
    from cunibs.head import tissue_depths_mm, tissue_thickness_mm

    ctx = build_context(_mesh_with_tagged_interface(two_region_mesh, two_region_z0))
    pts = np.array([[50.0, 50.0, 130.0]])
    depths = tissue_depths_mm(ctx, pts, ["gray_matter"])
    assert depths["gray_matter"][0] == pytest.approx(100.0 - two_region_z0)
    # Same surface twice is a zero-thickness layer, which pins the sign convention.
    assert tissue_thickness_mm(ctx, pts, "gray_matter", "gray_matter")[0] == pytest.approx(0.0)


@pytest.mark.gpu
def test_missing_surface_is_nan_rather_than_a_plausible_number(two_region_mesh, two_region_z0):
    """A ray that never crosses the surface must not come back as a deep target."""
    from cunibs.fem import build_context
    from cunibs.head import depth_to_surface_mm

    ctx = build_context(_mesh_with_tagged_interface(two_region_mesh, two_region_z0))
    # Projects onto a side wall, so the inward normal runs parallel to the interface plane.
    depth = depth_to_surface_mm(ctx, np.array([[130.0, 50.0, 75.0]]))
    assert np.isnan(depth[0])


@pytest.mark.gpu
@pytest.mark.reference
@pytest.mark.slow
def test_scalp_to_cortex_at_real_electrode_sites(reference_mesh_path):
    """Real anatomy at the sites this quantity is actually quoted for.

    Reported scalp-to-cortex distances run roughly 10-25 mm at the lateral 10-20 targets, which a
    synthetic interface at a known depth cannot check. The midline is deliberately excluded: the
    inward normal at Cz runs down the interhemispheric fissure before reaching cortex, so its
    distance is genuinely much larger, and sampling only the top of the head measures that rather
    than scalp-to-cortex distance. Needs the full mesh; the cropped patch has only the scalp.
    """
    from cunibs import Subject
    from cunibs.head import find_cap, read_cap, scalp_to_cortex_mm

    cap = read_cap(find_cap(reference_mesh_path.parent))
    lateral = [cap[k] for k in ("C3", "C4", "F3", "P3")]

    with Subject.from_mesh(reference_mesh_path) as subject:
        assert 1002 in subject.context.mesh.surface_tags
        scd = scalp_to_cortex_mm(subject.context, np.array(lateral))
        midline = scalp_to_cortex_mm(subject.context, np.array([cap["Cz"]]))

    assert np.isfinite(scd).all()
    assert (scd > 8.0).all() and (scd < 25.0).all(), scd
    # The fissure is the reason the midline is excluded above, so pin that it really is different.
    assert midline[0] > scd.max()


@pytest.mark.gpu
@pytest.mark.realmesh
def test_a_mesh_without_the_surface_says_so(patch_subject):
    """The cropped patch carries only the scalp, and the error has to name what is there."""
    from cunibs.head import scalp_to_cortex_mm

    ctx = patch_subject.context
    assert ctx.mesh.surface_tags == (1005,)
    with pytest.raises(KeyError, match="no surface with tag 1002"):
        scalp_to_cortex_mm(ctx, np.array([[0.0, 0.0, 0.0]]))


# --- EEG cap ----------------------------------------------------------------------------------


_CAP = """\
Electrode,-20.9,109.6,19.8,Fp1
Electrode,-44.0,94.7,17.8,AF7
ReferenceElectrode,3.0,2.1,107.4,Cz
Fiducial,0.5,120.0,0.0,Nz
Fiducial,-80.0,0.0,-40.0,LPA
"""


def test_read_cap_keeps_the_reference_electrode(tmp_path):
    """Cz is a ReferenceElectrode row in every CHARM cap file, not an Electrode row.

    Filtering on "Electrode" therefore drops the vertex silently and reports it as absent, and Cz
    is one of the most commonly used TMS targets there is.
    """
    path = tmp_path / "cap.csv"
    path.write_text(_CAP)

    positions = eeg.read_cap(path)
    assert "Cz" in positions
    np.testing.assert_allclose(positions["Cz"], [3.0, 2.1, 107.4])
    assert set(positions) == {"Fp1", "AF7", "Cz"}


def test_fiducials_are_not_electrodes(tmp_path):
    """They are registration landmarks, so a target sweep must not pick them up as sites."""
    path = tmp_path / "cap.csv"
    path.write_text(_CAP)

    assert set(eeg.read_fiducials(path)) == {"Nz", "LPA"}
    assert not set(eeg.read_cap(path)) & {"Nz", "LPA"}


def test_a_file_with_no_electrodes_is_an_error_not_an_empty_dict(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Fiducial,0,0,0,Nz\n")
    with pytest.raises(ValueError, match="no electrode rows"):
        eeg.read_cap(path)


def test_find_cap_names_what_is_available(tmp_path):
    root = tmp_path / "m2m_sub-001" / "eeg_positions"
    root.mkdir(parents=True)
    (root / "EEG10-20_Okamoto_2004.csv").write_text(_CAP)
    with pytest.raises(FileNotFoundError, match="Okamoto"):
        eeg.find_cap(tmp_path / "m2m_sub-001")
    assert eeg.find_cap(tmp_path / "m2m_sub-001", "EEG10-20_Okamoto_2004").exists()


# --- candidate grid ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.realmesh
def test_scalp_grid_candidates_lie_on_the_scalp_inside_the_radius(patch_subject, patch_top_mm):
    """Both properties together are why there is no second clip.

    A tangent-plane raster projected onto a curved head lands outside the requested radius and off
    the surface, which is what forces the usual construction to discard twice. Sampling the scalp
    means the radius test is the disc and the candidates are already where they belong.
    """
    from cunibs.head import scalp_grid
    from cunibs.head.geometry import project_to_scalp

    ctx = patch_subject.context
    grid = scalp_grid(ctx, patch_top_mm, radius_mm=12.0, spacing_mm=3.0)
    assert len(grid) > 4

    offset = np.linalg.norm(grid.centers_mm - grid.center_mm, axis=1)
    assert offset.max() <= 12.0 + 1e-9

    projected, _ = project_to_scalp(patch_subject, grid.centers_mm)
    assert isinstance(projected, np.ndarray)
    assert np.abs(projected - grid.centers_mm).max() < 1e-6


@pytest.mark.gpu
@pytest.mark.realmesh
def test_scalp_grid_spacing_tracks_the_request(patch_subject, patch_top_mm):
    """A coarser request must return fewer, further-apart candidates, not the same set."""
    from cunibs.head import scalp_grid

    ctx = patch_subject.context
    fine = scalp_grid(ctx, patch_top_mm, radius_mm=12.0, spacing_mm=1.5)
    coarse = scalp_grid(ctx, patch_top_mm, radius_mm=12.0, spacing_mm=4.0)
    assert len(fine) > len(coarse)
    assert fine.achieved_spacing_mm < coarse.achieved_spacing_mm


@pytest.mark.gpu
@pytest.mark.realmesh
def test_scalp_grid_subdivides_past_the_facet_size(patch_subject, patch_top_mm):
    """The mesh's own facets are the floor, and subdividing them is what lifts it.

    Without this a request finer than the scalp triangles would silently return one candidate per
    facet, so a coarse mesh would cap the search resolution.
    """
    from cunibs.head import scalp_grid

    ctx = patch_subject.context
    grid = scalp_grid(ctx, patch_top_mm, radius_mm=12.0, spacing_mm=0.5)
    assert len(grid) > grid.n_source_triangles
    assert grid.achieved_spacing_mm < 1.0


@pytest.mark.gpu
@pytest.mark.realmesh
def test_scalp_grid_feeds_optimize_directly(patch_subject, patch_mesh, d70_coil, patch_top_mm):
    """It exists to be passed to a search, so it has to be accepted as centers_mm unchanged."""
    from cunibs.adm import Target, optimize
    from cunibs.head import scalp_grid

    gm = patch_mesh.tet_barycenters_mm[patch_mesh.tet_tags == 2]
    target = gm[np.argmax(gm[:, 2])] - np.array([0.0, 0.0, 4.0])

    ctx = patch_subject.context
    grid = scalp_grid(ctx, patch_top_mm, radius_mm=8.0, spacing_mm=4.0)
    result = optimize(ctx, d70_coil, Target(target, region="gray_matter"), grid, spacing_mm=4.0)
    assert result.centers_mm.shape[0] == len(grid)
    assert np.isfinite(result.best_objective) and result.best_objective > 0.0


def test_thickness_with_the_surfaces_swapped_raises(monkeypatch):
    from cunibs.head import geometry

    monkeypatch.setattr(
        geometry,
        "tissue_depths_mm",
        lambda *_: {
            "csf": np.array([9.0, np.nan]),
            "cortical_bone": np.array([6.0, 7.0]),
        },
    )
    assert geometry.tissue_thickness_mm(None, None, "cortical_bone", "csf")[0] == 3.0
    with pytest.raises(ValueError, match="swapped"):
        geometry.tissue_thickness_mm(None, None, "csf", "cortical_bone")
