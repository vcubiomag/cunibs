"""Patch-recovery post-processing: the operator and its invariants.

The oracle for ``"spr_global"`` is a NumPy transcription of SimNIBS's ``elm_data2node_data``,
written in its absolute-coordinate basis rather than the centred-and-scaled one the kernel uses.
The two are the same fit under an affine change of basis, so agreeing to fp32 roundoff is what
shows the reparameterisation is sound.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _numpy_spr(nodes, tets, values, is_boundary, tags=None, slots=None):
    """SimNIBS's recovery, transcribed: a linear fit per patch, volume average on the boundary.

    Pass ``tags`` and a list of ``(node, tag)`` ``slots`` for the tissue-restricted path, which
    SimNIBS reaches by cropping to one tag before recovering: the patch is then that tag's
    incident tetrahedra alone, and ``is_boundary`` is the cropped volume's boundary rather than
    the whole mesh's. Without them a slot is a node and every incident tetrahedron counts.
    """
    n_nodes = nodes.shape[0]
    bary = nodes[tets].mean(axis=1)
    vol = np.abs(np.linalg.det(nodes[tets][:, 1:] - nodes[tets][:, :1])) / 6.0
    incident: list[list[int]] = [[] for _ in range(n_nodes)]
    for e, tet in enumerate(tets):
        for node in tet:
            incident[node].append(e)
    if slots is None:
        slots = [(node, None) for node in range(n_nodes)]

    out = np.zeros((len(slots), 3))
    for s, (node, tag) in enumerate(slots):
        ets = np.array(incident[node], dtype=np.int64)
        if tag is not None:
            ets = ets[tags[ets] == tag]
        if ets.size == 0:
            continue
        if is_boundary[node] or ets.size < 4:
            out[s] = (values[ets] * vol[ets, None]).sum(0) / vol[ets].sum()
            continue
        design = np.hstack([np.ones((ets.size, 1)), bary[ets]])
        coef = np.linalg.solve(design.T @ design, design.T @ values[ets])
        out[s] = np.hstack([[1.0], nodes[node]]) @ coef
    return out


@pytest.fixture(scope="module")
def cube_op(refined_cube_mesh):
    from cunibs.fem import build_context, ensure_recovery

    ctx = build_context(refined_cube_mesh)
    return ctx, ensure_recovery(ctx, "spr_global")


def test_outer_boundary_matches_a_face_count(cp, refined_cube_mesh, boundary_faces):
    """The boundary set is a set, so it must be exact rather than merely well-ordered."""
    from cunibs.fem import build_context, outer_boundary_nodes

    mesh = build_context(refined_cube_mesh).mesh
    tets = cp.asarray(mesh.tet_nodes)
    got = cp.asnumpy(outer_boundary_nodes(tets, mesh.n_nodes)).astype(bool)
    expected = np.zeros(mesh.n_nodes, dtype=bool)
    expected[np.unique(boundary_faces(mesh.tet_nodes))] = True
    np.testing.assert_array_equal(got, expected)
    assert 0 < expected.sum() < mesh.n_nodes, "fixture must have both kinds of node"


def test_weights_are_a_partition_of_unity(cp, cube_op):
    """sum_c w_c == 1 per patch, for the fit and the volume-average fallback alike.

    It follows from A e_0 = sum_e p_e, the first basis component being identically one, so a
    break here means the normal matrix or its solve is wrong rather than merely inaccurate.
    """
    _, op = cube_op
    w, ptr = cp.asnumpy(op.w), cp.asnumpy(op.ptr)
    sums = np.add.reduceat(w, ptr[:-1])
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_recovery_reproduces_a_linear_field(cp, cube_op):
    """A linear field is in the fit space, so an interior patch must return it exactly.

    Checked at the nodes and again after sampling back at the barycentres. Boundary nodes take a
    volume-weighted average, which is deliberately not linear-exact, so the element check is
    restricted to tetrahedra all of whose corners are interior.
    """
    from cunibs.fem import apply_recovery, outer_boundary_nodes

    ctx, op = cube_op
    nodes, tets = ctx.mesh.nodes_mm, ctx.mesh.tet_nodes
    bary = nodes[tets].mean(axis=1)
    rng = np.random.default_rng(0)
    offset, slope = rng.normal(size=3), rng.normal(size=(3, 3))
    field = (offset + bary @ slope.T).astype(np.float32)
    scale = np.abs(offset + bary @ slope.T).max()

    rec = apply_recovery(op, tets.shape[0], elements=[cp.asarray(field)])
    slots, elems = rec.E_slots, rec.E
    interior = ~cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, nodes.shape[0])).astype(bool)
    inner_tet = interior[tets].all(axis=1)
    assert interior.sum() and inner_tet.sum()

    at_nodes = cp.asnumpy(slots[0])[interior]
    np.testing.assert_allclose(at_nodes, offset + nodes[interior] @ slope.T, atol=1e-5 * scale)
    at_bary = cp.asnumpy(elems[0])[inner_tet]
    np.testing.assert_allclose(at_bary, offset + bary[inner_tet] @ slope.T, atol=1e-5 * scale)


def test_matches_a_numpy_transcription_of_simnibs(cp, cube_op):
    """The kernel's centred, scaled fit equals SimNIBS's absolute-coordinate one."""
    from cunibs.fem import apply_recovery, outer_boundary_nodes

    ctx, op = cube_op
    nodes, tets = ctx.mesh.nodes_mm, ctx.mesh.tet_nodes
    rng = np.random.default_rng(1)
    field = rng.normal(size=(tets.shape[0], 3)).astype(np.float32)

    slots = apply_recovery(op, tets.shape[0], elements=[cp.asarray(field)]).E_slots
    is_boundary = cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, nodes.shape[0])).astype(bool)
    expected = _numpy_spr(nodes, tets.astype(np.int64), field.astype(np.float64), is_boundary)
    np.testing.assert_allclose(cp.asnumpy(slots[0]), expected, atol=2e-5, rtol=2e-5)


def test_boundary_nodes_take_the_volume_weighted_average(cp, cube_op):
    """The boundary rule is SimNIBS's, so it is pinned separately from the fit."""
    from cunibs.fem import apply_recovery, outer_boundary_nodes

    ctx, op = cube_op
    nodes, tets = ctx.mesh.nodes_mm, ctx.mesh.tet_nodes
    rng = np.random.default_rng(2)
    field = rng.normal(size=(tets.shape[0], 3)).astype(np.float32)
    slots = apply_recovery(op, tets.shape[0], elements=[cp.asarray(field)]).E_slots

    vol = np.abs(np.linalg.det(nodes[tets][:, 1:] - nodes[tets][:, :1])) / 6.0
    is_boundary = cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, nodes.shape[0])).astype(bool)
    got = cp.asnumpy(slots[0])
    for node in np.flatnonzero(is_boundary)[::7]:
        ets = np.flatnonzero((tets == node).any(axis=1))
        want = (field[ets].astype(np.float64) * vol[ets, None]).sum(0) / vol[ets].sum()
        np.testing.assert_allclose(got[node], want, atol=1e-5, rtol=1e-5)


def test_operator_build_is_bit_reproducible(cp, refined_cube_mesh):
    from cunibs.fem import build_context, ensure_recovery

    first = cp.asnumpy(ensure_recovery(build_context(refined_cube_mesh), "spr_global").w)
    for _ in range(2):
        again = cp.asnumpy(ensure_recovery(build_context(refined_cube_mesh), "spr_global").w)
        np.testing.assert_array_equal(first, again)


def test_apply_is_bit_reproducible_and_block_width_invariant(cp, cube_op):
    """A placement's recovered field must not depend on who shared its block.

    The per-slot walk order is fixed by ptr/idx and does not depend on the compiled width, so
    every column has to come back byte for byte identical at every k.
    """
    from cunibs.fem import apply_recovery

    ctx, op = cube_op
    n_tet = ctx.mesh.tet_nodes.shape[0]
    rng = np.random.default_rng(3)
    fields = [cp.asarray(rng.normal(size=(n_tet, 3)).astype(np.float32)) for _ in range(8)]

    reference = [cp.asnumpy(a) for a in apply_recovery(op, n_tet, elements=fields).E]
    for _ in range(2):
        repeat = [cp.asnumpy(a) for a in apply_recovery(op, n_tet, elements=fields).E]
        for got, want in zip(repeat, reference, strict=True):
            np.testing.assert_array_equal(got, want)
    for k in range(1, 9):
        narrowed = apply_recovery(op, n_tet, elements=fields[:k]).E
        for column in range(k):
            np.testing.assert_array_equal(
                cp.asnumpy(narrowed[column]),
                reference[column],
                err_msg=f"k={k}, column={column}",
            )


def test_unknown_mode_and_nodal_without_recovery_are_rejected(
    refined_cube_subject, figure8_coil
):
    from cunibs import Placement

    placement = Placement([50.0, 50.0, 130.0], [50.0, 100.0, 130.0], 4.0)
    with pytest.raises(ValueError, match="Unknown recovery"):
        refined_cube_subject.simulate(figure8_coil, placement, 1e6, recovery="nope")
    with pytest.raises(ValueError, match="Unknown recovery"):
        refined_cube_subject.simulate(figure8_coil, placement, 1e6, recovery="spr-tissue")
    with pytest.raises(ValueError, match="nodal=True needs a recovery"):
        refined_cube_subject.simulate(figure8_coil, placement, 1e6, nodal=True, recovery="raw")


def test_tier2_rejects_a_mode_name(refined_cube_mesh):
    """The FEM layer takes the built operator, so a mode string is a type error, not a build."""
    from cunibs.fem import build_context, solve_placements_block

    ctx = build_context(refined_cube_mesh)
    with pytest.raises(TypeError, match="ensure_recovery"):
        solve_placements_block(
            ctx, [[0.0, 0.0, 0.5]], [[0.0, 0.0, 1.0]], [], 1e6, recovery="harmonic"
        )


@pytest.mark.realmesh
def test_harmonic_is_the_default(patch_subject, d70_coil, patch_placement):
    """The default is the mode that converges at an interface, not the cheapest one."""
    default = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, vectors=True
    )
    explicit = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, vectors=True, recovery="harmonic"
    )
    assert default.recovery == "harmonic"
    np.testing.assert_array_equal(default.magnE, explicit.magnE)
    np.testing.assert_array_equal(default.E, explicit.E)


@pytest.mark.realmesh
def test_raw_reports_the_unsmoothed_peak(patch_subject, d70_coil, patch_placement):
    """The raw peak is set by a sliver element, so it sits above every recovered mode's."""
    raw = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, vectors=True, recovery="raw"
    )
    harmonic = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, recovery="harmonic"
    )
    assert raw.recovery == "raw"
    assert raw.peak_magnE() > harmonic.peak_magnE()


@pytest.mark.realmesh
def test_boundary_nodes_are_a_superset_of_the_skin(cp, patch_subject):
    """Skin is surface tag 1005 alone; volume-tag filtering exposes more than that."""
    from cunibs.fem import outer_boundary_nodes

    ctx = patch_subject.context
    is_boundary = cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, ctx.n_nodes)).astype(bool)
    skin = np.unique(patch_subject.mesh.skin_tris)
    assert is_boundary[skin].all()
    assert is_boundary.sum() > skin.size


@pytest.mark.realmesh
def test_spr_smooths_the_peak_and_keeps_provenance(patch_subject, d70_coil, patch_placement):
    """Recovery is meant to suppress the sliver-driven peak the raw field reports."""
    raw = patch_subject.simulate(d70_coil, patch_placement, 1e6, magnitude=True, recovery="raw")
    spr = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, nodal=True, recovery="spr_global"
    )
    assert spr.recovery == "spr_global"
    assert spr.peak_magnE() < raw.peak_magnE()
    assert spr.summary["distribution"]["p99.9"] < raw.summary["distribution"]["p99.9"]
    assert spr.E_slots.shape == (patch_subject.mesh.n_nodes, 3)


@pytest.mark.realmesh
def test_nodal_field_masks_to_the_region(patch_subject, d70_coil, patch_placement):
    result = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, nodal=True, recovery="spr_global"
    )
    everywhere = result.nodal_field("all")
    gray = result.nodal_field("gray_matter")
    assert not np.isnan(everywhere).any()
    touched = ~np.isnan(gray[:, 0])
    assert touched.any() and not touched.all()
    np.testing.assert_array_equal(gray[touched], everywhere[touched])


@pytest.mark.realmesh
@pytest.mark.parametrize("mode", ["spr_global", "spr_tissue", "harmonic"])
def test_nodal_field_is_per_node_for_every_mode(patch_subject, d70_coil, patch_placement, mode):
    """The tissue-slot modes carry more slots than nodes; nodal_field still answers per node."""
    n_nodes = patch_subject.mesh.n_nodes
    result = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, nodal=True, recovery=mode
    )
    if mode != "spr_global":
        assert result.E_slots.shape[0] > n_nodes
    gray = result.nodal_field("gray_matter")
    assert gray.shape == (n_nodes, 3)
    touched = ~np.isnan(gray[:, 0])
    assert touched.any() and not touched.all()


@pytest.mark.realmesh
def test_nodal_field_keeps_the_jump_across_a_tissue_boundary(
    patch_subject, d70_coil, patch_placement
):
    """A tissue-restricted mode is two-valued at a boundary, and each side reads its own slot."""
    result = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, nodal=True, recovery="harmonic"
    )
    gray = result.nodal_field("gray_matter")
    csf = result.nodal_field("csf")
    both = ~np.isnan(gray[:, 0]) & ~np.isnan(csf[:, 0])
    assert both.any(), "no node carries both a gray-matter and a CSF slot"
    # Two-valued at the jump: were the slots collapsed to one value per node these would match.
    assert not np.allclose(gray[both], csf[both])

    # And each side really is its own slot's value.
    from cunibs.metrics import region_tag

    pick = result.slot_tag == region_tag("gray_matter")
    np.testing.assert_array_equal(gray[result.slot_node[pick]], result.E_slots[pick])


@pytest.mark.realmesh
def test_nodal_field_rejects_all_for_two_valued_modes(patch_subject, d70_coil, patch_placement):
    """'all' has no single value at a boundary node, so it is refused rather than guessed."""
    result = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, nodal=True, recovery="harmonic"
    )
    with pytest.raises(ValueError, match="two-valued"):
        result.nodal_field("all")
    # The default region is gray matter, so the common call needs no argument.
    np.testing.assert_array_equal(result.nodal_field(), result.nodal_field("gray_matter"))


@pytest.mark.realmesh
@pytest.mark.parametrize("block_k", [1, 2, 3, 5, 8])
def test_recovered_field_is_block_width_invariant(
    patch_subject, d70_coil, patch_placement, block_k
):
    sites = [patch_placement] * 3
    reference = [
        r.magnE
        for r in patch_subject.iter_simulate(
            d70_coil, sites, 1e6, magnitude=True, recovery="spr_global", block_k=1
        )
    ]
    got = [
        r.magnE
        for r in patch_subject.iter_simulate(
            d70_coil, sites, 1e6, magnitude=True, recovery="spr_global", block_k=block_k
        )
    ]
    for i, (a, b) in enumerate(zip(got, reference, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"block_k={block_k}, placement {i}")


@pytest.mark.realmesh
def test_uq_moments_and_per_draw_arrays_describe_the_recovered_field(
    fresh_subject, patch_mesh, d70_coil, patch_placement
):
    """Every statistic in the UQ loop has to read one field, not a mix of two."""
    from cunibs import ConductivityUQConfig

    subject = fresh_subject(patch_mesh)
    config = ConductivityUQConfig(n_samples=8, tissue_cov={2: 0.1}, seed=0)
    raw = subject.simulate_conductivity_uq(
        d70_coil, patch_placement, config, 1e6, moments=True, recovery="raw"
    )
    spr = subject.simulate_conductivity_uq(
        d70_coil, patch_placement, config, 1e6, moments=True, recovery="spr_global"
    )
    assert raw.recovery == "raw"
    assert spr.recovery == "spr_global"
    # The moment field and the per-draw peaks are separate reductions over the same array, so
    # if only one had been switched they would disagree about which field they describe.
    assert spr.peak_mean_magnE() < raw.peak_mean_magnE()
    assert spr.peak_samples.max() < raw.peak_samples.max()
    np.testing.assert_allclose(spr.peak_mean_magnE(), spr.peak_samples.mean(), rtol=0.15)


def test_tissue_slots_split_each_node_by_tissue(cp, two_tissue_cube_mesh):
    """A (node, tissue) CSR is the node CSR with each segment split, so it must agree on both."""
    from cunibs.fem.recovery import tissue_slots

    tets = cp.asarray(two_tissue_cube_mesh.tet_nodes)
    tags = cp.asarray(two_tissue_cube_mesh.tet_tags)
    n_nodes = two_tissue_cube_mesh.n_nodes
    ptr, idx, slot_of_corner, slot_node, _ = tissue_slots(tets, tags)

    ptr, idx = cp.asnumpy(ptr), cp.asnumpy(idx)
    slot_of_corner, slot_node = cp.asnumpy(slot_of_corner), cp.asnumpy(slot_node)
    host_tets, host_tags = two_tissue_cube_mesh.tet_nodes, two_tissue_cube_mesh.tet_tags

    assert ptr[0] == 0 and ptr[-1] == idx.size == host_tets.size
    np.testing.assert_array_equal(np.sort(idx), np.arange(host_tets.size))
    for slot in range(slot_node.size):
        corners = idx[ptr[slot] : ptr[slot + 1]]
        assert corners.size > 0
        # Every corner of a slot names the same node and the same tissue.
        np.testing.assert_array_equal(host_tets.reshape(-1)[corners], slot_node[slot])
        assert len(set(host_tags[corners >> 2].tolist())) == 1
        np.testing.assert_array_equal(slot_of_corner[corners], slot)
    # This mesh gives every node both tissues at the interface, so it must split somewhere.
    assert slot_node.size > n_nodes


def test_spr_tissue_equals_spr_on_a_single_tissue_mesh(cp, refined_cube_mesh):
    """One tissue means the restriction is vacuous, which pins the two slot constructions."""
    from cunibs.fem import apply_recovery, build_context, ensure_recovery

    ctx = build_context(refined_cube_mesh)
    n_tet = refined_cube_mesh.tet_nodes.shape[0]
    rng = np.random.default_rng(4)
    field = [cp.asarray(rng.normal(size=(n_tet, 3)).astype(np.float32))]

    plain = apply_recovery(ensure_recovery(ctx, "spr_global"), n_tet, elements=field).E[0]
    tissue = apply_recovery(ensure_recovery(ctx, "spr_tissue"), n_tet, elements=field).E[0]
    np.testing.assert_array_equal(cp.asnumpy(plain), cp.asnumpy(tissue))


def _per_tag_boundary(boundary_faces, tets, tags, n_nodes):
    """The union over tags of each cropped submesh's boundary nodes, counted face by face."""
    is_boundary = np.zeros(n_nodes, dtype=bool)
    for tag in np.unique(tags):
        is_boundary[np.unique(boundary_faces(tets[tags == tag]))] = True
    return is_boundary


def _check_tissue_boundary(cp, boundary_faces, tets, tags, n_nodes):
    """Check the mask tag by tag against a crop-and-count oracle.

    One node mask serves every tag, so checking only the union would pass even if the mask were
    right for one tag and wrong for another. Restricting to the nodes a tag touches is what makes
    each tag its own assertion.
    """
    from cunibs.fem import tissue_boundary_nodes
    from cunibs.fem.recovery import tissue_slots

    assert np.unique(tags).size > 1, "fixture must carry more than one tag"
    dev_tets = cp.asarray(tets)
    slot_node = tissue_slots(dev_tets, cp.asarray(tags)).slot_node
    got = cp.asnumpy(tissue_boundary_nodes(dev_tets, n_nodes, slot_node)).astype(bool)

    for tag in np.unique(tags):
        sub = tets[tags == tag]
        touches = np.zeros(n_nodes, dtype=bool)
        touches[np.unique(sub)] = True
        expected = np.zeros(n_nodes, dtype=bool)
        expected[np.unique(boundary_faces(sub))] = True
        np.testing.assert_array_equal(got & touches, expected)


def test_tissue_boundary_matches_a_per_tag_face_count(cp, two_region_mesh, boundary_faces):
    """SimNIBS crops to one tag before asking for the boundary, so the oracle must crop too."""
    _check_tissue_boundary(
        cp,
        boundary_faces,
        two_region_mesh.tet_nodes,
        two_region_mesh.tet_tags,
        two_region_mesh.n_nodes,
    )


@pytest.mark.realmesh
def test_tissue_boundary_matches_a_per_tag_face_count_on_a_head(
    cp, patch_subject, boundary_faces
):
    """A head mesh is where a non-manifold interface would show up, if one existed."""
    mesh = patch_subject.mesh
    _check_tissue_boundary(cp, boundary_faces, mesh.tet_nodes, mesh.tet_tags, mesh.n_nodes)


def test_tissue_boundary_is_a_superset_of_the_outer_boundary(cp, two_region_mesh):
    """The interface term is the whole difference between the two masks."""
    from cunibs.fem import outer_boundary_nodes, tissue_boundary_nodes
    from cunibs.fem.recovery import tissue_slots

    tets = cp.asarray(two_region_mesh.tet_nodes)
    n_nodes = two_region_mesh.n_nodes
    slot_node = tissue_slots(tets, cp.asarray(two_region_mesh.tet_tags)).slot_node

    outer = cp.asnumpy(outer_boundary_nodes(tets, n_nodes)).astype(bool)
    tissue = cp.asnumpy(tissue_boundary_nodes(tets, n_nodes, slot_node)).astype(bool)
    assert (tissue | outer == tissue).all()
    assert tissue.sum() > outer.sum(), "fixture must have interior interface nodes"


def test_spr_tissue_matches_a_numpy_transcription_of_simnibs(
    cp, two_region_mesh, boundary_faces
):
    """The tissue path end to end, against a transcription that crops the way SimNIBS does.

    The oracle's boundary set comes from a per-tag face count rather than from the mask under
    test, so this pins the boundary rule and the fit together.
    """
    from cunibs.fem import apply_recovery, build_context, ensure_recovery

    ctx = build_context(two_region_mesh)
    op = ensure_recovery(ctx, "spr_tissue")
    # ctx.mesh, not the fixture: build_context renumbers, and that ordering is what op indexes.
    nodes, tets, tags = ctx.mesh.nodes_mm, ctx.mesh.tet_nodes, ctx.mesh.tet_tags
    rng = np.random.default_rng(11)
    field = rng.normal(size=(tets.shape[0], 3)).astype(np.float32)

    got = cp.asnumpy(apply_recovery(op, tets.shape[0], elements=[cp.asarray(field)]).E_slots[0])
    slots = list(
        zip(cp.asnumpy(op.slot_node).tolist(), cp.asnumpy(op.slot_tag).tolist(), strict=True)
    )
    expected = _numpy_spr(
        nodes,
        tets.astype(np.int64),
        field.astype(np.float64),
        _per_tag_boundary(boundary_faces, tets, tags, ctx.n_nodes),
        tags=tags,
        slots=slots,
    )
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)


def test_spr_tissue_averages_at_a_tissue_interface(cp, two_region_mesh, two_region_z0):
    """The interface rule is SimNIBS's, so it is pinned separately from the fit.

    Each side of the jump takes its own tissue's volume-weighted average, which is what keeps
    the field two-valued there without either side fitting a half-ball.
    """
    from cunibs.fem import apply_recovery, build_context, ensure_recovery, outer_boundary_nodes

    ctx = build_context(two_region_mesh)
    op = ensure_recovery(ctx, "spr_tissue")
    nodes, tets, tags = ctx.mesh.nodes_mm, ctx.mesh.tet_nodes, ctx.mesh.tet_tags
    rng = np.random.default_rng(12)
    field = rng.normal(size=(tets.shape[0], 3)).astype(np.float32)
    got = cp.asnumpy(apply_recovery(op, tets.shape[0], elements=[cp.asarray(field)]).E_slots[0])

    vol = np.abs(np.linalg.det(nodes[tets][:, 1:] - nodes[tets][:, :1])) / 6.0
    outer = cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, ctx.n_nodes)).astype(bool)
    slot_node, slot_tag = cp.asnumpy(op.slot_node), cp.asnumpy(op.slot_tag)
    # Interior to the volume, so only the interface rule can be sending these to the average.
    interface = ~outer[slot_node] & np.isclose(nodes[slot_node][:, 2], two_region_z0)
    assert interface.sum() > 50

    for s in np.flatnonzero(interface)[::5]:
        ets = np.flatnonzero((tets == slot_node[s]).any(axis=1) & (tags == slot_tag[s]))
        want = (field[ets].astype(np.float64) * vol[ets, None]).sum(0) / vol[ets].sum()
        np.testing.assert_allclose(got[s], want, atol=1e-5, rtol=1e-5)


def test_spr_tissue_still_fits_away_from_an_interface(cp, two_region_mesh):
    """Averaging at the interface must not disable the fit everywhere else."""
    from cunibs.fem import apply_recovery, build_context, ensure_recovery, tissue_boundary_nodes

    ctx = build_context(two_region_mesh)
    op = ensure_recovery(ctx, "spr_tissue")
    nodes, tets = ctx.mesh.nodes_mm, ctx.mesh.tet_nodes
    bary = nodes[tets].mean(axis=1)
    rng = np.random.default_rng(13)
    offset, slope = rng.normal(size=3), rng.normal(size=(3, 3))
    field = (offset + bary @ slope.T).astype(np.float32)
    scale = np.abs(offset + bary @ slope.T).max()

    got = cp.asnumpy(apply_recovery(op, tets.shape[0], elements=[cp.asarray(field)]).E_slots[0])
    is_boundary = cp.asnumpy(
        tissue_boundary_nodes(ctx.tet_nodes, ctx.n_nodes, op.slot_node)
    ).astype(bool)
    slot_node = cp.asnumpy(op.slot_node)
    fitted = ~is_boundary[slot_node]
    assert fitted.sum() > 100

    # A linear field is in the fit space, so a fitted slot must return it exactly.
    want = offset + nodes[slot_node[fitted]] @ slope.T
    np.testing.assert_allclose(got[fitted], want, atol=1e-5 * scale)


def _count_rejected_fits(cp, ctx, mode):
    """Slots sent to the average by a guard rather than by the boundary rule.

    The kernel rejects a patch on three tests SimNIBS has no counterpart for: fewer than four
    incident tetrahedra, a non-positive Cholesky pivot, and a Lebesgue constant above 8. Each
    rejection lands in n_fallback alongside the boundary slots, so subtracting the boundary
    count isolates them.
    """
    from cunibs.fem import (
        TISSUE_SLOT_MODES,
        ensure_recovery,
        outer_boundary_nodes,
        tissue_boundary_nodes,
    )

    op = ensure_recovery(ctx, mode)
    mask = (
        tissue_boundary_nodes(ctx.tet_nodes, ctx.n_nodes, op.slot_node)
        if mode in TISSUE_SLOT_MODES
        else outer_boundary_nodes(ctx.tet_nodes, ctx.n_nodes)
    )
    return op.n_fallback - int(cp.asnumpy(mask[op.slot_node]).sum())


@pytest.mark.parametrize("mode", ["spr_global", "spr_tissue"])
def test_no_patch_is_rejected_by_a_guard_simnibs_lacks(cp, two_region_mesh, mode):
    """Every fallback must be the boundary rule, which is the rule SimNIBS shares.

    SimNIBS averages on the boundary and fits everywhere else. A slot rejected by one of the
    kernel's guards is one where cuNIBS reports an average and SimNIBS reports a fit, so the two
    codes would no longer be computing the same quantity.
    """
    from cunibs.fem import build_context

    assert _count_rejected_fits(cp, build_context(two_region_mesh), mode) == 0


@pytest.mark.realmesh
@pytest.mark.parametrize("mode", ["spr_global", "spr_tissue"])
def test_no_patch_is_rejected_by_a_guard_on_a_head(cp, patch_subject, mode):
    """The same on real anatomy, where thin structures actually occur."""
    assert _count_rejected_fits(cp, patch_subject.context, mode) == 0


@pytest.mark.realmesh
def test_spr_tissue_preserves_the_jump_that_spr_smears(
    patch_subject, d70_coil, patch_placement
):
    """Restricting a patch to one tissue is the whole point, so it must change the answer.

    Measured where it should: on tetrahedra that touch a tissue interface. Away from one the two
    modes see the same patch and should agree closely.
    """
    kw = {"magnitude": True, "vectors": True}
    plain = patch_subject.simulate(d70_coil, patch_placement, 1e6, recovery="spr_global", **kw)
    tissue = patch_subject.simulate(d70_coil, patch_placement, 1e6, recovery="spr_tissue", **kw)

    tets = patch_subject.mesh.tet_nodes
    tags = patch_subject.mesh.tet_tags
    mixed_node = np.zeros(patch_subject.mesh.n_nodes, dtype=bool)
    seen = np.full(patch_subject.mesh.n_nodes, -1, dtype=np.int64)
    for tet, tag in zip(tets, tags, strict=True):
        for node in tet:
            if seen[node] == -1:
                seen[node] = tag
            elif seen[node] != tag:
                mixed_node[node] = True
    at_interface = mixed_node[tets].any(axis=1)
    assert at_interface.any() and not at_interface.all()

    def rel(mask):
        a, b = plain.magnE[mask], tissue.magnE[mask]
        return float(np.abs(a - b).max() / np.abs(b).max())

    assert rel(at_interface) > 10 * rel(~at_interface)


# --- harmonic-constrained potential recovery -------------------------------------------------


def _harmonic_grads(op, v, dadt, n_tet, cp):
    """Run the harmonic operator for one placement and return (per-slot E, per-tet E)."""
    from cunibs.fem import apply_recovery

    rec = apply_recovery(
        op,
        n_tet,
        potential=cp.ascontiguousarray(cp.asarray(v, dtype=cp.float64).reshape(-1, 1)),
        dadt_nodes=[cp.ascontiguousarray(cp.asarray(dadt, dtype=cp.float32))],
    )
    return cp.asnumpy(rec.E_slots[0]), cp.asnumpy(rec.E[0])


@pytest.fixture(scope="module")
def harmonic_op(refined_cube_mesh):
    from cunibs.fem import build_context, ensure_recovery

    ctx = build_context(refined_cube_mesh)
    return ctx, ensure_recovery(ctx, "harmonic")


def test_harmonic_reproduces_its_own_fit_space(cp, harmonic_op):
    """Constant, linear and *harmonic* quadratic potentials must all come back exactly.

    The harmonic quadratic is the load-bearing one: it is in the 9-term space precisely because
    Laplace's equation holds inside a tissue, so exactness there is what the constraint buys.
    """
    ctx, op = harmonic_op
    nodes = ctx.mesh.nodes_mm
    n_tet = ctx.mesh.tet_nodes.shape[0]
    zero = np.zeros((nodes.shape[0], 3), dtype=np.float32)

    at_nodes, _ = _harmonic_grads(op, np.full(nodes.shape[0], 3.7), zero, n_tet, cp)
    # Relative to what the same operator returns for a potential that does vary: an absolute
    # bound would just be tracking the weights' 1/m scale.
    reference, _ = _harmonic_grads(op, nodes[:, 0], zero, n_tet, cp)
    assert np.abs(at_nodes).max() < 1e-4 * np.abs(reference).max()

    # Coordinates are millimetres but E is V/m, so a potential rising by ``slope`` per mm has
    # a gradient of ``1000 * slope`` per metre. Asserting in physical units is what catches a
    # weight operator built in the wrong length unit; a per-mm reference would cancel it.
    rng = np.random.default_rng(0)
    slope = rng.normal(size=3)
    at_nodes, _ = _harmonic_grads(op, nodes @ slope, zero, n_tet, cp)
    expected = np.broadcast_to(-slope * 1e3, at_nodes.shape)
    np.testing.assert_allclose(at_nodes, expected, rtol=1e-4)

    # x^2 - y^2 + 2xz + 0.5yz: harmonic, since the three second derivatives sum to zero.
    x, y, z = nodes.T
    potential = x**2 - y**2 + 2 * x * z + 0.5 * y * z
    gradient = np.stack([2 * x + 2 * z, -2 * y + 0.5 * z, 2 * x + 0.5 * y], axis=1) * 1e3
    at_nodes, _ = _harmonic_grads(op, potential, zero, n_tet, cp)
    assert np.abs(at_nodes + gradient).max() / np.abs(gradient).max() < 1e-5


def test_harmonic_constraint_is_actually_active(cp, harmonic_op):
    """A non-harmonic quadratic must NOT be reproduced, or the basis is the unconstrained one.

    Without this the exactness test above would pass just as well for a plain 10-term quadratic
    fit, which is a measurably worse recovery at interfaces.
    """
    ctx, op = harmonic_op
    nodes = ctx.mesh.nodes_mm
    n_tet = ctx.mesh.tet_nodes.shape[0]
    zero = np.zeros((nodes.shape[0], 3), dtype=np.float32)

    exact = np.zeros_like(nodes)
    exact[:, 0] = 2 * nodes[:, 0] * 1e3
    at_nodes, _ = _harmonic_grads(op, nodes[:, 0] ** 2, zero, n_tet, cp)
    assert np.abs(at_nodes + exact).max() / np.abs(exact).max() > 1e-3


def test_harmonic_adds_dadt_exactly_at_the_nodes(cp, harmonic_op):
    """The raw path averages dA/dt onto elements; this one keeps the nodal value it was given."""
    ctx, op = harmonic_op
    nodes = ctx.mesh.nodes_mm
    n_nodes = ctx.mesh.n_nodes
    n_tet = ctx.mesh.tet_nodes.shape[0]
    rng = np.random.default_rng(1)
    dadt = rng.normal(size=(n_nodes, 3)).astype(np.float32)
    at_nodes, _ = _harmonic_grads(op, np.zeros(n_nodes), dadt, n_tet, cp)
    np.testing.assert_array_equal(at_nodes, -dadt)

    # Both terms at once, at a comparable magnitude. E = -grad(v) - dA/dt only comes out right
    # if the two are in the same units, which a test with either term zeroed cannot see.
    slope = rng.normal(size=3)
    at_nodes, _ = _harmonic_grads(op, nodes @ slope, dadt, n_tet, cp)
    np.testing.assert_allclose(at_nodes, -slope * 1e3 - dadt, rtol=1e-4)


def test_harmonic_patch_ladder_covers_every_slot(harmonic_op):
    """Growing to the next ring is what keeps the 9-term fit determined on real patches."""
    _, op = harmonic_op
    assert op.n_slots > 0
    assert op.idx.shape[0] / op.n_slots >= 9, "patches must be big enough for the basis"


def test_harmonic_weights_cannot_amplify(cp, harmonic_op):
    """h * max_c sum_m |w[m, c]| is bounded, so the fit cannot turn rounding into field error.

    The gradient weights carry a length unit, so unlike the SPR ones they are only bounded once
    scaled by the patch's own radius. A pivot test alone does not catch a merely very flat patch:
    it passes and returns weights of order 1e10.
    """
    ctx, op = harmonic_op
    ptr, idx = cp.asnumpy(op.ptr), cp.asnumpy(op.idx)
    w, slot_node = cp.asnumpy(op.w), cp.asnumpy(op.slot_node)
    nodes_m = cp.asnumpy(ctx.nodes_mm) * 1e-3

    owner = np.repeat(np.arange(op.n_slots), np.diff(ptr))
    offset = np.abs(nodes_m[idx] - nodes_m[slot_node[owner]]).max(axis=1)
    radius = np.maximum.reduceat(offset, ptr[:-1])
    amplification = radius * np.add.reduceat(np.abs(w), ptr[:-1], axis=0).max(axis=1)
    assert amplification.max() <= 50.0, f"max amplification = {amplification.max()}"


@pytest.mark.realmesh
def test_harmonic_is_reproducible_and_block_width_invariant(
    patch_subject, d70_coil, patch_placement
):
    reference = [
        r.magnE
        for r in patch_subject.iter_simulate(
            d70_coil, [patch_placement] * 3, 1e6, magnitude=True, recovery="harmonic", block_k=1
        )
    ]
    for block_k in (1, 2, 3, 5, 8):
        got = [
            r.magnE
            for r in patch_subject.iter_simulate(
                d70_coil,
                [patch_placement] * 3,
                1e6,
                magnitude=True,
                recovery="harmonic",
                block_k=block_k,
            )
        ]
        for i, (a, b) in enumerate(zip(got, reference, strict=True)):
            np.testing.assert_array_equal(a, b, err_msg=f"block_k={block_k}, placement {i}")


@pytest.mark.realmesh
def test_harmonic_serial_matches_the_block_path(patch_subject, d70_coil, patch_placement):
    """solve_placement keeps its own code path, so it has to agree with the width-1 block one."""
    from cunibs.fem import ensure_recovery, solve_placement

    ctx = patch_subject.context
    serial = solve_placement(
        ctx,
        d70_coil.positions_m,
        d70_coil.moments,
        patch_placement.center_mm,
        patch_placement.handle_mm,
        patch_placement.distance_mm,
        1e6,
        recovery=ensure_recovery(ctx, "harmonic"),
    )
    blocked = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, recovery="harmonic"
    )
    import cupy as cp_

    np.testing.assert_allclose(
        cp_.asnumpy(serial["magnE"]), blocked.magnE, rtol=2e-5, atol=1e-6
    )


# --- the interface comparison ------------------------------------------------------------------

_SIGMA = {2: 0.275, 3: 1.654}  # gray matter, CSF -- the table cuNIBS solves with
# 1/mm. The exact solution varies over ~1/k = 50 mm, so at the fixture's 10 mm cells the
# mesh resolves it (h/lambda = 0.2). Coarser than about 0.4 and plain discretisation error
# swamps the recovery being compared, which makes the comparison meaningless rather than
# merely noisy.
_K = 0.02


def _exact_two_region(points, region, z0):
    """v = e^{kz} cos kx below, matched above so v and sigma dv/dz are continuous.

    Both branches are harmonic, and dv/dz jumps by sigma_lo / sigma_hi across the plane, which
    is exactly the structure of E at a tissue boundary.
    """
    ratio = _SIGMA[2] / _SIGMA[3]
    a2 = (1.0 + ratio) / 2.0
    b2 = np.exp(2 * _K * z0) * (1.0 - ratio) / 2.0
    x, z = points[..., 0], points[..., 2]
    if region == 2:
        f, fz = np.exp(_K * z), _K * np.exp(_K * z)
    else:
        f = a2 * np.exp(_K * z) + b2 * np.exp(-_K * z)
        fz = _K * (a2 * np.exp(_K * z) - b2 * np.exp(-_K * z))
    return f * np.cos(_K * x), fz


def _exact_grad(points, region, z0):
    """The exact gradient in V/m. Points are in mm, so the per-mm derivative scales by 1e3."""
    f, fz = _exact_two_region(points, region, z0)
    grad = np.zeros(points.shape)
    grad[..., 0] = -_K * (f / np.cos(_K * points[..., 0])) * np.sin(_K * points[..., 0])
    grad[..., 2] = fz * np.cos(_K * points[..., 0])
    return grad * 1e3


def _p1_solve(nodes, tets, tags, z0):
    """Solve the two-region problem on the host, with Dirichlet data from the exact solution."""
    n = nodes.shape[0]
    sigma = np.where(tags == 2, _SIGMA[2], _SIGMA[3])
    # Metres, so the returned gradient operator is 1/m and the field built from it is V/m,
    # matching both the solver and the recovery operators.
    nodes_m = nodes * 1e-3
    edges = nodes_m[tets][:, 1:] - nodes_m[tets][:, :1]
    vol = np.abs(np.linalg.det(edges)) / 6.0
    basis = np.hstack([-np.ones((3, 1)), np.eye(3)])
    grad = np.transpose(
        np.linalg.solve(edges, np.broadcast_to(basis, (tets.shape[0], 3, 4))), (0, 2, 1)
    )
    local = (vol * sigma)[:, None, None] * np.einsum("eik,ejk->eij", grad, grad)
    stiffness = np.zeros((n, n))
    np.add.at(stiffness, (tets[:, :, None], tets[:, None, :]), local)

    on_boundary = np.zeros(n, dtype=bool)
    for value in (0.0, 100.0):
        for axis in range(3):
            on_boundary |= np.isclose(nodes[:, axis], value)
    exact = np.where(
        nodes[:, 2] < z0,
        _exact_two_region(nodes, 2, z0)[0],
        _exact_two_region(nodes, 3, z0)[0],
    )
    v = exact.copy()
    free = ~on_boundary
    v[free] = np.linalg.solve(
        stiffness[np.ix_(free, free)],
        -stiffness[np.ix_(free, on_boundary)] @ exact[on_boundary],
    )
    return v, grad


@pytest.mark.slow
def test_harmonic_beats_spr_across_a_conductivity_jump(cp, two_region_mesh, two_region_z0):
    """Global SPR cannot represent a two-valued field; the tissue-restricted modes can.

    Every mode is scored on the same (node, tissue) slot set against the one-sided exact
    gradient, which is the only apples-to-apples comparison: SimNIBS's global patch produces one
    value per node, so at an interface it reports the same number for both tissues and can at
    best land between the two truths.
    """
    from cunibs.fem import apply_recovery, build_context, ensure_recovery, outer_boundary_nodes

    z0 = two_region_z0
    ctx = build_context(two_region_mesh)
    mesh = ctx.mesh
    nodes, tets, tags = mesh.nodes_mm, mesh.tet_nodes.astype(np.int64), mesh.tet_tags
    v, grad = _p1_solve(nodes, tets, tags, z0)
    n_tet = tets.shape[0]
    e_raw = -np.einsum(
        "ei,eik->ek", v[tets], grad
    )  # dA/dt is zero in this manufactured problem

    tissue_op = ensure_recovery(ctx, "spr_tissue")
    slot_node = cp.asnumpy(tissue_op.slot_node)
    slot_tag = cp.asnumpy(tissue_op.slot_tag)
    exact = np.zeros((slot_node.shape[0], 3))
    for tag in (2, 3):
        sel = slot_tag == tag
        exact[sel] = -_exact_grad(nodes[slot_node[sel]], tag, z0)

    # Score away from the outer surface, where every mode applies the same volume-weighted
    # average and a rule they share would dominate a comparison about the rules they do not.
    # Interface slots stay in: spr_tissue averages there too, but over one tissue alone, and how
    # that one-sided average compares to a global fit and a harmonic one is the question.
    on_surface = cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, nodes.shape[0])).astype(bool)
    interior = ~on_surface[slot_node]
    at_interface = interior & np.isclose(nodes[slot_node][:, 2], z0)
    assert interior.sum() > 100
    assert at_interface.sum() > 50

    def rms(recovered_at_slot, where):
        scale = np.sqrt(np.mean(np.sum(exact[where] ** 2, axis=1)))
        residual = recovered_at_slot[where] - exact[where]
        return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))) / scale)

    field = [cp.asarray(e_raw.astype(np.float32))]
    global_op = ensure_recovery(ctx, "spr_global")
    per_node = cp.asnumpy(apply_recovery(global_op, n_tet, elements=field).E_slots[0])
    recovered = {
        "spr_global": per_node[slot_node],
        "spr_tissue": cp.asnumpy(apply_recovery(tissue_op, n_tet, elements=field).E_slots[0]),
        "harmonic": _harmonic_grads(
            ensure_recovery(ctx, "harmonic"),
            v,
            np.zeros((nodes.shape[0], 3), np.float32),
            n_tet,
            cp,
        )[0],
    }
    errors = {mode: rms(e, interior) for mode, e in recovered.items()}
    # Again over the interface slots alone, where the modes actually differ.
    interface = {mode: rms(e, at_interface) for mode, e in recovered.items()}

    assert errors["spr_tissue"] < errors["spr_global"], errors
    assert errors["harmonic"] < errors["spr_tissue"], errors
    assert errors["harmonic"] < 0.25 * errors["spr_global"], errors
    # A single value per node cannot represent a two-valued field, so global SPR is not merely
    # less accurate at the interface, it is wrong there by a margin the mesh cannot reduce.
    assert interface["harmonic"] < 0.1 * interface["spr_global"], interface
    assert interface["spr_tissue"] < 0.5 * interface["spr_global"], interface


@pytest.mark.realmesh
@pytest.mark.parametrize("mode", ["spr_global", "spr_tissue", "harmonic"])
def test_uq_matches_the_deterministic_field_for_every_mode(
    fresh_subject, patch_mesh, d70_coil, patch_placement, mode
):
    """A one-draw ensemble at nominal conductivity is the deterministic solve, recovery included.

    This is what catches a mode wired into the deterministic path but not the UQ one: the
    harmonic operator consumes the nodal potential rather than the per-element field, so it
    needs its own branch in the sampling loop.
    """
    from cunibs import ConductivityUQConfig

    subject = fresh_subject(patch_mesh)
    # perturbed_tags has to be pinned as well as the CoV: cov_for falls back to the default
    # table for any tissue tissue_cov does not name, so leaving it open would perturb the rest
    # of the head and the single draw would not be the nominal one.
    config = ConductivityUQConfig(n_samples=1, tissue_cov={2: 0.0}, perturbed_tags=(2,), seed=0)
    uq = subject.simulate_conductivity_uq(
        d70_coil, patch_placement, config, 1e6, moments=True, recovery=mode
    )
    deterministic = subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, recovery=mode
    )
    assert uq.recovery == mode
    np.testing.assert_allclose(
        uq.mean_magnE, deterministic.magnE, rtol=2e-3, atol=1e-4 * deterministic.peak_magnE()
    )


@pytest.mark.parametrize("mode", ["spr_global", "spr_tissue"])
def test_patch_weights_cannot_amplify(cp, refined_cube_mesh, mode):
    """sum|w| per patch is bounded, so a recovery smooths its input rather than magnifying it.

    The weights sum to one, so sum|w| is the patch's Lebesgue constant and bounds how far the
    recovered value can travel outside the range of its inputs. A pivot test alone does not catch
    a merely very flat patch: it passes and returns weights of order thousands.
    """
    from cunibs.fem import build_context, ensure_recovery

    op = ensure_recovery(build_context(refined_cube_mesh), mode)
    w, ptr = cp.asnumpy(op.w), cp.asnumpy(op.ptr)
    lebesgue = np.add.reduceat(np.abs(w), ptr[:-1])
    assert lebesgue.max() <= 8.0 + 1e-4, f"max sum|w| = {lebesgue.max()}"


@pytest.mark.realmesh
@pytest.mark.parametrize("mode", ["spr_global", "spr_tissue", "harmonic"])
def test_recovery_does_not_raise_the_peak(patch_subject, d70_coil, patch_placement, mode):
    """Recovery exists to suppress the sliver-driven peak, so it must never exceed the raw one."""
    raw = patch_subject.simulate(d70_coil, patch_placement, 1e6, magnitude=True, recovery="raw")
    recovered = patch_subject.simulate(
        d70_coil, patch_placement, 1e6, magnitude=True, recovery=mode
    )
    assert recovered.peak_magnE() <= raw.peak_magnE(), (
        f"{mode} peak {recovered.peak_magnE()} exceeds raw {raw.peak_magnE()}"
    )
