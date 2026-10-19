from __future__ import annotations

import mmap
import struct
import time

import numpy as np
import pytest
from msh_writer import pack_msh

from cunibs.mesh import VOLUME_KEY_TO_LABEL, load_mesh, parse_msh_binary

# Node 5 is unused and must be removed during reindexing.
_NODES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [9.0, 9.0, 9.0],
    ]
)
_TETS_1B = np.array([[1, 2, 3, 4]], dtype=np.int32)
_TET_TAGS = np.array([2], dtype=np.int32)  # gray matter
_TRIS_1B = np.array([[1, 2, 3]], dtype=np.int32)
_TRI_TAGS = np.array([1005], dtype=np.int32)  # scalp surface


def _write(tmp_path):
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "mesh.msh"
    path.write_bytes(blob)
    return path


def test_parse_msh_binary_reindexes_and_filters(tmp_path):
    nodes, tet_nodes, tet_tags, surf_tris, surf_tags = parse_msh_binary(_write(tmp_path))
    assert nodes.shape == (4, 3)
    np.testing.assert_allclose(nodes, _NODES[:4])
    assert tet_nodes.shape == (1, 4)
    np.testing.assert_array_equal(tet_nodes, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(tet_tags, [2])
    np.testing.assert_array_equal(surf_tris, [[0, 1, 2]])
    np.testing.assert_array_equal(surf_tags, [1005])


def test_load_mesh_selects_skin_and_derives_geometry(tmp_path):
    mesh = load_mesh(_write(tmp_path))
    assert mesh.n_nodes == 4
    assert mesh.skin_tris.shape == (1, 3)
    assert mesh.tet_barycenters_mm.shape == (1, 3)
    np.testing.assert_allclose(mesh.tet_barycenters_mm[0], _NODES[:4].mean(0))


def _disjoint_mesh(n_tets: int, tags):
    """``n_tets`` tetrahedra on disjoint node quadruples, so no node is ever reindexed away."""
    unit = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    offsets = np.arange(n_tets, dtype=np.float64)[:, None, None] * np.array([2.0, 0.0, 0.0])
    nodes = (unit + offsets).reshape(-1, 3)
    tets = np.arange(4 * n_tets, dtype=np.int32).reshape(n_tets, 4)
    return nodes, tets, np.asarray(tags, dtype=np.int32)


def _element_block(elem_type: int, count: int, nodes_per_elem: int) -> tuple[int, int, bytes]:
    """A Gmsh 2.2 element block: each record is (number, 2 tags, node ids) as int32."""
    payload = np.ones((count, 3 + nodes_per_elem), dtype="<i4").tobytes()
    return elem_type, count, payload


def test_parser_skips_unknown_element_types(tmp_path):
    """Types the reader does not consume are skipped by byte count, not by desynchronizing."""
    blob = pack_msh(
        _NODES,
        _TETS_1B,
        _TET_TAGS,
        _TRIS_1B,
        _TRI_TAGS,
        extra_blocks=[
            _element_block(3, 5, 4),  # 4-node quad, 28 B each
            _element_block(5, 2, 8),  # 8-node hexahedron, 44 B each
        ],
    )
    path = tmp_path / "extra.msh"
    path.write_bytes(blob)
    _, tet_nodes, _, surf_tris, surf_tags = parse_msh_binary(path)
    np.testing.assert_array_equal(tet_nodes, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(surf_tris, [[0, 1, 2]])
    np.testing.assert_array_equal(surf_tags, [1005])


@pytest.mark.parametrize(
    "elem_type,nodes_per_elem,name",
    [(1, 2, "2-node line"), (15, 1, "point")],
)
def test_parser_skips_line_and_point_elements(tmp_path, elem_type, nodes_per_elem, name):
    """A mesh carrying physical curves and points loads like any other.

    Gmsh routinely emits these alongside the surfaces and volumes the reader consumes, so
    both must be skipped by their true record size: 20 B for a 2-node line and 16 B for a
    point under the num_tags=2 layout.
    """
    blob = pack_msh(
        _NODES,
        _TETS_1B,
        _TET_TAGS,
        _TRIS_1B,
        _TRI_TAGS,
        extra_blocks=[_element_block(elem_type, 3, nodes_per_elem)],
    )
    path = tmp_path / f"{name}.msh"
    path.write_bytes(blob)
    _, tet_nodes, tet_tags, _, _ = parse_msh_binary(path)
    np.testing.assert_array_equal(tet_nodes, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(tet_tags, [2])


def test_parser_rejects_unsized_element_type(tmp_path):
    """An element type the reader cannot size is refused by name.

    It cannot be skipped without knowing its record width, so consuming nothing would
    desynchronize the stream and surface later as an unrelated header mismatch.
    """
    blob = pack_msh(
        _NODES,
        _TETS_1B,
        _TET_TAGS,
        _TRIS_1B,
        _TRI_TAGS,
        extra_blocks=[_element_block(6, 2, 6)],  # 6-node prism
    )
    path = tmp_path / "prism.msh"
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="element type 6"):
        parse_msh_binary(path)


def test_parser_rejects_unknown_volume_tags(tmp_path):
    """A tag with no tissue is refused, not dropped: losing a tissue must never be quiet."""
    tags = [2, 77, 5, 77]
    nodes, tets, _ = _disjoint_mesh(4, tags)
    path = tmp_path / "vol.msh"
    path.write_bytes(
        pack_msh(nodes, tets + 1, tags, np.empty((0, 3), np.int32), np.empty(0, np.int32))
    )
    with pytest.raises(ValueError, match=r"2 tetrahedra carry volume tags.*\(77: 2\)"):
        parse_msh_binary(path)


def test_parser_rejects_nonfinite_referenced_nodes(tmp_path):
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]])
    path = tmp_path / "invalid-geometry.msh"
    path.write_bytes(
        pack_msh(
            nodes,
            _TETS_1B,
            _TET_TAGS,
            np.empty((0, 3), np.int32),
            np.empty(0, np.int32),
        )
    )
    with pytest.raises(ValueError, match="1 referenced node has non-finite coordinates"):
        parse_msh_binary(path)


def test_every_conductivity_tag_is_loadable():
    """The two tables must agree, or the loader narrows the tissue model behind an error.

    Imported here rather than at module scope to keep this file runnable without CUDA.
    """
    from cunibs.fem.assembly import TISSUE_CONDUCTIVITY

    assert set(VOLUME_KEY_TO_LABEL) == set(TISSUE_CONDUCTIVITY)


def test_parser_joins_multiple_tetrahedron_blocks(tmp_path):
    """Gmsh may emit one tet block per volume entity; every block must survive."""
    nodes, tets, _ = _disjoint_mesh(3, [2, 5, 9])
    # A second tet block, written by hand because pack_msh emits only one: per record an
    # element id, the two tags, then the four 1-based node ids.
    second = np.empty((2, 7), dtype="<i4")
    second[:, 0] = [98, 99]
    second[:, 1] = [5, 9]
    second[:, 2] = [5, 9]
    second[:, 3:] = tets[1:] + 1
    path = tmp_path / "twoblocks.msh"
    path.write_bytes(
        pack_msh(
            nodes,
            tets[:1] + 1,
            [2],
            np.empty((0, 3), np.int32),
            np.empty(0, np.int32),
            extra_blocks=[(4, len(second), second.tobytes())],
        )
    )
    _, tet_nodes, tet_tags, _, _ = parse_msh_binary(path)
    np.testing.assert_array_equal(tet_tags, [2, 5, 9])
    np.testing.assert_array_equal(tet_nodes, tets)


def test_parser_filters_unknown_surface_tags(tmp_path):
    """2000 is not a known surface tag; 1099 is known to the parser but is not skin."""
    tris = np.array([[1, 2, 3], [1, 2, 4], [2, 3, 4]], dtype=np.int32)
    path = tmp_path / "surf.msh"
    path.write_bytes(pack_msh(_NODES, _TETS_1B, _TET_TAGS, tris, [1005, 1099, 2000]))
    *_, surf_tags = parse_msh_binary(path)
    np.testing.assert_array_equal(surf_tags, [1005, 1099])
    assert load_mesh(path).skin_tris.shape == (1, 3)


def test_parser_rejects_orphan_surface_node(tmp_path):
    """A skin triangle standing on nodes no tetrahedron uses is a hard error.

    ``tools/make_test_patch.py`` depends on this: it may only keep triangles whose three
    nodes all survive the tet subset.
    """
    nodes, tets, _ = _disjoint_mesh(3, [2, 5, 9])
    orphan_tri = tets[2][:3] + 1
    path = tmp_path / "orphan.msh"
    # Only the first two tets are written, so the third quadruple's nodes reach the node
    # table without any tetrahedron referencing them.
    path.write_bytes(pack_msh(nodes, tets[:2] + 1, [2, 5], orphan_tri[None], [1005]))
    with pytest.raises(ValueError, match="no tetrahedron uses"):
        parse_msh_binary(path)


def test_parser_rejects_ascii_header(tmp_path):
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "ascii.msh"
    path.write_bytes(blob.replace(b"2.2 1 8", b"2.2 0 8", 1))
    with pytest.raises(ValueError, match=r"binary Gmsh 2\.2 header"):
        parse_msh_binary(path)


def test_parser_rejects_unexpected_tag_count(tmp_path):
    """Every block must carry exactly the two tags the reader indexes by."""
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "tags.msh"
    # The tet block header is (type 4, count 1, num_tags 2); make it claim three tags.
    path.write_bytes(blob.replace(struct.pack("<3i", 4, 1, 2), struct.pack("<3i", 4, 1, 3), 1))
    with pytest.raises(ValueError, match="declares 3 tags"):
        parse_msh_binary(path)


@pytest.mark.gpu
def test_skin_normals_unit_and_outward_on_cube(cube_mesh, cube_skin_normals):
    """Two-pass normal smoothing on a closed surface: unit length, all pointing outward."""
    assert cube_skin_normals.shape == cube_mesh.skin_tris.shape
    np.testing.assert_allclose(np.linalg.norm(cube_skin_normals, axis=1), 1.0, atol=1e-12)
    centroids = cube_mesh.nodes_mm[cube_mesh.skin_tris].mean(axis=1)
    outward = centroids - cube_mesh.nodes_mm.mean(axis=0)
    assert np.all((outward * cube_skin_normals).sum(axis=1) > 0)


def test_empty_surface_block(tmp_path):
    path = tmp_path / "noskin.msh"
    path.write_bytes(
        pack_msh(_NODES, _TETS_1B, _TET_TAGS, np.empty((0, 3), np.int32), np.empty(0, np.int32))
    )
    mesh = load_mesh(path)
    assert mesh.skin_tris.shape == (0, 3)


def test_region_labels_cover_every_documented_tag(tmp_path):
    """Every tag in VOLUME_KEY_TO_LABEL survives the parser's tag filter."""
    tags = sorted(VOLUME_KEY_TO_LABEL)
    nodes, tets, _ = _disjoint_mesh(len(tags), tags)
    path = tmp_path / "alltags.msh"
    path.write_bytes(
        pack_msh(nodes, tets + 1, tags, np.empty((0, 3), np.int32), np.empty(0, np.int32))
    )
    _, _, tet_tags, _, _ = parse_msh_binary(path)
    np.testing.assert_array_equal(tet_tags, tags)


def test_pack_msh_roundtrip_at_scale(tmp_path):
    """41k elements must round-trip exactly, and pack fast enough to stay vectorized."""
    n = 41_139
    rng = np.random.default_rng(0)
    nodes, tets, _ = _disjoint_mesh(n, np.full(n, 2))
    tags = rng.choice(np.array(sorted(VOLUME_KEY_TO_LABEL), np.int32), n)
    tris = tets[:, :3]

    start = time.perf_counter()
    blob = pack_msh(nodes, tets + 1, tags, tris + 1, np.full(n, 1005, np.int32))
    assert time.perf_counter() - start < 1.0

    path = tmp_path / "big.msh"
    path.write_bytes(blob)
    out_nodes, tet_nodes, tet_tags, surf_tris, surf_tags = parse_msh_binary(path)
    np.testing.assert_array_equal(out_nodes, nodes)
    np.testing.assert_array_equal(tet_nodes, tets)
    np.testing.assert_array_equal(tet_tags, tags)
    np.testing.assert_array_equal(surf_tris, tris)
    np.testing.assert_array_equal(surf_tags, np.full(n, 1005))


def test_pack_msh_rejects_mismatched_tag_counts():
    with pytest.raises(ValueError):
        pack_msh(_NODES, _TETS_1B, [2, 2], _TRIS_1B, _TRI_TAGS)


# --- Truncation and malformed headers -----------------------------------------------------
#
# The reader maps the file and consumes blocks as views, so every bounds check is its own
# responsibility: an unchecked length would read past the mapping rather than raise.


@pytest.mark.parametrize("keep", [0.1, 0.5, 0.9, 0.99])
def test_parser_rejects_truncation_at_any_point(tmp_path, keep):
    """A file cut anywhere must raise, never read past the mapping or return short arrays."""
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "cut.msh"
    path.write_bytes(blob[: int(len(blob) * keep)])
    with pytest.raises(ValueError):
        parse_msh_binary(path)


def test_parser_rejects_a_node_block_longer_than_the_file(tmp_path):
    """An inflated count must be caught by the block bounds check, not by a short read."""
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "bignodes.msh"
    path.write_bytes(blob.replace(b"$Nodes\n5\n", b"$Nodes\n9999\n", 1))
    with pytest.raises(ValueError, match="truncated"):
        parse_msh_binary(path)


def test_parser_rejects_a_non_numeric_count(tmp_path):
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "badcount.msh"
    path.write_bytes(blob.replace(b"$Nodes\n5\n", b"$Nodes\nmany\n", 1))
    with pytest.raises(ValueError, match="expected a node count"):
        parse_msh_binary(path)


def test_parser_rejects_big_endian_marker(tmp_path):
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "bigendian.msh"
    path.write_bytes(blob.replace(struct.pack("<i", 1), struct.pack(">i", 1), 1))
    with pytest.raises(ValueError, match="big-endian"):
        parse_msh_binary(path)


def test_parser_rejects_a_missing_section_marker(tmp_path):
    blob = pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
    path = tmp_path / "nomarker.msh"
    path.write_bytes(blob.replace(b"$EndNodes\n", b"$Bogus___\n", 1))
    with pytest.raises(ValueError, match=r"expected \$EndNodes"):
        parse_msh_binary(path)


def test_parser_rejects_a_mesh_with_no_tetrahedra(tmp_path):
    """A surface-only file has nothing to solve on and must say so, not return empty arrays."""
    blob = pack_msh(
        _NODES, np.empty((0, 4), np.int32), np.empty(0, np.int32), _TRIS_1B, _TRI_TAGS
    )
    # pack_msh always emits a tet block; drop the now-empty header so none is present at all.
    empty_tet_header = struct.pack("<3i", 4, 0, 2)
    assert empty_tet_header in blob
    path = tmp_path / "surfonly.msh"
    path.write_bytes(blob.replace(empty_tet_header, b"", 1))
    with pytest.raises(ValueError, match="no tetrahedron block"):
        parse_msh_binary(path)


@pytest.mark.parametrize("bad_id", [0, 6])
def test_parser_rejects_out_of_range_node_ids(tmp_path, bad_id):
    """Gmsh node ids are 1-based; 0 underflows to -1 and 6 overruns a 5-node table."""
    tets = np.array([[1, 2, 3, bad_id]], dtype=np.int32)
    path = tmp_path / "badid.msh"
    path.write_bytes(
        pack_msh(_NODES, tets, _TET_TAGS, np.empty((0, 3), np.int32), np.empty(0, np.int32))
    )
    with pytest.raises(ValueError, match="node id outside the node table"):
        parse_msh_binary(path)


@pytest.mark.parametrize("bad_id", [0, 6])
def test_parser_rejects_out_of_range_surface_node_ids(tmp_path, bad_id):
    """The same bounds check on the surface side, on the all-nodes-used fast path."""
    tris = np.array([[1, 2, bad_id]], dtype=np.int32)
    nodes = _NODES[:4]
    tets = np.array([[1, 2, 3, 4]], dtype=np.int32)
    path = tmp_path / "badtriid.msh"
    path.write_bytes(pack_msh(nodes, tets, _TET_TAGS, tris, _TRI_TAGS))
    with pytest.raises(ValueError, match="node id outside the node table"):
        parse_msh_binary(path)


# --- Memory-map ownership -----------------------------------------------------------------


@pytest.mark.parametrize("all_nodes_used", [True, False])
def test_parsed_arrays_do_not_alias_the_mapping(tmp_path, all_nodes_used):
    """Nothing returned may be a view into the mmap, which is released on return.

    ``_reindex_nodes`` has a fast path for when every node is referenced that skips the
    gather entirely, so both branches have to be checked: a view surviving either one would
    read freed memory rather than fail loudly.
    """
    nodes = _NODES if not all_nodes_used else _NODES[:4]
    path = tmp_path / "alias.msh"
    path.write_bytes(pack_msh(nodes, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS))

    arrays = parse_msh_binary(path)
    expected_nodes = nodes[:4]
    for name, value in arrays._asdict().items():
        base = value
        while (base := getattr(base, "base", None)) is not None:
            assert not isinstance(base, mmap.mmap), f"{name} aliases the mapping"

    # Overwrite the file, then re-read every array: values must be unaffected.
    path.write_bytes(b"\0" * path.stat().st_size)
    np.testing.assert_allclose(arrays.nodes_mm, expected_nodes)
    np.testing.assert_array_equal(arrays.tet_nodes, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(arrays.tet_tags, [2])
    np.testing.assert_array_equal(arrays.surf_tris, [[0, 1, 2]])
    np.testing.assert_array_equal(arrays.surf_tags, [1005])


def test_parsed_arrays_are_writable(tmp_path):
    """A view of a read-only mapping would be non-writable; owned copies are not."""
    path = tmp_path / "writable.msh"
    path.write_bytes(pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS))
    arrays = parse_msh_binary(path)
    for name, value in arrays._asdict().items():
        assert value.flags.writeable, name


def test_mesh_arrays_is_a_named_tuple(tmp_path):
    """The result still unpacks positionally, so older five-tuple call sites keep working."""
    path = tmp_path / "named.msh"
    path.write_bytes(pack_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS))
    arrays = parse_msh_binary(path)
    nodes, tet_nodes, tet_tags, surf_tris, surf_tags = arrays
    assert nodes is arrays.nodes_mm
    assert tet_nodes is arrays.tet_nodes
    assert tet_tags is arrays.tet_tags
    assert surf_tris is arrays.surf_tris
    assert surf_tags is arrays.surf_tags


def _write_mesh(tmp_path, nodes, tets, tags, tris=None, tri_tags=None, name="m.msh"):
    path = tmp_path / name
    path.write_bytes(
        pack_msh(
            np.asarray(nodes, dtype=np.float64),
            np.asarray(tets, dtype=np.int32) + 1,
            np.asarray(tags, dtype=np.int32),
            np.empty((0, 3), np.int32) if tris is None else np.asarray(tris, np.int32) + 1,
            np.empty(0, np.int32) if tri_tags is None else np.asarray(tri_tags, np.int32),
        )
    )
    return path


_UNIT_TET = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _head_like(n_tets=8, scale=1.0):
    """Disjoint unit tetrahedra carrying the tags a head model must have."""
    tags = [1, 2, 3, 5, 7, 8, 9, 2][:n_tets]
    nodes = np.vstack(
        [(_UNIT_TET + np.array([40.0 * i, 0.0, 0.0])) * scale for i in range(n_tets)]
    )
    tets = np.arange(4 * n_tets, dtype=np.int32).reshape(n_tets, 4)
    return nodes, tets, np.asarray(tags, dtype=np.int32)


def test_validate_mesh_accepts_a_sound_mesh(tmp_path):
    from cunibs import validate_mesh

    nodes, tets, tags = _head_like()
    report = validate_mesh(
        _write_mesh(tmp_path, nodes, tets, tags, tets[:, :3], [1005] * len(tets))
    )
    assert report.ok and not report.errors and not report.warnings
    assert report.n_tets == len(tets) and report.n_skin_tris == len(tets)
    assert tuple(report.tag_counts) == (1, 2, 3, 5, 7, 8, 9)
    # A unit tetrahedron is far from regular but nowhere near degenerate.
    assert 0.0 < report.min_flatness < 1.0


def test_validate_mesh_reports_a_degenerate_tet_without_a_gpu(tmp_path):
    """The check build_context only reaches after a CUDA context and a full upload."""
    from cunibs import validate_mesh

    nodes, tets, tags = _head_like()
    nodes[3] = nodes[2]  # collapse the first tetrahedron onto a face
    report = validate_mesh(_write_mesh(tmp_path, nodes, tets, tags))
    assert not report.ok
    assert report.n_nonpositive_volume == 1
    assert "zero, negative or non-finite volume" in report.errors[0]


@pytest.mark.gpu
def test_validate_mesh_volume_verdict_agrees_with_build_context(tmp_path):
    """Two implementations of one predicate must not drift.

    validate_mesh computes volumes on the host and build_context on the device, so a mesh either
    passes both or fails both; otherwise screening a cohort would admit meshes the solver rejects.
    """
    from cunibs import load_mesh, validate_mesh
    from cunibs.fem import build_context

    nodes, tets, tags = _head_like()
    good = _write_mesh(tmp_path, nodes, tets, tags, name="good.msh")
    assert validate_mesh(good).ok
    build_context(load_mesh(good))  # must not raise

    nodes[3] = nodes[2]
    bad = _write_mesh(tmp_path, nodes, tets, tags, name="bad.msh")
    assert not validate_mesh(bad).ok
    with pytest.raises(ValueError, match="zero or non-finite volume"):
        build_context(load_mesh(bad))


def test_validate_mesh_flatness_is_one_for_a_regular_tetrahedron(tmp_path):
    """The quality measure has to be scale-free and normalised, or a threshold means nothing."""
    from cunibs import validate_mesh

    a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
        ]
    )
    for scale in (1.0, 137.0):
        report = validate_mesh(
            _write_mesh(tmp_path, a * scale, [[0, 1, 2, 3]], [2], name=f"reg{scale}.msh")
        )
        assert report.min_flatness == pytest.approx(1.0)


def test_validate_mesh_warns_rather_than_raises_on_a_usable_but_odd_mesh(tmp_path):
    from cunibs import validate_mesh

    # No scalp surface: loads fine, then has nothing for a placement to project onto.
    nodes, tets, tags = _head_like()
    report = validate_mesh(_write_mesh(tmp_path, nodes, tets, tags, name="noskin.msh"))
    assert report.ok and any("1005" in w for w in report.warnings)

    # Metres instead of millimetres, the classic silent unit error.
    nodes, tets, tags = _head_like(scale=1e-3)
    report = validate_mesh(_write_mesh(tmp_path, nodes, tets, tags, name="metres.msh"))
    assert report.ok and any("millimetres" in w for w in report.warnings)

    # A brain with no skull is not a head model.
    nodes, tets, tags = _head_like(n_tets=4)
    report = validate_mesh(_write_mesh(tmp_path, nodes, tets, tags, name="nobone.msh"))
    assert report.ok and any("bone" in w for w in report.warnings)


def test_validate_mesh_returns_a_reason_instead_of_raising(tmp_path):
    """A cohort driver needs a verdict per mesh, not an exception on the first bad one."""
    from cunibs import validate_mesh

    missing = validate_mesh(tmp_path / "absent.msh")
    assert not missing.ok and missing.errors

    truncated = tmp_path / "trunc.msh"
    nodes, tets, tags = _head_like()
    truncated.write_bytes(_write_mesh(tmp_path, nodes, tets, tags).read_bytes()[:200])
    report = validate_mesh(truncated)
    assert not report.ok and report.errors
