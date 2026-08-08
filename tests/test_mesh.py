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
    assert mesh.skin_triangle_normals.shape == (1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(mesh.skin_triangle_normals, axis=1), 1.0, atol=1e-12
    )
    assert mesh.tet_barycenters_mm.shape == (1, 3)
    np.testing.assert_allclose(mesh.tet_barycenters_mm[0], _NODES[:4].mean(0))


def _disjoint_mesh(n_tets: int, tags):
    """``n_tets`` tetrahedra on disjoint node quadruples, so no node is ever reindexed away."""
    nodes = np.arange(4 * n_tets * 3, dtype=np.float64).reshape(-1, 3)
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


def test_every_conductivity_tag_is_loadable():
    """The two tables must agree, or the loader narrows the tissue model behind an error.

    Imported here rather than at module scope to keep this file runnable without CUDA.
    """
    from cunibs.fem.assembly import TISSUE_CONDUCTIVITY

    assert set(VOLUME_KEY_TO_LABEL) == set(TISSUE_CONDUCTIVITY)


def test_parser_filters_unknown_volume_tags(tmp_path):
    """Tags outside VOLUME_KEY_TO_LABEL drop out, and their nodes drop with them."""
    nodes, tets, _ = _disjoint_mesh(3, [2, 77, 5])
    path = tmp_path / "vol.msh"
    path.write_bytes(
        pack_msh(nodes, tets + 1, [2, 77, 5], np.empty((0, 3), np.int32), np.empty(0, np.int32))
    )
    out_nodes, tet_nodes, tet_tags, _, _ = parse_msh_binary(path)
    np.testing.assert_array_equal(tet_tags, [2, 5])
    assert out_nodes.shape == (8, 3)  # tag-77's four nodes are gone
    np.testing.assert_array_equal(tet_nodes, [[0, 1, 2, 3], [4, 5, 6, 7]])
    np.testing.assert_allclose(out_nodes, nodes[[0, 1, 2, 3, 8, 9, 10, 11]])


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
    """A skin triangle whose nodes died with a filtered tet is a hard error.

    ``tools/make_test_patch.py`` depends on this: it may only keep triangles whose three
    nodes all survive the tet subset.
    """
    nodes, tets, _ = _disjoint_mesh(2, [2, 77])
    orphan_tri = tets[1][:3] + 1  # nodes owned only by the tag-77 tet
    path = tmp_path / "orphan.msh"
    path.write_bytes(pack_msh(nodes, tets + 1, [2, 77], orphan_tri[None], [1005]))
    with pytest.raises(ValueError, match="dropped with a filtered tetrahedron"):
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


def test_skin_normals_unit_and_outward_on_cube(cube_mesh):
    """Two-pass normal smoothing on a closed surface: unit length, all pointing outward."""
    normals = cube_mesh.skin_triangle_normals
    assert normals.shape == cube_mesh.skin_tris.shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    centroids = cube_mesh.nodes_mm[cube_mesh.skin_tris].mean(axis=1)
    outward = centroids - cube_mesh.nodes_mm.mean(axis=0)
    assert np.all((outward * normals).sum(axis=1) > 0)


def test_empty_surface_block(tmp_path):
    path = tmp_path / "noskin.msh"
    path.write_bytes(
        pack_msh(_NODES, _TETS_1B, _TET_TAGS, np.empty((0, 3), np.int32), np.empty(0, np.int32))
    )
    mesh = load_mesh(path)
    assert mesh.skin_tris.shape == (0, 3)
    assert mesh.skin_triangle_normals.shape == (0, 3)


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
