from __future__ import annotations

import numpy as np

from conftest import build_binary_msh
from cunibs.mesh import load_mesh, parse_msh_binary

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
    blob = build_binary_msh(_NODES, _TETS_1B, _TET_TAGS, _TRIS_1B, _TRI_TAGS)
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
    np.testing.assert_allclose(mesh.nodes_m, mesh.nodes_mm * 1e-3)
    assert mesh.skin_triangle_normals.shape == (1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(mesh.skin_triangle_normals, axis=1), 1.0, atol=1e-12
    )
    assert mesh.tet_barycenters_mm.shape == (1, 3)
    np.testing.assert_allclose(mesh.tet_barycenters_mm[0], _NODES[:4].mean(0))
