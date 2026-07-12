from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

_ELEM_TYPE_TRIANGLE = 2
_ELEM_TYPE_TET = 4
_NODE_DTYPE = np.dtype([("id", "<i4"), ("xyz", "<f8", (3,))])
_ELEMENT_BYTES = {1: 16, 2: 24, 3: 28, 4: 28, 5: 44}

SKIN_SURFACE_TAG = 1005

TissueLabel: TypeAlias = Literal[
    "white_matter",
    "gray_matter",
    "csf",
    "scalp",
    "eye_balls",
    "cortical_bone",
    "cancellous_bone",
    "blood",
    "muscle",
]

VOLUME_KEY_TO_LABEL: dict[int, TissueLabel] = {
    1: "white_matter",
    2: "gray_matter",
    3: "csf",
    5: "scalp",
    6: "eye_balls",
    7: "cortical_bone",
    8: "cancellous_bone",
    9: "blood",
    10: "muscle",
}

SURFACE_KEY_TO_LABEL: dict[int, TissueLabel | Literal["internal_air"]] = {
    1001: "white_matter",
    1002: "gray_matter",
    1003: "csf",
    1005: "scalp",
    1006: "eye_balls",
    1007: "cortical_bone",
    1008: "cancellous_bone",
    1009: "blood",
    1010: "muscle",
    1099: "internal_air",
}

_VOLUME_KEYS = np.fromiter(VOLUME_KEY_TO_LABEL, dtype=np.int32)
_SURFACE_KEYS = np.fromiter(SURFACE_KEY_TO_LABEL, dtype=np.int32)


def parse_msh_binary(
    mesh_file: Path,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Parse a binary Gmsh 2.2 .msh file.

    Returns
    -------
    nodes       :   (N, 3)  float64 node XYZ coordinates in mm, 0-indexed
    tet_nodes   :   (M, 4)  int32   tetrahedron node indices, 0-indexed
    tet_tags    :   (M,)    int32   tissue label per tet
    surf_tris   :   (S, 3)  int32   surface triangle node indices, 0-indexed
    surf_tags   :   (S,)    int32   tissue label per surface triangle
    """
    with open(mesh_file, "rb") as f:
        assert f.readline().decode().strip() == "$MeshFormat"
        _, file_type, data_size = f.readline().decode().strip().split()
        assert file_type == "1" and data_size == "8"
        assert struct.unpack("<i", f.read(4))[0] == 1
        f.readline()
        assert f.readline().decode().strip() == "$EndMeshFormat"

        assert f.readline().decode().strip() == "$Nodes"
        num_nodes = int(f.readline().decode().strip())
        node_records = np.frombuffer(
            f.read(num_nodes * _NODE_DTYPE.itemsize), dtype=_NODE_DTYPE
        )
        nodes_xyz = node_records["xyz"]
        assert f.readline().decode().strip() == "$EndNodes"

        assert f.readline().decode().strip() == "$Elements"
        total_elements = int(f.readline().decode().strip())

        tet_nodes_raw: npt.NDArray[np.int32] | None = None
        tet_tags_raw: npt.NDArray[np.int32] | None = None
        tri_nodes_list: list[npt.NDArray[np.int32]] = []
        tri_tags_list: list[npt.NDArray[np.int32]] = []
        consumed = 0

        while consumed < total_elements:
            elem_type, count, num_tags = struct.unpack("<3i", f.read(12))
            assert num_tags == 2
            if elem_type == _ELEM_TYPE_TRIANGLE:
                block = np.frombuffer(f.read(count * 24), dtype="<i4").reshape(count, 6)
                tri_tags_list.append(block[:, 1])
                tri_nodes_list.append(block[:, 3:])
            elif elem_type == _ELEM_TYPE_TET:
                block = np.frombuffer(f.read(count * 28), dtype="<i4").reshape(count, 7)
                tet_tags_raw = block[:, 1]
                tet_nodes_raw = block[:, 3:]
            else:
                bytes_per = _ELEMENT_BYTES.get(elem_type, 0)
                if bytes_per:
                    f.read(count * bytes_per)
            consumed += count

        assert f.readline().decode().strip() == "$EndElements"

    assert tet_nodes_raw is not None and tet_tags_raw is not None

    valid_tet = np.isin(tet_tags_raw, _VOLUME_KEYS)
    tet_tags = tet_tags_raw[valid_tet]
    tet_nodes_filt = tet_nodes_raw[valid_tet]

    used_nodes = np.zeros(num_nodes + 1, dtype=bool)
    used_nodes[tet_nodes_filt.ravel()] = True
    unique_ids = np.flatnonzero(used_nodes)
    node_index = np.full(num_nodes + 1, -1, dtype=np.int32)
    node_index[unique_ids] = np.arange(unique_ids.size, dtype=np.int32)

    nodes_out = np.ascontiguousarray(nodes_xyz[unique_ids - 1])
    tet_nodes_out = node_index[tet_nodes_filt]

    if tri_nodes_list:
        tri_nodes_raw = np.concatenate(tri_nodes_list, axis=0)
        tri_tags_raw_all = np.concatenate(tri_tags_list, axis=0)
    else:
        tri_nodes_raw = np.empty((0, 3), dtype=np.int32)
        tri_tags_raw_all = np.empty((0,), dtype=np.int32)

    valid_surf = np.isin(tri_tags_raw_all, _SURFACE_KEYS)
    surf_tags = tri_tags_raw_all[valid_surf]
    surf_nodes_filt = tri_nodes_raw[valid_surf]

    surf_tris_out = node_index[surf_nodes_filt]
    assert np.all(surf_tris_out >= 0)

    return nodes_out, tet_nodes_out, tet_tags, surf_tris_out, surf_tags


@dataclass
class HeadMesh:
    """Store a tetrahedral head mesh.

    Node coordinates use millimetres. Element indices are zero-based.
    """

    nodes_mm: npt.NDArray[np.float64]
    tet_nodes: npt.NDArray[np.int32]
    tet_tags: npt.NDArray[np.int32]
    skin_tris: npt.NDArray[np.int32]

    @property
    def nodes_m(self) -> npt.NDArray[np.float64]:
        return self.nodes_mm * 1e-3

    @property
    def n_nodes(self) -> int:
        return self.nodes_mm.shape[0]

    @cached_property
    def skin_triangle_normals(self) -> npt.NDArray[np.float64]:
        return _skin_smoothed_triangle_normals(self.nodes_mm, self.skin_tris)

    @cached_property
    def tet_barycenters_mm(self) -> npt.NDArray[np.float64]:
        """Per-tetrahedron barycentre in mm."""
        return self.nodes_mm[self.tet_nodes].mean(axis=1)


def load_mesh(mesh_file: str | Path) -> HeadMesh:
    """Load a binary Gmsh 2.2 tetrahedral head mesh."""
    nodes, tet_nodes, tet_tags, surf_tris, surf_tags = parse_msh_binary(Path(mesh_file))
    skin_tris = surf_tris[surf_tags == SKIN_SURFACE_TAG]
    return HeadMesh(
        nodes_mm=np.ascontiguousarray(nodes, dtype=np.float64),
        tet_nodes=np.ascontiguousarray(tet_nodes, dtype=np.int32),
        tet_tags=np.ascontiguousarray(tet_tags, dtype=np.int32),
        skin_tris=np.ascontiguousarray(skin_tris, dtype=np.int32),
    )


def _skin_smoothed_triangle_normals(
    nodes_mm: npt.NDArray[np.float64], skin_tris: npt.NDArray[np.int32]
) -> npt.NDArray[np.float64]:
    """Compute smoothed unit normals for skin triangles.

    Sum area-weighted face normals at each node, smooth once, then average the three
    node normals for each triangle. Reindex skin nodes to limit the bincount arrays.
    """
    local_nodes, faces = np.unique(skin_tris, return_inverse=True)
    faces = faces.reshape(-1, 3)
    coords = nodes_mm[local_nodes]
    n_local = local_nodes.shape[0]

    side_a = coords[faces[:, 1]] - coords[faces[:, 0]]
    side_b = coords[faces[:, 2]] - coords[faces[:, 0]]
    face_normals = np.cross(side_a, side_b)

    nd = np.zeros((n_local, 3))
    normals = face_normals
    for _ in range(2):
        for i in range(3):
            nd[:, i] = np.bincount(faces.reshape(-1), np.repeat(normals[:, i], 3), n_local)
        normals = np.sum(nd[faces], axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
    nd /= np.linalg.norm(nd, axis=1)[:, None]

    tri_normals = np.mean(nd[faces], axis=1)
    tri_normals /= np.linalg.norm(tri_normals, axis=1)[:, None]
    return tri_normals
