from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, NamedTuple, NoReturn

import numpy as np
import numpy.typing as npt

_ELEM_TYPE_TRIANGLE = 2
_ELEM_TYPE_TET = 4
_NODE_DTYPE = np.dtype([("id", "<i4"), ("xyz", "<f8", (3,))])
_I4 = np.dtype("<i4")
_ELEMENT_NODES = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 15: 1}

SKIN_SURFACE_TAG = 1005

type TissueLabel = Literal[
    "white_matter",
    "gray_matter",
    "csf",
    "average_bone",
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
    4: "average_bone",
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
    1004: "average_bone",
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


class MeshArrays(NamedTuple):
    """The arrays a .msh file yields, filtered to known tissue tags and 0-indexed.

    nodes_mm  : (N, 3) float64  node coordinates in millimetres
    tet_nodes : (M, 4) int32    tetrahedron node indices
    tet_tags  : (M,)   int32    tissue label per tetrahedron
    surf_tris : (S, 3) int32    surface triangle node indices
    surf_tags : (S,)   int32    tissue label per surface triangle
    """

    nodes_mm: npt.NDArray[np.float64]
    tet_nodes: npt.NDArray[np.int32]
    tet_tags: npt.NDArray[np.int32]
    surf_tris: npt.NDArray[np.int32]
    surf_tags: npt.NDArray[np.int32]


class _SurfaceBlock(NamedTuple):
    nodes: npt.NDArray[np.int32]
    tags: npt.NDArray[np.int32]


class _MshCursor:
    """Byte cursor over a memory-mapped mesh, for Gmsh 2.2's interleaved text and binary."""

    __slots__ = ("mm", "path", "pos")

    def __init__(self, mm: mmap.mmap, path: Path) -> None:
        self.mm = mm
        self.path = path
        self.pos = 0

    def fail(self, problem: str) -> NoReturn:
        raise ValueError(f"{self.path}: {problem}")

    def line(self) -> str:
        end = self.mm.find(b"\n", self.pos)
        if end < 0:
            self.fail("truncated: expected a newline-terminated line")
        text = self.mm[self.pos : end].decode(errors="replace").strip()
        self.pos = end + 1
        return text

    def expect(self, marker: str) -> None:
        if (found := self.line()) != marker:
            self.fail(f"expected {marker}, found {found!r}")

    def count(self, what: str) -> int:
        if not (text := self.line()).isdigit():
            self.fail(f"expected a {what} count, found {text!r}")
        return int(text)

    def unpack(self, fmt: str) -> tuple[int, ...]:
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.mm):
            self.fail("truncated: expected another element block header")
        values = struct.unpack_from(fmt, self.mm, self.pos)
        self.pos += size
        return values

    def block(self, dtype: np.dtype, count: int) -> npt.NDArray:
        n_bytes = count * dtype.itemsize
        if count < 0 or self.pos + n_bytes > len(self.mm):
            self.fail("truncated: a block extends past the end of the file")
        out = np.frombuffer(self.mm, dtype=dtype, count=count, offset=self.pos)
        self.pos += n_bytes
        return out

    def skip(self, n_bytes: int) -> None:
        self.pos += n_bytes


def _read_format(cursor: _MshCursor) -> None:
    cursor.expect("$MeshFormat")
    fields = cursor.line().split()
    if fields[1:] != ["1", "8"]:
        cursor.fail(f"expected a binary Gmsh 2.2 header, found {' '.join(fields)!r}")
    if cursor.unpack("<i")[0] != 1:
        cursor.fail("byte-order marker is not 1; big-endian meshes are not supported")
    cursor.line()
    cursor.expect("$EndMeshFormat")


def _read_nodes(cursor: _MshCursor) -> tuple[int, npt.NDArray[np.float64]]:
    cursor.expect("$Nodes")
    num_nodes = cursor.count("node")
    nodes_xyz = cursor.block(_NODE_DTYPE, num_nodes)["xyz"]
    cursor.expect("$EndNodes")
    return num_nodes, nodes_xyz


def _known_surfaces(block: npt.NDArray[np.int32]) -> _SurfaceBlock:
    """Filter per block so triangles with unknown tags never reach a contiguous buffer."""
    tags = block[:, 1]
    keep = np.isin(tags, _SURFACE_KEYS)
    if keep.all():
        return _SurfaceBlock(block[:, 3:], np.array(tags))
    return _SurfaceBlock(block[:, 3:][keep], tags[keep])


def _read_element_blocks(
    cursor: _MshCursor,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], list[_SurfaceBlock]]:
    """Walk $Elements, keeping tetrahedra and surface triangles and skipping the rest."""
    cursor.expect("$Elements")
    total = cursor.count("element")

    tet_nodes: npt.NDArray[np.int32] | None = None
    tet_tags: npt.NDArray[np.int32] | None = None
    surfaces: list[_SurfaceBlock] = []
    consumed = 0

    while consumed < total:
        elem_type, count, num_tags = cursor.unpack("<3i")
        if num_tags != 2:
            cursor.fail(f"element block declares {num_tags} tags, expected 2")
        if elem_type == _ELEM_TYPE_TRIANGLE:
            surfaces.append(_known_surfaces(cursor.block(_I4, count * 6).reshape(count, 6)))
        elif elem_type == _ELEM_TYPE_TET:
            block = cursor.block(_I4, count * 7).reshape(count, 7)
            tet_tags, tet_nodes = block[:, 1], block[:, 3:]
        elif (n_elem_nodes := _ELEMENT_NODES.get(elem_type)) is not None:
            cursor.skip(count * (1 + num_tags + n_elem_nodes) * 4)
        else:
            cursor.fail(f"unsupported Gmsh element type {elem_type}")
        consumed += count

    cursor.expect("$EndElements")
    if tet_nodes is None or tet_tags is None:
        cursor.fail("no tetrahedron block")
    return tet_nodes, tet_tags, surfaces


def _join_surfaces(surfaces: list[_SurfaceBlock]) -> _SurfaceBlock:
    match surfaces:
        case []:
            return _SurfaceBlock(np.empty((0, 3), np.int32), np.empty(0, np.int32))
        case [only]:
            return only
        case _:
            return _SurfaceBlock(
                np.concatenate([s.nodes for s in surfaces]),
                np.concatenate([s.tags for s in surfaces]),
            )


def _node_id_error(path: Path, what: str) -> ValueError:
    return ValueError(f"{path}: {what} references a node id outside the node table")


def _reindex_nodes(
    nodes_xyz: npt.NDArray[np.float64],
    num_nodes: int,
    tet_nodes: npt.NDArray[np.int32],
    surf_tris: npt.NDArray[np.int32],
    path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Drop nodes that no surviving tetrahedron references, and rebase indices to 0.

    Both index arrays arrive already shifted to 0-based. When every node is referenced
    that shift is the whole answer, the id-to-index map is the identity, so the
    gather is skipped entirely.
    """
    used = np.zeros(num_nodes, dtype=bool)
    # Viewing as unsigned turns a negative index, meaning a node id of 0 or less in the
    # file, into a huge one, so numpy's own bounds check covers both ends in one pass.
    try:
        used[tet_nodes.view(np.uint32)] = True
    except IndexError:
        raise _node_id_error(path, "a tetrahedron") from None

    if used.all():
        if surf_tris.size and (surf_tris.min() < 0 or surf_tris.max() >= num_nodes):
            raise _node_id_error(path, "a surface triangle")
        return np.array(nodes_xyz), tet_nodes, surf_tris

    unique_ids = np.flatnonzero(used)
    node_index = np.full(num_nodes, -1, dtype=np.int32)
    node_index[unique_ids] = np.arange(unique_ids.size, dtype=np.int32)
    try:
        surf_tris = node_index[surf_tris.view(np.uint32)]
    except IndexError:
        raise _node_id_error(path, "a surface triangle") from None
    if (orphaned := surf_tris < 0).any():
        raise ValueError(
            f"{path}: {int(orphaned.any(axis=1).sum())} surface triangles reference nodes "
            "that were dropped with a filtered tetrahedron"
        )
    return np.ascontiguousarray(nodes_xyz[unique_ids]), node_index[tet_nodes], surf_tris


def parse_msh_binary(mesh_file: Path) -> MeshArrays:
    """Parse a binary Gmsh 2.2 .msh file.

    The file is mapped rather than read, so node and element blocks are consumed as views
    and only the returned arrays are materialized. Every field of the result owns its
    data: nothing may alias the mapping, which is released when this frame returns.
    """
    with Path.open(mesh_file, "rb") as f:
        cursor = _MshCursor(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), mesh_file)
        _read_format(cursor)
        num_nodes, nodes_xyz = _read_nodes(cursor)
        tet_nodes_raw, tet_tags_raw, surfaces = _read_element_blocks(cursor)

        known = np.isin(tet_tags_raw, _VOLUME_KEYS)
        if known.all():
            tet_tags, tet_nodes = np.array(tet_tags_raw), tet_nodes_raw
        else:
            tet_tags, tet_nodes = tet_tags_raw[known], tet_nodes_raw[known]

        surface = _join_surfaces(surfaces)
        nodes_mm, tet_nodes, surf_tris = _reindex_nodes(
            nodes_xyz, num_nodes, tet_nodes - 1, surface.nodes - 1, mesh_file
        )
        return MeshArrays(nodes_mm, tet_nodes, tet_tags, surf_tris, surface.tags)


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
    def n_nodes(self) -> int:
        return self.nodes_mm.shape[0]

    # cached_property holds no lock as of 3.12, so under the free-threaded build two threads
    # racing on either of these can both compute it. Both are pure functions of the arrays
    # above, so the loser's result is identical and the dict store that publishes it is atomic.
    # A lock would buy nothing here and would make HeadMesh unpicklable.
    @cached_property
    def skin_triangle_normals(self) -> npt.NDArray[np.float64]:
        return _skin_smoothed_triangle_normals(self.nodes_mm, self.skin_tris)

    @cached_property
    def tet_barycenters_mm(self) -> npt.NDArray[np.float64]:
        """Per-tetrahedron barycentre in mm."""
        return self.nodes_mm[self.tet_nodes].mean(axis=1)


def load_mesh(mesh_file: str | Path) -> HeadMesh:
    """Load a binary Gmsh 2.2 tetrahedral head mesh."""
    arrays = parse_msh_binary(Path(mesh_file))
    skin_tris = arrays.surf_tris[arrays.surf_tags == SKIN_SURFACE_TAG]
    return HeadMesh(
        nodes_mm=np.ascontiguousarray(arrays.nodes_mm, dtype=np.float64),
        tet_nodes=np.ascontiguousarray(arrays.tet_nodes, dtype=np.int32),
        tet_tags=np.ascontiguousarray(arrays.tet_tags, dtype=np.int32),
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
