"""Build synthetic test fixtures."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from cunibs.mesh import HeadMesh

_CUBE_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)
_CUBE_TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int32,
)
_CUBE_TRIS = np.array(
    [
        [0, 2, 3],
        [0, 3, 1],  # z=0
        [4, 5, 7],
        [4, 7, 6],  # z=1
        [0, 1, 5],
        [0, 5, 4],  # y=0
        [2, 6, 7],
        [2, 7, 3],  # y=1
        [0, 4, 6],
        [0, 6, 2],  # x=0
        [1, 3, 7],
        [1, 7, 5],  # x=1
    ],
    dtype=np.int32,
)


@pytest.fixture
def cube_mesh() -> HeadMesh:
    """Return a solvable 100 mm gray-matter cube."""
    nodes_mm = _CUBE_CORNERS * 100.0
    tet_tags = np.full(_CUBE_TETS.shape[0], 2, dtype=np.int32)  # gray matter
    return HeadMesh(
        nodes_mm=np.ascontiguousarray(nodes_mm),
        tet_nodes=_CUBE_TETS.copy(),
        tet_tags=tet_tags,
        skin_tris=_CUBE_TRIS.copy(),
    )


def build_binary_msh(
    nodes_mm: np.ndarray,
    tets_1based: np.ndarray,
    tet_tags: np.ndarray,
    tris_1based: np.ndarray,
    tri_tags: np.ndarray,
) -> bytes:
    """Serialize a minimal binary Gmsh 2.2 mesh matching ``parse_msh_binary``'s reader."""
    buf = bytearray()
    buf += b"$MeshFormat\n2.2 1 8\n"
    buf += struct.pack("<i", 1)
    buf += b"\n$EndMeshFormat\n"

    buf += b"$Nodes\n%d\n" % nodes_mm.shape[0]
    for i, (x, y, z) in enumerate(nodes_mm, start=1):
        buf += struct.pack("<i3d", i, float(x), float(y), float(z))
    buf += b"$EndNodes\n"

    total = tets_1based.shape[0] + tris_1based.shape[0]
    buf += b"$Elements\n%d\n" % total
    buf += struct.pack("<3i", 2, tris_1based.shape[0], 2)
    for e, (tri, tag) in enumerate(zip(tris_1based, tri_tags), start=1):
        buf += struct.pack("<6i", e, int(tag), int(tag), *[int(n) for n in tri])
    base = tris_1based.shape[0]
    buf += struct.pack("<3i", 4, tets_1based.shape[0], 2)
    for e, (tet, tag) in enumerate(zip(tets_1based, tet_tags), start=base + 1):
        buf += struct.pack("<7i", e, int(tag), int(tag), *[int(n) for n in tet])
    buf += b"$EndElements\n"
    return bytes(buf)


def synthetic_coil():
    """Return a two-dipole coil in its local frame."""
    from cunibs.coil import Coil

    positions_m = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    moments = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    return Coil(positions_m=positions_m, moments=moments, name="synthetic", didt_max=1e6)
