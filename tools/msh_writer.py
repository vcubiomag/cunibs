"""Serialize binary Gmsh 2.2 meshes in the dialect ``cunibs.mesh.parse_msh_binary`` reads.

Shared by ``tests/test_mesh.py`` and ``tools/make_test_patch.py`` via the ``pythonpath``
setting in ``pyproject.toml``.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

_NODE_DTYPE = np.dtype([("id", "<i4"), ("xyz", "<f8", (3,))])


def pack_msh(
    nodes_mm: npt.ArrayLike,
    tets_1based: npt.ArrayLike,
    tet_tags: npt.ArrayLike,
    tris_1based: npt.ArrayLike,
    tri_tags: npt.ArrayLike,
    extra_blocks: Sequence[tuple[int, int, bytes]] = (),
) -> bytes:
    """Serialize a binary Gmsh 2.2 mesh.

    Element ids are assigned triangles first, then tetrahedra. Every block carries the two
    tags ``parse_msh_binary`` asserts on. ``extra_blocks`` appends raw ``(elem_type, count,
    payload)`` blocks after the tets, for exercising the reader's element-skipping path.
    """
    nodes = np.ascontiguousarray(nodes_mm, dtype=np.float64).reshape(-1, 3)
    tets = np.ascontiguousarray(tets_1based, dtype=np.int32).reshape(-1, 4)
    tris = np.ascontiguousarray(tris_1based, dtype=np.int32).reshape(-1, 3)
    tet_tags = np.ascontiguousarray(tet_tags, dtype=np.int32).reshape(-1)
    tri_tags = np.ascontiguousarray(tri_tags, dtype=np.int32).reshape(-1)
    if tet_tags.shape[0] != tets.shape[0] or tri_tags.shape[0] != tris.shape[0]:
        raise ValueError("Tag counts must match element counts.")

    n_nodes, n_tri, n_tet = nodes.shape[0], tris.shape[0], tets.shape[0]

    node_rec = np.empty(n_nodes, dtype=_NODE_DTYPE)
    node_rec["id"] = np.arange(1, n_nodes + 1, dtype=np.int32)
    node_rec["xyz"] = nodes

    tri_blk = np.empty((n_tri, 6), dtype="<i4")
    tri_blk[:, 0] = np.arange(1, n_tri + 1)
    tri_blk[:, 1] = tri_tags
    tri_blk[:, 2] = tri_tags
    tri_blk[:, 3:] = tris

    tet_blk = np.empty((n_tet, 7), dtype="<i4")
    tet_blk[:, 0] = np.arange(n_tri + 1, n_tri + n_tet + 1)
    tet_blk[:, 1] = tet_tags
    tet_blk[:, 2] = tet_tags
    tet_blk[:, 3:] = tets

    total = n_tri + n_tet + sum(count for _, count, _ in extra_blocks)

    parts = [
        b"$MeshFormat\n2.2 1 8\n",
        struct.pack("<i", 1),
        b"\n$EndMeshFormat\n",
        b"$Nodes\n%d\n" % n_nodes,
        node_rec.tobytes(),
        b"$EndNodes\n",
        b"$Elements\n%d\n" % total,
        struct.pack("<3i", 2, n_tri, 2),
        tri_blk.tobytes(),
        struct.pack("<3i", 4, n_tet, 2),
        tet_blk.tobytes(),
    ]
    for elem_type, count, payload in extra_blocks:
        parts.append(struct.pack("<3i", elem_type, count, 2))
        parts.append(payload)
    parts.append(b"$EndElements\n")
    return b"".join(parts)
