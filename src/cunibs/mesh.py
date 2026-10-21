from __future__ import annotations

import hashlib
import mmap
import struct
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NoReturn

import h5py
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """The arrays a .msh file yields, filtered to known surface tags and 0-indexed.

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


class _ElementBlock(NamedTuple):
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


def _known_surfaces(block: npt.NDArray[np.int32]) -> _ElementBlock:
    """Filter per block so triangles with unknown tags never reach a contiguous buffer."""
    tags = block[:, 1]
    keep = np.isin(tags, _SURFACE_KEYS)
    if keep.all():
        return _ElementBlock(block[:, 3:], np.array(tags))
    return _ElementBlock(block[:, 3:][keep], tags[keep])


def _reject_unknown_volume_tags(tags: npt.NDArray[np.int32], path: Path) -> None:
    """Refuse a mesh whose tetrahedra carry tags this build has no tissue for.

    Dropping them instead would delete a whole tissue while the field computed from what is
    left still looks plausible. Surface tags are filtered rather than rejected: only the skin
    surface is consumed, and meshes routinely carry unrelated surface entities.
    """
    known = np.isin(tags, _VOLUME_KEYS)
    if known.all():
        return
    offenders, counts = np.unique(tags[~known], return_counts=True)
    histogram = ", ".join(f"{int(t)}: {int(c)}" for t, c in zip(offenders, counts, strict=True))
    raise ValueError(
        f"{path}: {int(counts.sum())} tetrahedra carry volume tags absent from "
        f"VOLUME_KEY_TO_LABEL ({histogram})"
    )


def _read_element_blocks(cursor: _MshCursor) -> tuple[list[_ElementBlock], list[_ElementBlock]]:
    """Walk $Elements, keeping tetrahedra and surface triangles and skipping the rest."""
    cursor.expect("$Elements")
    total = cursor.count("element")

    tet_blocks: list[_ElementBlock] = []
    surf_blocks: list[_ElementBlock] = []
    consumed = 0

    while consumed < total:
        elem_type, count, num_tags = cursor.unpack("<3i")
        if num_tags != 2:
            cursor.fail(f"element block declares {num_tags} tags, expected 2")
        if elem_type == _ELEM_TYPE_TRIANGLE:
            surf_blocks.append(_known_surfaces(cursor.block(_I4, count * 6).reshape(count, 6)))
        elif elem_type == _ELEM_TYPE_TET:
            block = cursor.block(_I4, count * 7).reshape(count, 7)
            tet_blocks.append(_ElementBlock(block[:, 3:], block[:, 1]))
        elif (n_elem_nodes := _ELEMENT_NODES.get(elem_type)) is not None:
            cursor.skip(count * (1 + num_tags + n_elem_nodes) * 4)
        else:
            cursor.fail(f"unsupported Gmsh element type {elem_type}")
        consumed += count

    cursor.expect("$EndElements")
    if not tet_blocks:
        cursor.fail("no tetrahedron block")
    return tet_blocks, surf_blocks


def _join_blocks(blocks: list[_ElementBlock], nodes_per_element: int) -> _ElementBlock:
    match blocks:
        case []:
            return _ElementBlock(
                np.empty((0, nodes_per_element), np.int32), np.empty(0, np.int32)
            )
        case [only]:
            return only
        case _:
            return _ElementBlock(
                np.concatenate([b.nodes for b in blocks]),
                np.concatenate([b.tags for b in blocks]),
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
    """Drop nodes that no tetrahedron references, and rebase indices to 0.

    Both index arrays arrive already shifted to 0-based. When every node is referenced
    that shift is the whole answer, the id-to-index map is the identity, so the
    gather is skipped entirely.
    """
    used = np.zeros(num_nodes, dtype=bool)
    # Viewing as unsigned turns a negative index, meaning a node id of 0 or less in the
    # file, into a huge one, so numpy's own bounds check covers both ends in one pass.
    try:
        used[tet_nodes.view(np.dtype(np.uint32))] = True
    except IndexError:
        raise _node_id_error(path, "a tetrahedron") from None

    if used.all():
        if surf_tris.size and surf_tris.view(np.dtype(np.uint32)).max() >= num_nodes:
            raise _node_id_error(path, "a surface triangle")
        return np.array(nodes_xyz), tet_nodes, surf_tris

    unique_ids = np.flatnonzero(used)
    node_index = np.full(num_nodes, -1, dtype=np.int32)
    node_index[unique_ids] = np.arange(unique_ids.size, dtype=np.int32)
    try:
        surf_tris = node_index[surf_tris.view(np.dtype(np.uint32))]
    except IndexError:
        raise _node_id_error(path, "a surface triangle") from None
    if (orphaned := surf_tris < 0).any():
        raise ValueError(
            f"{path}: {int(orphaned.any(axis=1).sum())} surface triangles reference nodes "
            "that no tetrahedron uses"
        )
    return np.ascontiguousarray(nodes_xyz[unique_ids]), node_index[tet_nodes], surf_tris


def _reject_nonfinite_nodes(nodes_mm: npt.NDArray[np.float64], path: Path) -> None:
    """Reject referenced coordinates that cannot define finite element geometry."""
    if not np.isfinite(nodes_mm).all():
        n_bad = int((~np.isfinite(nodes_mm).all(axis=1)).sum())
        subject = "referenced node has" if n_bad == 1 else "referenced nodes have"
        raise ValueError(f"{path}: {n_bad} {subject} non-finite coordinates")


def parse_msh_binary(mesh_file: Path) -> MeshArrays:
    """Parse a binary Gmsh 2.2 .msh file.

    The file is mapped rather than read, so node and element blocks are consumed as views
    and only the returned arrays are materialized. Every field of the result owns its
    data: nothing may alias the mapping, which is released when this frame returns.
    """
    with mesh_file.open("rb") as f:
        cursor = _MshCursor(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), mesh_file)
        _read_format(cursor)
        num_nodes, nodes_xyz = _read_nodes(cursor)
        tet_blocks, surf_blocks = _read_element_blocks(cursor)

        tet = _join_blocks(tet_blocks, _ELEMENT_NODES[_ELEM_TYPE_TET])
        # The tag check reads the contiguous copy, which is faster to scan than the stride-28 view
        # over the mapping it comes from.
        tet_tags = np.array(tet.tags)
        _reject_unknown_volume_tags(tet_tags, mesh_file)

        surface = _join_blocks(surf_blocks, _ELEMENT_NODES[_ELEM_TYPE_TRIANGLE])
        nodes_mm, tet_nodes, surf_tris = _reindex_nodes(
            nodes_xyz, num_nodes, tet.nodes - 1, surface.nodes - 1, mesh_file
        )
        _reject_nonfinite_nodes(nodes_mm, mesh_file)
        return MeshArrays(nodes_mm, tet_nodes, tet_tags, surf_tris, surface.tags)


def _pack_msh(
    nodes_mm: npt.ArrayLike,
    tets_1based: npt.ArrayLike,
    tet_tags: npt.ArrayLike,
    tris_1based: npt.ArrayLike,
    tri_tags: npt.ArrayLike,
    extra_blocks: Sequence[tuple[int, int, bytes]] = (),
) -> bytes:
    """Serialize a binary Gmsh 2.2 mesh in the dialect :func:`parse_msh_binary` reads.

    Element ids are assigned triangles first, then tetrahedra. ``extra_blocks`` appends raw
    ``(elem_type, count, payload)`` blocks after the tets, which the tests use to exercise the
    reader's element-skipping path.
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


_H5_MAGIC = b"\x89HDF\r\n\x1a\n"
_H5_FORMAT_VERSION = 1
# Tags are stored at the narrowest width that holds every known key and widened on load. Nothing
# is compressed, because loading has to stay faster than parsing the equivalent .msh.
_H5_DTYPES = {
    "nodes_mm": np.dtype("<f8"),
    "tet_nodes": np.dtype("<i4"),
    "tet_tags": np.dtype("u1"),
    "surf_tris": np.dtype("<i4"),
    "surf_tags": np.dtype("<u2"),
}
EEG_POSITIONS_GROUP = "eeg_positions"


def is_hdf5(path: str | Path) -> bool:
    with Path(path).open("rb") as f:
        return f.read(len(_H5_MAGIC)) == _H5_MAGIC


def _write_h5(arrays: MeshArrays, path: Path, caps: dict[str, str]) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = _H5_FORMAT_VERSION
        for name, dtype in _H5_DTYPES.items():
            f.create_dataset(name, data=np.asarray(getattr(arrays, name)).astype(dtype))
        if caps:
            group = f.create_group(EEG_POSITIONS_GROUP)
            for cap, text in caps.items():
                group[cap] = text


def parse_h5(mesh_file: Path) -> MeshArrays:
    """Read a mesh written by :meth:`HeadMesh.save`, applying the checks the .msh parser makes."""
    with h5py.File(mesh_file, "r") as f:
        stored = int(f.attrs.get("format_version", 0))
        if stored != _H5_FORMAT_VERSION:
            raise ValueError(
                f"{mesh_file}: mesh format version {stored} is not readable by this version of "
                f"cuNIBS (expected {_H5_FORMAT_VERSION})"
            )
        missing = [name for name in _H5_DTYPES if name not in f]
        if missing:
            raise ValueError(f"{mesh_file}: no {', '.join(missing)} dataset")
        arrays = MeshArrays(
            f["nodes_mm"][...],
            f["tet_nodes"][...],
            f["tet_tags"][...].astype(np.int32),
            f["surf_tris"][...],
            f["surf_tags"][...].astype(np.int32),
        )

    for name, width in (("nodes_mm", 3), ("tet_nodes", 4), ("surf_tris", 3)):
        shape = getattr(arrays, name).shape
        if len(shape) != 2 or shape[1] != width:
            raise ValueError(f"{mesh_file}: {name} has shape {shape}, expected (n, {width})")
    if arrays.tet_tags.shape[0] != arrays.tet_nodes.shape[0]:
        raise ValueError(f"{mesh_file}: tet_tags and tet_nodes differ in length")
    if arrays.surf_tags.shape[0] != arrays.surf_tris.shape[0]:
        raise ValueError(f"{mesh_file}: surf_tags and surf_tris differ in length")
    if arrays.tet_nodes.shape[0] == 0:
        raise ValueError(f"{mesh_file}: no tetrahedra")
    _reject_unknown_volume_tags(arrays.tet_tags, mesh_file)
    n_nodes = arrays.nodes_mm.shape[0]
    for index, what in (
        (arrays.tet_nodes, "a tetrahedron"),
        (arrays.surf_tris, "a surface triangle"),
    ):
        if index.size and index.view(np.dtype(np.uint32)).max() >= n_nodes:
            raise _node_id_error(mesh_file, what)
    _reject_nonfinite_nodes(arrays.nodes_mm, mesh_file)
    return arrays


def parse_mesh(mesh_file: Path) -> MeshArrays:
    """Parse a mesh in either format :meth:`HeadMesh.save` writes, told apart by content."""
    return parse_h5(mesh_file) if is_hdf5(mesh_file) else parse_msh_binary(mesh_file)


def _eeg_caps(source: str | Path | None) -> dict[str, str]:
    """Cap name -> CSV text, from a CHARM ``eeg_positions`` directory or one cap CSV in it."""
    if source is None:
        return {}
    source = Path(source)
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"{source}: no EEG cap CSV files")
    return {p.stem: p.read_text() for p in files}


def _on_host(array: Any) -> npt.NDArray[Any]:
    """Copy a device-resident array to the host, passing a host array straight through.

    Duck-typed on cupy's ``.get()`` so that this module stays importable without cupy.
    """
    return array if isinstance(array, np.ndarray) else array.get()


class HeadMesh:
    """Store a tetrahedral head mesh.

    Node coordinates use millimetres. Element indices are zero-based.

    Each array may be given on the device instead of on the host, in which case it is copied down
    the first time something reads it and not before. :func:`~cunibs.fem.solve.build_context`
    builds the reordered mesh that way: a forward run reads neither the connectivity nor the tags
    on the host, so nothing pays for a copy back.

    Every tissue boundary the file carries is held as the parser produced it, one triangle array
    and one tag array. :meth:`surface` and :attr:`skin_tris` split out a tag on demand, so a
    forward run pays for none of the surfaces it does not touch.

    Derived surface geometry is not here. The smoothed skin normals a placement projects against
    are built on the device with the rest of the solver context; see
    :func:`~cunibs.fem.solve.skin_triangle_normals` and ``SolverContext.skin_tri_normals``.
    """

    def __init__(
        self,
        nodes_mm: npt.NDArray[np.float64],
        tet_nodes: npt.NDArray[np.int32],
        tet_tags: npt.NDArray[np.int32],
        surf_tris: npt.NDArray[np.int32],
        surf_tags: npt.NDArray[np.int32],
    ) -> None:
        self._nodes_mm = nodes_mm
        self._tet_nodes = tet_nodes
        self._tet_tags = tet_tags
        self._surf_tris = surf_tris
        self._surf_tags = surf_tags
        self._surfaces: dict[int, npt.NDArray[np.int32]] = {}

    # None of the cached_property members below holds a lock. cached_property has held none since
    # 3.12, so under the free-threaded build two threads racing on one can both compute it. All are
    # pure functions of the arrays handed to __init__, so the loser's result is identical and the
    # dict store that publishes it is atomic. A lock would buy nothing here and would make HeadMesh
    # unpicklable.
    @cached_property
    def nodes_mm(self) -> npt.NDArray[np.float64]:
        return _on_host(self._nodes_mm)

    @cached_property
    def tet_nodes(self) -> npt.NDArray[np.int32]:
        return _on_host(self._tet_nodes)

    @cached_property
    def tet_tags(self) -> npt.NDArray[np.int32]:
        return _on_host(self._tet_tags)

    @property
    def skin_tris(self) -> npt.NDArray[np.int32]:
        """The scalp triangles, on the host, or empty where the mesh carries no scalp surface.

        Empty rather than an error, unlike :meth:`surface`: a mesh without a scalp loads and
        solves, and it is only a coil placement that has nothing to project onto.
        """
        return self._tagged(SKIN_SURFACE_TAG)

    @property
    def n_nodes(self) -> int:
        # Off the source array, so asking a device-backed mesh its size does not fetch it.
        return self._nodes_mm.shape[0]

    @property
    def n_tets(self) -> int:
        return self._tet_nodes.shape[0]

    @cached_property
    def surf_tags(self) -> npt.NDArray[np.int32]:
        """Surface tag per triangle in :attr:`surf_tris`, on the host."""
        return _on_host(self._surf_tags)

    @cached_property
    def surf_tris(self) -> npt.NDArray[np.int32]:
        """Every surface triangle, of every tag, on the host."""
        return _on_host(self._surf_tris)

    @property
    def surface_tags(self) -> tuple[int, ...]:
        """Surface tags this mesh carries, in ascending order."""
        return tuple(int(t) for t in np.unique(self.surf_tags))

    def surface(self, tag: int, *, host: bool = True) -> Any:
        """The triangles of one surface tag, on the host.

        ``tag`` is a surface tag (1001-1010, 1099), not a volume tag; see
        :data:`SURFACE_KEY_TO_LABEL`. Raises when the mesh carries no such surface.

        ``host=False`` returns them wherever the mesh holds its surfaces instead, so a
        device-backed mesh, such as ``Subject.mesh``, splits them out on the device without
        copying every surface to the host. That form is not cached.
        """
        tris = self._tagged(tag) if host else self._surf_tris[self._surf_tags == tag]
        if tris.shape[0] == 0:
            known = ", ".join(str(t) for t in self.surface_tags) or "none"
            raise KeyError(f"this mesh carries no surface with tag {tag} (has: {known})")
        return tris

    def _tagged(self, tag: int) -> npt.NDArray[np.int32]:
        """Triangles of one tag, split out of :attr:`surf_tris` on first request and cached."""
        cached = self._surfaces.get(tag)
        if cached is None:
            cached = np.ascontiguousarray(self.surf_tris[self.surf_tags == tag])
            self._surfaces[tag] = cached
        return cached

    @cached_property
    def tet_barycenters_mm(self) -> npt.NDArray[np.float64]:
        """Per-tetrahedron barycentre in mm."""
        return self.nodes_mm[self.tet_nodes].mean(axis=1)

    @cached_property
    def content_hash(self) -> str:
        """A digest of the arrays that define this mesh, for tying a result to its input.

        Covers coordinates, connectivity and every tag array, each with its shape and dtype, so
        two meshes agree here only if they would produce the same field.

        The mesh :func:`~cunibs.fem.solve.build_context` returns is a Morton reordering of the
        loaded one and so hashes differently; the identity worth recording is the loaded mesh's.
        """
        h = hashlib.blake2b(digest_size=16)
        for name in ("nodes_mm", "tet_nodes", "tet_tags", "surf_tris", "surf_tags"):
            a = np.ascontiguousarray(getattr(self, name))
            h.update(f"{name}{a.shape}{a.dtype.str}".encode())
            h.update(a)
        return h.hexdigest()

    def save(self, path: str | Path, *, eeg_positions: str | Path | None = None) -> None:
        """Write this mesh to ``path``, as HDF5 for ``.h5`` or binary Gmsh 2.2 for ``.msh``.

        Either file loads back through :func:`load_mesh` to the same :attr:`content_hash`. The
        HDF5 file is smaller and faster to load.

        ``eeg_positions`` is a CHARM ``eeg_positions`` directory, or one cap CSV from it. Each cap
        is stored verbatim in the HDF5 file, so :func:`~cunibs.head.read_cap` reads electrode
        positions from the mesh file itself. A .msh file has nowhere to put them.
        """
        path = Path(path)
        if path.suffix == ".msh":
            if eeg_positions is not None:
                raise ValueError("a .msh file cannot carry EEG positions; save as .h5")
            path.write_bytes(
                _pack_msh(
                    self.nodes_mm,
                    self.tet_nodes + 1,
                    self.tet_tags,
                    self.surf_tris + 1,
                    self.surf_tags,
                )
            )
        elif path.suffix in (".h5", ".hdf5"):
            arrays = MeshArrays(
                self.nodes_mm, self.tet_nodes, self.tet_tags, self.surf_tris, self.surf_tags
            )
            _write_h5(arrays, path, _eeg_caps(eeg_positions))
        else:
            raise ValueError(f"{path}: expected a .h5 or .msh suffix")

    def __repr__(self) -> str:
        return f"HeadMesh(n_nodes={self.n_nodes}, n_tets={self.n_tets})"


def load_mesh(mesh_file: str | Path) -> HeadMesh:
    """Load a tetrahedral head mesh from binary Gmsh 2.2 or from :meth:`HeadMesh.save`'s HDF5."""
    return HeadMesh(*parse_mesh(Path(mesh_file)))


# Tissues a head model has to have for a TMS field to mean anything. Deliberately not the whole
# table: CHARM emits no tag 4 (it splits bone into 7 and 8), and a cropped patch legitimately has
# no eyes or muscle, so requiring every tag would warn on every real mesh.
_ESSENTIAL_TAGS = (1, 2, 3, 5)
# Bone is required as a group rather than by tag: 4 is the legacy single-compartment tag and 7/8
# are what CHARM writes.
_BONE_TAGS = (4, 7, 8)

# A head in millimetres spans a few hundred; one in metres spans a fraction of one. Anything inside
# this is not a head at either scale, and the likeliest explanation is the unit.
_MIN_PLAUSIBLE_EXTENT_MM = 20.0


@dataclass(frozen=True)
class MeshReport:
    """What :func:`validate_mesh` found, without a GPU and without raising.

    ``ok`` is false when ``errors`` is non-empty; those are the conditions that would make
    :func:`load_mesh` or :func:`~cunibs.fem.build_context` refuse the mesh. ``warnings`` are things
    that load and solve but are usually not what was intended.

    ``missing_tags`` names only the tissues a head model has to have, not every tag in the table.

    ``min_flatness`` is a scale-free element quality, ``12·(3V)^(2/3) / Σ|edge|²``, which is 1 for
    a regular tetrahedron and tends to 0 as one flattens. Quality is reported this way rather than
    as a volume because element size is graded across tissues: the smallest tetrahedron on a
    healthy head mesh is smaller than a degenerate one elsewhere, so no volume threshold separates
    them, while flatness does by orders of magnitude.
    """

    path: Path
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    n_nodes: int = 0
    n_tets: int = 0
    n_skin_tris: int = 0
    tag_counts: dict[int, int] = dataclass_field(default_factory=dict)
    missing_tags: tuple[int, ...] = ()
    bbox_mm: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    n_nonpositive_volume: int = 0
    min_abs_volume_mm3: float | None = None
    min_flatness: float | None = None


def _volumes_and_flatness(
    nodes_mm: npt.NDArray[np.float64], tet_nodes: npt.NDArray[np.int32]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Signed volumes and flatness in one pass; the four vertex gathers dominate.

    The volume sign convention is the one ``build_context`` checks.
    """
    p0, e1, e2, e3 = (nodes_mm.take(column, axis=0) for column in tet_nodes.T)
    e1 -= p0
    e2 -= p0
    e3 -= p0
    vols = np.einsum("ij,ij->i", np.cross(e1, e2), e3) / 6.0

    def sq(d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.einsum("ij,ij->i", d, d)

    # Over the six edges, sum |pi - pj|^2 = 4 sum |ei|^2 - |sum ei|^2 with e0 = 0, which avoids
    # materialising the three edges that do not touch p0.
    edges_sq = 4.0 * (sq(e1) + sq(e2) + sq(e3)) - sq(e1 + e2 + e3)
    with np.errstate(divide="ignore", invalid="ignore"):
        flatness = 12.0 * np.cbrt(3.0 * np.abs(vols)) ** 2 / edges_sq
    return vols, np.where(np.isfinite(flatness), flatness, 0.0)


def validate_mesh(mesh_file: str | Path) -> MeshReport:
    """Check a mesh on the host, without building a solver context.

    Screens for everything a mesh can be rejected for, including the zero and inverted tetrahedra
    that :func:`~cunibs.fem.build_context` otherwise only catches after a CUDA context, a full
    upload and a Morton reorder.

    Problems are collected rather than raised, so a cohort driver gets a reason per mesh instead of
    an exception on the first bad one. ``build_context`` remains the authority on the volume test:
    host float64 and the device kernel can disagree in the last bit on an element that is
    degenerate either way.
    """
    path = Path(mesh_file)
    try:
        arrays = parse_mesh(path)
    except (ValueError, OSError) as exc:
        return MeshReport(path=path, ok=False, errors=(str(exc),))

    errors: list[str] = []
    warnings: list[str] = []
    nodes_mm, tet_nodes = arrays.nodes_mm, arrays.tet_nodes
    vols, flatness = _volumes_and_flatness(nodes_mm, tet_nodes)

    n_bad = int(np.count_nonzero((vols <= 0.0) | ~np.isfinite(vols)))
    if n_bad:
        subject = "tetrahedron has" if n_bad == 1 else "tetrahedra have"
        errors.append(f"{n_bad} {subject} zero, negative or non-finite volume")

    tags, counts = np.unique(arrays.tet_tags, return_counts=True)
    tag_counts = {int(t): int(c) for t, c in zip(tags, counts, strict=True)}
    missing = tuple(t for t in _ESSENTIAL_TAGS if t not in tag_counts)
    if missing:
        named = ", ".join(f"{t} ({VOLUME_KEY_TO_LABEL[t]})" for t in missing)
        warnings.append(f"no tetrahedra carry volume tags {named}")
    if not any(t in tag_counts for t in _BONE_TAGS):
        warnings.append(
            "no tetrahedra carry a bone tag (4 average, or 7 cortical and 8 cancellous); the "
            "skull is what shapes the field"
        )

    n_skin_tris = int(np.count_nonzero(arrays.surf_tags == SKIN_SURFACE_TAG))
    if n_skin_tris == 0:
        warnings.append(
            f"no surface triangles carry the scalp tag {SKIN_SURFACE_TAG}; the mesh loads but a "
            "coil placement has nothing to project onto"
        )

    lo, hi = nodes_mm.min(axis=0), nodes_mm.max(axis=0)
    if (extent := float((hi - lo).max())) < _MIN_PLAUSIBLE_EXTENT_MM:
        warnings.append(
            f"the mesh spans {extent:.3g} in its longest axis; coordinates are "
            "read as millimetres, so a head given in metres would look like this"
        )

    return MeshReport(
        path=path,
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        n_nodes=nodes_mm.shape[0],
        n_tets=tet_nodes.shape[0],
        n_skin_tris=n_skin_tris,
        tag_counts=tag_counts,
        missing_tags=missing,
        bbox_mm=(tuple(lo.tolist()), tuple(hi.tolist())),
        n_nonpositive_volume=n_bad,
        min_abs_volume_mm3=float(np.abs(vols).min()) if vols.size else None,
        min_flatness=float(flatness.min()) if flatness.size else None,
    )
