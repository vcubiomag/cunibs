"""Ask SimNIBS for a field or a timing, through ``simnibs_child.py`` in SimNIBS's own Python.

A request is one ``.npz`` holding the operation, its parameters and its arrays, mesh included,
in cuNIBS's element order, so both codes' fields line up element for element.

``--simnibs-python`` runs the child here. ``--simnibs-exchange DIR`` instead writes requests to a
directory that can be carried to another host and answered there with
``python simnibs_child.py serve DIR``; a later run reads the answers. A request is named by its
content, so an answer can only be read back by the identical question.

``call`` returns ``None`` when SimNIBS cannot answer yet, and the benchmark leaves out what
needed it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

CHILD = Path(__file__).with_name("simnibs_child.py")
SOLVERS = ("hypre", "pardiso", "mumps", "petsc_pardiso")


def write_request(path: Path, op: str, params: dict, arrays: dict[str, np.ndarray]) -> None:
    header = json.dumps({"op": op, **params}, sort_keys=True).encode()
    np.savez(path, __request__=np.frombuffer(header, np.uint8), **arrays)


def read_response(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files if k != "__response__"}
        return json.loads(f["__response__"].tobytes()), arrays


# Coil frames and target points come out of GPU kernels whose last bits can differ from run to
# run and between GPUs. Rounded to 1e-9 (mm, or unitless for a rotation), one question gets one
# name, at a precision far below anything SimNIBS resolves. Mesh arrays are read from the file and
# are left exact.
ROUNDED = ("frames", "frame")


def canonical(
    params: dict, arrays: dict[str, np.ndarray]
) -> tuple[dict, dict[str, np.ndarray]]:
    def rounded(v):
        if isinstance(v, float):
            return round(v, 9)
        return [rounded(x) for x in v] if isinstance(v, list) else v

    params = {k: rounded(v) for k, v in params.items()}
    arrays = {k: np.round(v, 9) if k in ROUNDED else v for k, v in arrays.items()}
    return params, arrays


def request_key(op: str, params: dict, arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256(json.dumps({"op": op, **params}, sort_keys=True).encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}{a.shape}{a.dtype.str}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


def mesh_arrays(mesh) -> dict[str, np.ndarray]:
    """What the child needs to rebuild ``mesh`` (a :class:`~cunibs.HeadMesh`) in SimNIBS."""
    return {
        "nodes": mesh.nodes_mm,
        "tets": mesh.tet_nodes,
        "tet_tags": mesh.tet_tags,
        "tris": mesh.surf_tris,
        "tri_tags": mesh.surf_tags,
    }


class Simnibs:
    def __init__(self, python: str | None, exchange: Path | None, scratch: Path) -> None:
        self.python = python
        self.exchange = exchange
        self.scratch = scratch
        self.pending = 0
        self.unanswered = 0

    @property
    def available(self) -> bool:
        return bool(self.python or self.exchange)

    def call(self, op: str, params: dict | None = None, **arrays: np.ndarray):
        """``(meta, arrays)`` from the child, or ``None`` if SimNIBS cannot answer yet."""
        params, arrays = canonical(params or {}, arrays)
        if self.exchange:
            key = request_key(op, params, arrays)
            response = self.exchange / f"{key}.response.npz"
            if response.exists():
                return read_response(response)
            self.exchange.mkdir(parents=True, exist_ok=True)
            write_request(self.exchange / f"{key}.request.npz", op, params, arrays)
            self.pending += 1
            return None
        if not self.python:
            self.unanswered += 1
            return None

        self.scratch.mkdir(parents=True, exist_ok=True)
        request = self.scratch / f"{os.getpid()}.request.npz"
        response = self.scratch / f"{os.getpid()}.response.npz"
        write_request(request, op, params, arrays)
        env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
        try:
            subprocess.run(
                [self.python, str(CHILD), "run", str(request), str(response)],
                env=env,
                check=True,
            )
            return read_response(response)
        finally:
            request.unlink(missing_ok=True)
            response.unlink(missing_ok=True)
