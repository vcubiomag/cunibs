"""Hash every stage of the AMG hierarchy build and print it, one fresh interpreter per run.

Not a test module (pytest collects ``test_*.py``): this is the child process
``test_determinism_cross_process.py`` spawns. It has to be a separate interpreter because the
thing under test is what a warm process hides -- cuSPARSE picks work splits from runtime state,
so a second call inside one process can agree bit for bit while a second *process* does not.

Takes the operator as a saved CSR rather than a mesh, so every child starts from byte-identical
input and the only thing that can move is the hierarchy build itself. Assembling per child would
fold in a separate, unrelated source: a head mesh carries near-degenerate tets, and the batched
solve behind ``gradient_operator`` takes a different pivoting path for those depending on how
much device memory is free.

Prints one ``CHILD_HASH <name> <sha256>`` line per array and one ``CHILD_SIZES`` line. The
parent compares them; anything that moves shows up under a name that says which stage it was.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cupy as cp
import numpy as np


def _emit(name: str, array: cp.ndarray) -> None:
    # Device-wide, not the current stream: some of these arrays are produced by library calls
    # that run on their own streams, and a plain ``asnumpy`` only orders against the current
    # one. Without this the hash can read a buffer mid-write and report the harness's
    # difference rather than the pipeline's.
    cp.cuda.runtime.deviceSynchronize()
    digest = hashlib.sha256(np.ascontiguousarray(cp.asnumpy(array)).tobytes()).hexdigest()
    print(f"CHILD_HASH {name} {digest}")


def save_operator(mesh_path: Path, out_dir: Path) -> None:
    """Assemble the reduced operator once and write it out, in a process of its own.

    The parent stays free of a CUDA context this way. That is not tidiness: a child building a
    full-mesh hierarchy while another process in the same session holds a context intermittently
    dies inside ``select_size4`` with an illegal access. Keeping the parent out of CUDA sidesteps
    a flake this test is not trying to measure.
    """
    from cunibs import Subject
    from cunibs.mesh import load_mesh

    subject = Subject(load_mesh(mesh_path))
    try:
        solver = subject.context.solver
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "indptr.npy", cp.asnumpy(solver.row_ptr))
        np.save(out_dir / "indices.npy", cp.asnumpy(solver.col_idx))
        np.save(out_dir / "data.npy", cp.asnumpy(solver.values.astype(cp.float32)))
    finally:
        subject.free()


def hash_hierarchy(operator_dir: Path) -> None:
    from cunibs.fem.solve import _l1_dinv, aggregation_levels

    def load(name: str, dtype) -> cp.ndarray:
        return cp.asarray(np.load(operator_dir / name), dtype=dtype)

    row_ptr = load("indptr.npy", cp.int32)
    col_idx = load("indices.npy", cp.int32)
    values_f32 = load("data.npy", cp.float32)

    levels, coarse = aggregation_levels(row_ptr, col_idx, values_f32)

    _emit("operator", values_f32)
    for i, level in enumerate(levels):
        _emit(f"l{i}.aggregates", level.aggregates)
        _emit(f"l{i}.a.data", level.a.data)
        _emit(f"l{i}.dinv", _l1_dinv(level.a))
        _emit(f"l{i}.p.data", level.p.data)
        _emit(f"l{i}.p.indices", level.p.indices)
    _emit("coarse.data", coarse.data)

    sizes = [int(row_ptr.shape[0]) - 1] + [level.n_coarse for level in levels]
    print("CHILD_SIZES " + ",".join(str(s) for s in sizes))


if __name__ == "__main__":
    if sys.argv[1] == "--save-operator":
        save_operator(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        hash_hierarchy(Path(sys.argv[1]))
