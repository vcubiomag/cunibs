"""Whether a placement's field depends on anything but its inputs, bit for bit, on one machine.

Backs the cuNIBS reproducibility result: a placement's field was bitwise identical when the solve
was repeated, when the block width changed from 1 to 8, when the placement moved within its block
and when it was solved in a fresh interpreter, compared there by the SHA-256 of the gray-matter
|E|. Every comparison here should find zero differing elements, so each is paired with a mutation
control that must find many: ``dI/dt`` scaled by ``1 + 1e-6``, which the field is linear in.
Without it a run of zeros could not be told apart from a comparison that sees nothing.

Block width and batch position are varied through ``Subject.iter_simulate``: a list of ``k``
placements solves as one block of width ``k``, with the probe placement first or last.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np

from benchmarks import _data, _run

GRAY_MATTER = 2


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "bitwise_equal": bool(np.array_equal(a, b)),
        "n_differing": int(np.count_nonzero(a != b)),
        "n_elements": int(a.size),
        "max_abs_over_peak": float(np.abs(a - b).max() / np.abs(b).max()),
    }


def _gm_digest(result) -> str:
    gm = np.ascontiguousarray(result.magnE[result.tet_tags == GRAY_MATTER])
    return hashlib.sha256(gm.tobytes()).hexdigest()


def _probe(mesh_path):
    """The subject, the coil, the C3 probe placement and seven others to batch it with."""
    from cunibs import Coil, Subject

    subject = Subject.from_mesh(mesh_path)
    others = _data.placements(
        subject, mesh_path, angles_deg=(_data.HANDLE_ANGLE_DEG, _data.HANDLE_ANGLE_DEG + 90)
    )
    return subject, Coil.load(_data.COIL), others[0], others[1:]


def _solve(subject, coil, placements, didt=_data.DIDT):
    return list(subject.iter_simulate(coil, placements, didt, magnitude=True))


def run(ctx: _run.Context) -> dict:
    mesh_path = ctx.data.mesh(ctx.data.reference_subject)
    subject, coil, probe, others = _probe(mesh_path)

    (base,) = _solve(subject, coil, [probe])
    (again,) = _solve(subject, coil, [probe])
    widths = (1, 2, 8) if ctx.quick else range(1, len(others) + 2)
    batches = {}
    for k in widths:
        rest = others[: k - 1]
        first = _solve(subject, coil, [probe, *rest])[0]
        last = _solve(subject, coil, [*rest, probe])[-1]
        batches[f"k{k}_first"] = _diff(first.magnE, base.magnE)
        batches[f"k{k}_last"] = _diff(last.magnE, base.magnE)
    (mutated,) = _solve(subject, coil, [probe], _data.DIDT * (1 + 1e-6))
    digest = _gm_digest(base)
    subject.free()

    child = subprocess.run(
        [sys.executable, "-m", "benchmarks.determinism", str(mesh_path), str(ctx.device)],
        cwd=_run.REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    fresh = child.stdout.strip().splitlines()[-1]

    repeat = _diff(again.magnE, base.magnE)
    mutation = _diff(mutated.magnE, base.magnE)
    return {
        "subject": ctx.data.reference_subject,
        "target": _data.TARGETS[0],
        "coil": _data.COIL,
        "repeat": repeat,
        "batch": batches,
        "fresh_process": {
            "gm_sha256_here": digest,
            "gm_sha256_fresh": fresh,
            "equal": digest == fresh,
        },
        "mutation": mutation,
        "all_bitwise_equal": repeat["bitwise_equal"]
        and all(b["bitwise_equal"] for b in batches.values())
        and digest == fresh,
        "mutation_detected": not mutation["bitwise_equal"],
    }


if __name__ == "__main__":
    import cupy as cp

    cp.cuda.Device(int(sys.argv[2])).use()
    subject, coil, probe, _ = _probe(sys.argv[1])
    print(_gm_digest(_solve(subject, coil, [probe])[0]))
