"""Subject-target pairs per hour on a whole CPU node running SimNIBS, against one GPU.

Backs the cohort claim. A study over many subjects runs SimNIBS as independent processes on a
cluster node, so for it the fair comparison is throughput, not one placement's latency. Each
SimNIBS process does one subject's work from scratch: read the mesh, set up, and solve the four
targets. The node runs 1 to 64 of them at once, its CPUs split evenly, under each solver, and the
best is the node's throughput. At about 12 GB per process, 64 is as many as a 1 TB node holds.
Run the SimNIBS side on an exclusive node, so that nothing else shares its CPUs or memory
bandwidth.

The GPU does the same work per subject in sequence (load, set up, recover and solve the four
targets) over several subjects, which is how a cohort runs on it.
"""

from __future__ import annotations

import time

import numpy as np

from benchmarks import _data, _measure, _run
from benchmarks._simnibs import mesh_arrays

PROCESSES = (1, 4, 8, 16, 32, 64)
SOLVERS = ("hypre", "pardiso")
GPU_SUBJECTS = 10


def _gpu_pairs_per_hour(ctx: _run.Context, coil) -> dict:
    from cunibs import Subject

    subjects = ctx.data.subjects[: 2 if ctx.quick else GPU_SUBJECTS]
    # Read once untimed, so the pace measures the work rather than whether the files are cached.
    for sid in subjects:
        ctx.data.mesh(sid).read_bytes()
    _measure.sync()
    t0 = time.perf_counter()
    for sid in subjects:
        path = ctx.data.mesh(sid)
        subject = Subject.from_mesh(path)
        list(subject.iter_simulate(coil, _data.placements(subject, path)))
        subject.free()
    _measure.sync()
    wall = time.perf_counter() - t0
    pairs = len(subjects) * len(_data.TARGETS)
    return {"subjects": len(subjects), "wall_s": wall, "pairs_per_hour": pairs * 3600 / wall}


def run(ctx: _run.Context) -> dict | None:
    from cunibs import Coil, Subject

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    mesh = mesh_arrays(subject.mesh)
    frames = _data.frames(subject, _data.placements(subject, path))
    subject.free()

    node, host = [], None
    for solver in SOLVERS:
        for n in PROCESSES[:2] if ctx.quick else PROCESSES:
            response = ctx.simnibs.call(
                "throughput",
                {"coil": _data.COIL, "didt": _data.DIDT, "solver": solver, "processes": n},
                frames=frames,
                **mesh,
            )
            if response is None:
                continue
            meta, host = response[0], response[0]["host"]
            per_process = [
                sum(r["setup_s"].values())
                + sum(map(sum, (p.values() for p in r["placements_s"])))
                for r in meta["runs"]
            ]
            node.append(
                {
                    "solver": solver,
                    "processes": n,
                    "threads_per_process": meta["threads_per_process"],
                    "wall_s": meta["wall_s"],
                    "process_s_median": float(np.median(per_process)),
                    "pairs_per_hour": n * len(frames) * 3600 / meta["wall_s"],
                }
            )
    if not node:
        return None

    best = max(node, key=lambda r: r["pairs_per_hour"])
    gpu = _gpu_pairs_per_hour(ctx, Coil.load(_data.COIL))
    return {
        "subject": sid,
        "simnibs_host": host,
        "node": node,
        "node_best": {k: best[k] for k in ("solver", "processes", "pairs_per_hour")},
        "gpu": gpu,
        "nodes_per_gpu": gpu["pairs_per_hour"] / best["pairs_per_hour"],
    }
