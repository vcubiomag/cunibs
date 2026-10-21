"""How much the headline timings move within one process and between fresh processes.

Backs the timing protocol of the cuNIBS paper (the interquartile range within a run, and the change
between processes) and the range of the sweep cost across processes in the cost section.

Three warm timings are repeated in each of several new interpreter processes: one placement
through ``Subject.simulate``, the cost per placement of the 64-placement sweep that
sweep_throughput quotes, and the cost per draw of a conductivity ensemble. Each process loads the
mesh, builds its own solver state and warms up before timing, so the spread between their
medians is what a rerun of any benchmark would see.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from benchmarks import _data, _measure, _run

PROCESSES = 4
SWEEP = 64


def measure(mesh_path: str, quick: bool) -> dict[str, list[float]]:
    """Every sample of the three timings, in ms, from this process."""
    from cunibs import Coil, ConductivityUQConfig, Subject

    subject = Subject.from_mesh(mesh_path)
    coil = Coil.load(_data.COIL)
    sweep = SWEEP // 4 if quick else SWEEP
    angles = _data.HANDLE_ANGLE_DEG + 360.0 * np.arange(sweep) / sweep
    placements = _data.placements(subject, mesh_path, labels=("C3",), angles_deg=angles)
    draws = ConductivityUQConfig(n_samples=20 if quick else 500)
    n = 5 if quick else 20

    def run_sweep() -> None:
        for _ in subject.iter_simulate(coil, placements, _data.DIDT, magnitude=True):
            pass

    return {
        "placement_ms": _measure.timed(
            lambda: subject.simulate(coil, placements[0], _data.DIDT), 4 * n
        ),
        "sweep_ms_per_placement": [t / sweep for t in _measure.timed(run_sweep, n // 2)],
        "uq_ms_per_draw": [
            t / draws.n_samples
            for t in _measure.timed(
                lambda: subject.simulate_conductivity_uq(coil, placements[0], draws),
                n // 2,
                warmup=1,
            )
        ],
    }


def run(ctx: _run.Context) -> dict:
    path = str(ctx.data.mesh(ctx.data.reference_subject))
    runs = []
    for _ in range(2 if ctx.quick else PROCESSES):
        out = subprocess.run(
            [sys.executable, "-m", __name__, path, str(ctx.device), str(int(ctx.quick))],
            cwd=_run.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        runs.append(json.loads(out.splitlines()[-1]))

    values = {"subject": ctx.data.reference_subject, "processes": len(runs), "timings": {}}
    for key in runs[0]:
        within = [_measure.stats(r[key]) for r in runs]
        medians = [s["median"] for s in within]
        values["timings"][key] = {
            "per_process": within,
            "median_min": min(medians),
            "median_max": max(medians),
            "between_processes": max(medians) / min(medians) - 1,
        }
    iqr = [s["iqr_over_median"] for t in values["timings"].values() for s in t["per_process"]]
    values["within_process_iqr_over_median"] = {
        "max": max(iqr),
        "median": float(np.median(iqr)),
    }
    values["between_processes_max"] = max(
        t["between_processes"] for t in values["timings"].values()
    )
    return values


if __name__ == "__main__":
    import cupy as cp

    cp.cuda.Device(int(sys.argv[2])).use()
    print(json.dumps(measure(sys.argv[1], bool(int(sys.argv[3])))))
