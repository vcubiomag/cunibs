"""Speed per placement at matched accuracy, with each code in turn supplying the reference.

Backs the matched-accuracy speedups of the cuNIBS paper (abstract and the cost section). The two
codes stop at different residuals: cuNIBS at 1e-6 by default, SimNIBS's ``hypre`` at a fixed 1e-10,
and its three direct backends at none. So cuNIBS sweeps its tolerance and SimNIBS contributes one
point per backend, each a time per placement and a distance from a reference field.

Every SimNIBS backend converges to SimNIBS's own discrete solution, so its distance from the
tightest cuNIBS solution is the cross-code difference, and that is the accuracy cuNIBS is asked to
match. The cuNIBS points are scored twice: against the tightest cuNIBS solution, and against the
SimNIBS direct PARDISO solution. For each backend the speedup is its time over the time of the
cheapest cuNIBS tolerance at least as accurate. There is no interpolation between tolerances, so
the speedup can only be understated. The headline is the smallest speedup over the backends.

Accuracy is the volume-weighted relative L2 distance of the raw element field in gray matter. Both
codes are timed per placement with the solver prepared: cuNIBS from ``Subject.simulate`` with the
field returned to the host, SimNIBS from the second of two identical placements, since the first
also builds solver state lazily. Each backend runs at the thread count where simnibs_timing found
its placements cheapest.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields, _measure, _run
from benchmarks._simnibs import SOLVERS, mesh_arrays

TOLERANCES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
REFERENCE_TOLERANCE = 1e-12
SIMNIBS_REFERENCE = "pardiso"


def _threads(ctx: _run.Context, solver: str) -> int:
    timing = ctx.latest("simnibs_timing") or {}
    configs = [c for c in timing.get("configs", []) if c["solver"] == solver]
    return min(configs, key=lambda c: c["steady_placement_s"])["threads"] if configs else 0


def _matched(cunibs: list[dict], simnibs: dict[str, dict], score: str) -> dict:
    per_backend = {}
    for name, point in simnibs.items():
        target = point["rel_l2"]
        good = [c for c in cunibs if c[score] <= target]
        if not good:
            per_backend[name] = {"simnibs_rel_l2": target, "cunibs_tolerance": None}
            continue
        best = min(good, key=lambda c: c["ms"]["median"])
        per_backend[name] = {
            "simnibs_rel_l2": target,
            "simnibs_ms": point["ms"],
            "cunibs_tolerance": best["tolerance"],
            "cunibs_rel_l2": best[score],
            "cunibs_ms": best["ms"]["median"],
            "speedup": point["ms"] / best["ms"]["median"],
        }
    reached = [v for v in per_backend.values() if v["cunibs_tolerance"] is not None]
    headline = min(reached, key=lambda v: v["speedup"]) if reached else None
    return {"per_backend": per_backend, "speedup": headline["speedup"] if headline else None}


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, Subject
    from cunibs.metrics import region_tag

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    frame = _data.frames(subject, [placement])
    mesh = subject.mesh
    gm = mesh.tet_tags == region_tag("gray_matter")
    w = _fields.tet_volumes(mesh.nodes_mm, mesh.tet_nodes)[gm]

    def field():
        return subject.simulate(
            coil, placement, _data.DIDT, magnitude=True, vectors=True, recovery="raw"
        )

    # The stopping rule is not a public option; a tolerance sweep has to set it on the solver.
    solver = subject.context.solver
    solver.tolerance = REFERENCE_TOLERANCE
    reference = field()
    cunibs = []
    for tol in (1e-3, 1e-6, 1e-10) if ctx.quick else TOLERANCES:
        solver.tolerance = tol
        samples = _measure.timed(field, 5 if ctx.quick else 20)
        result = field()
        cunibs.append(
            {
                "tolerance": tol,
                "iterations": result.iterations,
                "ms": _measure.stats(samples),
                "E": result.E[gm],
            }
        )
    for point in cunibs:
        point["rel_l2"] = _fields.compare(point["E"], reference.E[gm], w)["rel_l2"]

    simnibs = {}
    for name in (SIMNIBS_REFERENCE,) if ctx.quick else SOLVERS:
        threads = _threads(ctx, name)
        response = ctx.simnibs.call(
            "forward",
            {"coil": _data.COIL, "didt": _data.DIDT, "solver": name, "threads": threads},
            frames=np.repeat(frame, 2, axis=0),
            **mesh_arrays(mesh),
        )
        if response is not None:
            meta, arrays = response
            simnibs[name] = {
                "threads": threads,
                "ms": sum(meta["placements_s"][1].values()) * 1e3,
                "solve_ms": meta["placements_s"][1]["solve_s"] * 1e3,
                "E": arrays["E"][0][gm],
            }
            host = meta["host"]
    for point in simnibs.values():
        point["rel_l2"] = _fields.compare(point["E"], reference.E[gm], w)["rel_l2"]
    if SIMNIBS_REFERENCE in simnibs:
        for point in cunibs:
            point["rel_l2_vs_simnibs"] = _fields.compare(
                point["E"], simnibs[SIMNIBS_REFERENCE]["E"], w
            )["rel_l2"]

    values = {
        "subject": sid,
        "target": "C3",
        "reference_tolerance": REFERENCE_TOLERANCE,
        "reference_iterations": reference.iterations,
        "cunibs": [{k: v for k, v in p.items() if k != "E"} for p in cunibs],
        "simnibs": {n: {k: v for k, v in p.items() if k != "E"} for n, p in simnibs.items()},
    }
    if simnibs:
        values["simnibs_host"] = host
        values["against_cunibs_reference"] = _matched(cunibs, simnibs, "rel_l2")
    if SIMNIBS_REFERENCE in simnibs:
        values["against_simnibs_reference"] = _matched(cunibs, simnibs, "rel_l2_vs_simnibs")
    subject.free()
    return values
