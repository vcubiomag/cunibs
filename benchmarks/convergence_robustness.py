"""Whether the solver converges the same way on every head and at every resolution.

Backs the cuNIBS results that every solve over the cohort, the four targets and the resolution
ladder converged, with a largest final relative residual of 9.9993e-7; that the iteration count
varied by a factor of 1.22 over the ladder rungs with three-level hierarchies and by 1.8 over the
cohort; and that the 17.1M-element rung coarsened to a fourth level and took 34 iterations.

The cohort varies anatomy at nearly constant element count, and the ladder varies element count at
fixed anatomy. Counts are per placement: in a block solve each column stops at its own residual.
Coarsening runs to a fixed row count, so a large enough mesh gains a level, and the ladder is
grouped by hierarchy depth: within a depth a flat count is what mesh independence means. The
ladder solves C3 only, its pose derived on each rung, which moves the field slightly but not the
iteration count. A rung that does not fit on the device is listed under ``skipped``.

A ``SolverConvergenceError`` is raised only after a preconditioner rebuild and a restart have
both missed tolerance. A block that raises is re-solved one placement at a time, so a failure is
charged to its own placement.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _measure, _run


def _solve(path, cap_path, labels) -> dict:
    from cunibs import Coil, Subject
    from cunibs.fem import SolverConvergenceError

    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    placements = _data.placements(subject, cap_path, labels)
    failures = 0
    try:
        results = list(subject.iter_simulate(coil, placements, _data.DIDT))
    except SolverConvergenceError:
        results = []
        for placement in placements:
            try:
                results.append(subject.simulate(coil, placement, _data.DIDT))
            except SolverConvergenceError:
                failures += 1
    row = {
        "n_tets": subject.mesh.n_tets,
        # The hierarchy depth is not exposed publicly; the preconditioner reports it.
        "n_levels": int(subject.context.solver.precond.n_levels()),
        "targets": list(labels),
        "iterations": [r.iterations for r in results],
        "max_relative_residual": max((r.relative_residual for r in results), default=None),
        "tolerance": results[0].provenance["solver_tolerance"] if results else None,
        "n_failures": failures,
    }
    subject.free()
    return row


def _spread(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "min": float(x.min()),
        "median": float(np.median(x)),
        "max": float(x.max()),
        "max_over_min": float(x.max() / x.min()),
    }


def run(ctx: _run.Context) -> dict:
    subjects = ctx.data.subjects[:2] if ctx.quick else ctx.data.subjects
    cohort = {
        sid: _solve(ctx.data.mesh(sid), ctx.data.mesh(sid), _data.TARGETS) for sid in subjects
    }

    rungs = ctx.data.ladder[:2] if ctx.quick else ctx.data.ladder
    ladder, skipped = {}, []
    for rung in rungs:
        try:
            row = _solve(rung.path, ctx.data.mesh(rung.subject), _data.TARGETS[:1])
        except MemoryError as exc:
            skipped.append({"rung": rung.path.stem, "error": str(exc)})
            _measure.free_pools()
            continue
        ladder[rung.path.stem] = row | {"scale": rung.scale, "median": row["iterations"][0]}

    by_depth = {}
    for name, row in ladder.items():
        by_depth.setdefault(str(row["n_levels"]), {})[name] = row["median"]
    depths = {
        depth: {"rungs": members, **_spread(list(members.values()))}
        for depth, members in sorted(by_depth.items())
    }

    rows = [*cohort.values(), *ladder.values()]
    return {
        "coil": _data.COIL,
        "cohort": {
            "subjects": cohort,
            "n_solves": sum(len(r["iterations"]) for r in cohort.values()),
            "iterations": _spread([i for r in cohort.values() for i in r["iterations"]]),
        },
        "ladder": {
            "rungs": ladder,
            "skipped": skipped,
            "iterations": _spread([r["median"] for r in ladder.values()]) if ladder else None,
            "by_depth": depths,
            "finest": next(({"rung": k, **v} for k, v in reversed(ladder.items())), None),
        },
        "n_solves": sum(len(r["iterations"]) for r in rows),
        "n_failures": sum(r["n_failures"] for r in rows),
        "worst_relative_residual": max(
            r["max_relative_residual"] for r in rows if r["max_relative_residual"] is not None
        ),
    }
