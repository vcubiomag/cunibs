"""How the cost of a placement and peak device memory grow with mesh size.

Backs the scaling figure and paragraph of the cuNIBS paper: the power-law exponent of the cost per
placement in the element count, the device memory per degree of freedom, the largest mesh solved
on this card, and the largest the fitted memory line predicts it could hold.

The ladder is one subject remeshed at eight resolutions, so only the size changes. On every rung
an eight-placement rotation of the handle at C3 is solved as one block of width eight with
``fem.solve_placements_block``, fields left on the device. The pose is derived on each rung from
the same scalp target; this is a cost study and does not compare fields across rungs.

Peak memory covers the whole rung, from building the solver context to the last solve, measured
above what the device held before. A rung that runs out of memory is recorded and left out of the
fits. A degree of freedom is a mesh node.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields, _measure, _run

PLACEMENTS = 8


def _rung(ctx: _run.Context, rung: _data.Rung) -> dict:
    from cunibs import Coil, Subject
    from cunibs.fem import ensure_recovery, solve_placements_block

    subject = Subject.from_mesh(rung.path)
    coil = Coil.load(_data.COIL)
    try:
        with _measure.PeakMemory() as peak:
            setup_ms, context = _measure.once(lambda: subject.context)
            operator = ensure_recovery(context, "harmonic")
            angles = _data.HANDLE_ANGLE_DEG + 360.0 * np.arange(PLACEMENTS) / PLACEMENTS
            # Ladder files carry no EEG cap; the rungs share the subject's coordinates.
            placements = _data.placements(
                subject, ctx.data.mesh(rung.subject), labels=("C3",), angles_deg=angles
            )
            transforms = _data.frames(subject, placements)
            sites = [(p.center_mm, p.handle_mm, p.distance_mm) for p in placements]

            def sweep():
                return solve_placements_block(
                    context,
                    coil.positions_m,
                    coil.moments,
                    sites,
                    _data.DIDT,
                    recovery=operator,
                    transforms=transforms,
                )

            samples = _measure.timed(sweep, 3 if ctx.quick else 10)
            iterations = max(r["iterations"] for r in sweep())
        return {
            "scale": rung.scale,
            "n_tets": subject.mesh.n_tets,
            "n_nodes": subject.mesh.n_nodes,
            "setup_ms": setup_ms,
            "ms_per_placement": float(np.median(samples)) / PLACEMENTS,
            "sweep_ms": _measure.stats(samples),
            "iterations": iterations,
            "amg_levels": context.solver.precond.n_levels(),
            "peak_mib": peak.peak_mib,
        }
    finally:
        context = operator = None
        subject.free()
        _measure.free_pools()


def run(ctx: _run.Context) -> dict:
    import cupy as cp

    from cunibs import load_mesh
    from cunibs.solver import clear_cuda_error

    rungs = ctx.data.ladder[:3] if ctx.quick else ctx.data.ladder
    # The first solve in a process also loads the library's CUDA modules into device memory,
    # which would land in the first rung's peak.
    _rung(ctx, rungs[0])
    solved, skipped = [], []
    for rung in rungs:
        try:
            solved.append(_rung(ctx, rung))
        except MemoryError as exc:
            skipped.append(
                {"scale": rung.scale, "n_tets": load_mesh(rung.path).n_tets, "error": str(exc)}
            )
            # A failed allocation leaves the thread's CUDA error set for the next launch to find.
            clear_cuda_error()
        # Outside the handler, whose traceback keeps the failed rung's arrays alive.
        _measure.free_pools()

    total_bytes = cp.cuda.runtime.memGetInfo()[1]
    values = {
        "subject": ctx.data.ladder[0].subject if ctx.data.ladder else None,
        "coil": _data.COIL,
        "placements": PLACEMENTS,
        "block_width": PLACEMENTS,
        "device_total_mib": total_bytes / 2**20,
        "rungs": solved,
        "skipped": skipped,
    }
    if len(solved) < 2:
        return values

    n_tets = [r["n_tets"] for r in solved]
    levels = max({r["amg_levels"] for r in solved}, key=[r["amg_levels"] for r in solved].count)
    iterations = [r["iterations"] for r in solved if r["amg_levels"] == levels]
    memory = _fields.fit_line(
        [r["n_nodes"] for r in solved], [r["peak_mib"] * 2**20 for r in solved]
    )
    tets_per_node = float(np.median([r["n_tets"] / r["n_nodes"] for r in solved]))
    max_nodes = (total_bytes - memory["intercept"]) / memory["slope"]
    values |= {
        "cost_fit": _fields.fit_power(n_tets, [r["ms_per_placement"] for r in solved]),
        "memory_fit": {"bytes_per_dof": memory["slope"], **memory},
        "iteration_ratio": {"amg_levels": levels, "ratio": max(iterations) / min(iterations)},
        "largest_solved_tets": max(n_tets),
        "largest_predicted": {"n_nodes": max_nodes, "n_tets": max_nodes * tets_per_node},
    }
    return values
