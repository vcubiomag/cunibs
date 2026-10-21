"""What one more placement costs in a sweep, and how the block width changes it.

Backs the sweep rows of the cuNIBS cost table: the cost of an additional placement in a
64-placement sweep, the cost of solving the same placements one at a time, and the speedup over
SimNIBS's cheapest placement.

The sweep is a rotation of the coil handle at C3, so the mesh-sized state is shared and only the
pose changes. It is timed as a user runs it: ``Subject.iter_simulate`` with |E| returned to the
host, against a loop of ``Subject.simulate`` calls over the same placements. The block grid then
times ``fem.solve_placements_block`` at each width and sweep size with the fields left on the
device, which shows how the width alone moves the cost. Width never changes a field.

SimNIBS's cost is its cheapest steady placement over every backend and thread count, from
simnibs_timing. It builds one solver and loops the placements too, so both sides amortise setup.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _measure, _run

SIZES = (1, 2, 4, 8, 16, 32, 64)
WIDTHS = (1, 2, 4, 8)


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, Subject
    from cunibs.fem import ensure_recovery, solve_placements_block

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    sizes = (1, 4, 16) if ctx.quick else SIZES
    repeats = 2 if ctx.quick else 5
    n = sizes[-1]
    angles = _data.HANDLE_ANGLE_DEG + 360.0 * np.arange(n) / n
    placements = _data.placements(subject, path, labels=("C3",), angles_deg=angles)
    transforms = _data.frames(subject, placements)
    sites = [(p.center_mm, p.handle_mm, p.distance_mm) for p in placements]
    operator = ensure_recovery(subject.context, "harmonic")

    def sweep(m: int) -> None:
        for _ in subject.iter_simulate(coil, placements[:m], _data.DIDT, magnitude=True):
            pass

    def one_at_a_time() -> None:
        for p in placements:
            subject.simulate(coil, p, _data.DIDT, magnitude=True)

    def blocks(m: int, k: int) -> None:
        for s in range(0, m, k):
            solve_placements_block(
                subject.context,
                coil.positions_m,
                coil.moments,
                sites[s : min(s + k, m)],
                _data.DIDT,
                recovery=operator,
                transforms=transforms[s : min(s + k, m)],
            )

    per_placement = {
        m: _measure.stats(np.array(_measure.timed(lambda m=m: sweep(m), repeats)) / m)
        for m in sizes
    }
    single = _measure.stats(np.array(_measure.timed(one_at_a_time, repeats, warmup=1)) / n)
    grid = {
        k: {
            m: float(np.median(_measure.timed(lambda m=m, k=k: blocks(m, k), repeats))) / m
            for m in sizes
        }
        for k in WIDTHS
    }
    best = min(WIDTHS, key=lambda k: grid[k][n])

    ms = per_placement[n]["median"]
    values = {
        "subject": sid,
        "n_tets": subject.mesh.n_tets,
        "coil": _data.COIL,
        "n_placements": n,
        "ms_per_placement": ms,
        "sweep_ms_per_placement": {str(m): s for m, s in per_placement.items()},
        "one_at_a_time_ms_per_placement": single["median"],
        "one_at_a_time": single,
        "batching_gain": single["median"] / ms,
        "block_ms_per_placement": {
            str(k): {str(m): v for m, v in g.items()} for k, g in grid.items()
        },
        "best_block_width": best,
        "block_width_gain": grid[1][n] / grid[best][n],
    }
    subject.free()

    timing = ctx.latest("simnibs_timing")
    if timing and timing.get("configs"):
        winner = timing["best"]["sweep"]
        config = next(
            c
            for c in timing["configs"]
            if c["solver"] == winner["solver"] and c["threads"] == winner["threads"]
        )
        values["simnibs"] = {
            "machine": timing["machine"],
            "solver": winner["solver"],
            "threads": winner["threads"],
            "steady_placement_s": config["steady_placement_s"],
        }
        values["speedup"] = config["steady_placement_s"] * 1e3 / ms

    # OnlineFEM's full-volume output is the like-for-like counterpart; its ROI-only output is
    # what it was designed for, and returns far less.
    onlinefem = ctx.latest("simnibs_onlinefem")
    if onlinefem:
        of = onlinefem["best"]["sweep"]
        values["onlinefem"] = {"machine": onlinefem["machine"], **of}
        values["speedup_vs_onlinefem"] = {
            "volume": of["volume"]["s"] * 1e3 / ms,
            "roi": of["roi"]["s"] * 1e3 / ms,
            "volume_device_resident": of["volume"]["s"] * 1e3 / grid[best][n],
        }
    return values
