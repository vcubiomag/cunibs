"""SimNIBS's cost per stage under every solver backend and thread count, and per conductivity draw.

Backs the SimNIBS side of the cost table and every speedup against SimNIBS. The whole matrix is
published, including the configurations that lose, and each later speedup is taken against the
configuration that wins its case: the cheapest setup plus one placement for a cold start, and the
cheapest placement for a sweep. The result is filed under the CPU that ran SimNIBS.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _run
from benchmarks._simnibs import SOLVERS, mesh_arrays

# 0 is every CPU the SimNIBS process may use, which only that host knows.
THREADS = (1, 4, 8, 16, 0)
PLACEMENTS = 3
UQ_DRAWS = 9


def tag(values: dict) -> str:
    return _run.cpu_tag(values.get("host", {}).get("cpu"))


def _medians(rows: list[dict]) -> dict:
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def _summary(meta: dict) -> dict:
    setup = sum(meta["setup_s"].values())
    first, *rest = meta["placements_s"]
    # The first placement also pays for lazily built solver state, so the steady cost is the
    # median of the rest.
    steady = _medians(rest)
    return {
        "setup_s": meta["setup_s"],
        "steady_placement_phases_s": steady,
        "cold_s": setup + sum(first.values()),
        "steady_placement_s": sum(steady.values()),
    }


def run(ctx: _run.Context) -> dict | None:
    from cunibs import Subject

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    mesh = mesh_arrays(subject.mesh)
    placements = _data.placements(
        subject,
        path,
        labels=("C3",),
        angles_deg=_data.HANDLE_ANGLE_DEG + np.arange(PLACEMENTS) * 30,
    )
    frames = _data.frames(subject, placements)

    solvers = SOLVERS[:2] if ctx.quick else SOLVERS
    threads = (0,) if ctx.quick else THREADS
    configs = []
    for solver in solvers:
        for n in threads:
            response = ctx.simnibs.call(
                "forward",
                {
                    "coil": _data.COIL,
                    "didt": _data.DIDT,
                    "solver": solver,
                    "threads": n,
                    "fields": False,
                },
                frames=frames,
                **mesh,
            )
            if response is not None:
                host = response[0]["host"]
                configs.append({"solver": solver, "threads": n, **_summary(response[0])})
    if not configs:
        return None

    cold = min(configs, key=lambda c: c["cold_s"])
    sweep = min(configs, key=lambda c: c["steady_placement_s"])

    # A conductivity draw changes the stiffness matrix, so SimNIBS re-runs everything but the
    # mesh read for each one. Timed under both winners.
    tags = sorted(int(t) for t in np.unique(subject.mesh.tet_tags))
    rng = np.random.default_rng(0)
    sigmas = rng.lognormal(0.0, 0.2, (3 if ctx.quick else UQ_DRAWS, len(tags)))
    from cunibs.fem import TISSUE_CONDUCTIVITY

    sigmas *= np.array([TISSUE_CONDUCTIVITY[t] for t in tags])
    uq = {}
    for case, config in (("cold", cold), ("sweep", sweep)):
        response = ctx.simnibs.call(
            "uq",
            {
                "coil": _data.COIL,
                "didt": _data.DIDT,
                "solver": config["solver"],
                "threads": config["threads"],
                "tags": tags,
            },
            frame=frames[0],
            sigmas=sigmas,
            **mesh,
        )
        if response is not None:
            phases = _medians(
                [{k: v for k, v in d.items() if k.endswith("_s")} for d in response[0]["draws"]]
            )
            uq[case] = {
                "solver": config["solver"],
                "threads": config["threads"],
                "draw_s": sum(phases.values()),
                "draw_phases_s": phases,
            }

    return {
        "subject": sid,
        "coil": _data.COIL,
        "n_tets": subject.mesh.n_tets,
        "host": host,
        "configs": configs,
        "best": {
            "cold": {"solver": cold["solver"], "threads": cold["threads"]},
            "sweep": {"solver": sweep["solver"], "threads": sweep["threads"]},
        },
        "uq": uq,
    }
