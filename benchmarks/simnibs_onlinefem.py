"""SimNIBS's pre-factorised path for repeated placements (Cao et al. 2025), per stage and thread count.

Backs the OnlineFEM rows of the cuNIBS cost table and every speedup against SimNIBS for a sweep.
SimNIBS 4.6 ships it as ``simnibs.simulation.onlinefem.OnlineFEM``: the stiffness matrix is
factorised once with PARDISO and the right-hand-side force integrals are kept, so a placement
costs a source evaluation, a right-hand side, one back-substitution and the field at the points
asked for. Two outputs are priced from the same factorisation: every tetrahedron's own |E|, which
is what a cuNIBS sweep returns, and the ROI's tetrahedra alone, the use Cao et al. designed for.

As in Cao et al., the coil's dA/dt is sampled once on the coil file's own grid and interpolated
for each placement. The source is interpolated at the nodes and averaged onto elements, as the
standard SimNIBS solve and cuNIBS both do, and at the element centroids, OnlineFEM's own default.
Evaluating every dipole directly is timed too, as the variant with no sampling error. The field of
the first placement is checked against cuNIBS's unrecovered field on the same mesh, frame and
conductivities for both the sampled and the direct coil, so a speedup can only be quoted against a
solve of the same problem, and the cost of the sampling is stated beside it.

Conductivities enter the factorisation, so a conductivity draw costs a new setup plus one
placement, timed under the fastest configuration. The result is filed under the CPU that ran
SimNIBS.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _run
from benchmarks._simnibs import mesh_arrays

# 0 is every CPU the SimNIBS process may use, which only that host knows.
THREADS = (1, 4, 8, 16, 0)
# The first placement also compiles numba's kernels, so the steady cost is the median of the rest.
PLACEMENTS = 6
UQ_DRAWS = 3
GRAY_MATTER = 2


def tag(values: dict) -> str:
    return _run.cpu_tag(values.get("host", {}).get("cpu"))


def _medians(rows: list[dict]) -> dict:
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def _summary(meta: dict) -> dict:
    first, *rest = meta["placements_s"]
    steady = _medians(rest)

    def per_output(phases: dict, name: str) -> float:
        return phases["rhs_s"] + phases["solve_s"] + phases[f"fields_{name}_s"]

    setup = meta["setup_s"]
    common = setup["conductivity_s"] + setup["coil_s"] + setup["factorise_s"]
    return {
        "setup_s": setup,
        "steady_placement_phases_s": steady,
        "steady_placement_s": {n: per_output(steady, n) for n in ("volume", "roi")},
        "cold_s": {
            n: common + setup[f"cloud_{n}_s"] + per_output(first, n) for n in ("volume", "roi")
        },
    }


def _agreement(test: np.ndarray, ref: np.ndarray, vols: np.ndarray) -> dict:
    w = np.sqrt(vols)
    return {
        "relative_l2": float(np.linalg.norm((test - ref) * w) / np.linalg.norm(ref * w)),
        "max_abs_over_peak": float(np.abs(test - ref).max() / np.abs(ref).max()),
    }


def run(ctx: _run.Context) -> dict | None:
    import cupy as cp

    from cunibs import Coil, Subject

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
    (roi,) = _data.rois(subject, placements[:1])
    roi_tets = cp.asnumpy(roi.elem_idx).astype(np.int64)

    # (threads, source at element centroids, sampled coil). The thread sweep runs on Cao et al.'s
    # configuration; the other two vary one choice each at every CPU.
    threads = (0,) if ctx.quick else THREADS
    variants = [(n, False, True) for n in threads] + [(0, True, True), (0, False, False)]
    configs = []
    host = None
    fields = {}
    for n, use_elements, sampled in variants:
        reference = n == 0 and not use_elements
        response = ctx.simnibs.call(
            "onlinefem",
            {
                "coil": _data.COIL,
                "didt": _data.DIDT,
                "threads": n,
                "use_elements": use_elements,
                "sampled_coil": sampled,
                "field": reference,
            },
            frames=frames,
            roi_tets=roi_tets,
            **mesh,
        )
        if response is None:
            continue
        meta, arrays = response
        host = meta["host"]
        configs.append(
            {
                "threads": n,
                "use_elements": use_elements,
                "sampled_coil": sampled,
                **_summary(meta),
            }
        )
        if reference:
            fields["sampled_coil" if sampled else "direct_coil"] = arrays["magnE"]
    if not configs:
        return None

    agreement = {}
    if fields:
        result = subject.simulate(
            Coil.load(_data.COIL), placements[0], _data.DIDT, magnitude=True, recovery="raw"
        )
        vols = cp.asnumpy(subject.context.vols)
        ref = result.magnE.astype(np.float64)
        # The sampled coil's interpolation error sits next to the windings, in the scalp, so the
        # gray matter, where a study reads the field, is reported apart from the whole head.
        gm = subject.mesh.tet_tags == GRAY_MATTER
        for k, f in fields.items():
            f = f.reshape(-1).astype(np.float64)
            agreement[k] = {
                "volume": _agreement(f, ref, vols),
                "gray_matter": _agreement(f[gm], ref[gm], vols[gm]),
            }

    def best(key: str, output: str) -> dict:
        return min(configs, key=lambda c: c[key][output])

    sweep = {o: best("steady_placement_s", o) for o in ("volume", "roi")}
    cold = {o: best("cold_s", o) for o in ("volume", "roi")}

    tags = sorted(int(t) for t in np.unique(subject.mesh.tet_tags))
    rng = np.random.default_rng(0)
    sigmas = rng.lognormal(0.0, 0.2, (1 if ctx.quick else UQ_DRAWS, len(tags)))
    from cunibs.fem import TISSUE_CONDUCTIVITY

    sigmas *= np.array([TISSUE_CONDUCTIVITY[t] for t in tags])
    winner = sweep["roi"]
    uq = None
    response = ctx.simnibs.call(
        "onlinefem_uq",
        {
            "coil": _data.COIL,
            "didt": _data.DIDT,
            "threads": winner["threads"],
            "use_elements": winner["use_elements"],
            "sampled_coil": winner["sampled_coil"],
            "tags": tags,
        },
        frame=frames[0],
        sigmas=sigmas,
        roi_tets=roi_tets,
        **mesh,
    )
    if response is not None:
        phases = _medians(response[0]["draws"])
        uq = {
            "threads": winner["threads"],
            "use_elements": winner["use_elements"],
            "sampled_coil": winner["sampled_coil"],
            "draw_s": sum(phases.values()),
            "draw_phases_s": phases,
        }

    def pick(c: dict, key: str, output: str) -> dict:
        return {
            "threads": c["threads"],
            "use_elements": c["use_elements"],
            "sampled_coil": c["sampled_coil"],
            "s": c[key][output],
        }

    return {
        "subject": sid,
        "coil": _data.COIL,
        "n_tets": subject.mesh.n_tets,
        "n_roi_tets": int(roi_tets.size),
        "host": host,
        "configs": configs,
        "best": {
            "sweep": {o: pick(c, "steady_placement_s", o) for o, c in sweep.items()},
            "cold": {o: pick(c, "cold_s", o) for o, c in cold.items()},
        },
        "uq": uq,
        "agreement_with_cunibs_raw": agreement,
    }
