"""cuNIBS and SimNIBS fields on the same meshes, coil frames and conductivities, over the cohort.

Backs the cross-code agreement in both manuscripts: the median and largest RDM between the two
codes' element fields over every subject and target, and the ROI means that error_in_context sets
beside mesh resolution and conductivity uncertainty.

SimNIBS receives cuNIBS's own mesh arrays and coil frames, so both fields come back on the same
elements in the same order. Every cuNIBS recovery mode is compared with two SimNIBS outputs: the
raw element field ``E``, and ``E_round_trip``, the element field averaged to the nodes and back.
The pair ``raw`` against ``E`` compares the two solves with no post-processing on either side.
The other pairs show how far a post-processing choice moves the field; they are not solver error.

Fields are compared in gray matter, weighted by element volume. The ROI is a 10 mm sphere of gray
matter centred 20 mm below the scalp projection of the target.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields, _run
from benchmarks._simnibs import mesh_arrays

OUTPUTS = ("E", "E_round_trip")


def _summaries(field: np.ndarray, w: np.ndarray, roi: np.ndarray) -> dict:
    magn = np.linalg.norm(field, axis=1)
    return {
        "roi_mean": _fields.roi_mean(magn, roi, w),
        "p99": _fields.weighted_percentile(magn, w, 99),
    }


def _subject(ctx: _run.Context, sid: str) -> list[dict] | None:
    import cupy as cp

    from cunibs import RECOVERY_MODES, Coil, Subject
    from cunibs.metrics import region_tag

    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    placements = _data.placements(subject, path)
    mesh = subject.mesh
    response = ctx.simnibs.call(
        "forward",
        {
            "coil": _data.COIL,
            "didt": _data.DIDT,
            "solver": "hypre",
            "threads": 0,
            "node_round_trip": True,
        },
        frames=_data.frames(subject, placements),
        **mesh_arrays(mesh),
    )
    if response is None:
        subject.free()
        return None
    gm = mesh.tet_tags == region_tag("gray_matter")
    # Only gray matter is compared, so only gray matter is held; ROIs are gray matter too.
    simnibs = {out: response[1][out][:, gm] for out in OUTPUTS}
    del response
    w = _fields.tet_volumes(mesh.nodes_mm, mesh.tet_nodes)[gm]
    in_gm = np.cumsum(gm) - 1
    rois = [in_gm[cp.asnumpy(r.elem_idx)] for r in _data.rois(subject, placements)]
    coil = Coil.load(_data.COIL)
    pairs = [
        {"subject": sid, "target": label, "rdm": {}, "mag": {}, "roi_mean": {}, "p99": {}}
        for label in _data.TARGETS
    ]
    for i, pair in enumerate(pairs):
        for out in OUTPUTS:
            s = _summaries(simnibs[out][i], w, rois[i])
            pair["roi_mean"][f"simnibs_{out}"] = s["roi_mean"]
            pair["p99"][f"simnibs_{out}"] = s["p99"]

    for mode in RECOVERY_MODES:
        results = subject.iter_simulate(
            coil, placements, _data.DIDT, vectors=True, recovery=mode
        )
        for i, (pair, result) in enumerate(zip(pairs, results, strict=True)):
            field = result.E[gm]
            del result
            s = _summaries(field, w, rois[i])
            pair["roi_mean"][f"cunibs_{mode}"] = s["roi_mean"]
            pair["p99"][f"cunibs_{mode}"] = s["p99"]
            for out in OUTPUTS:
                c = _fields.compare(field, simnibs[out][i], w)
                pair["rdm"][f"{mode}__vs__{out}"] = c["rdm"]
                pair["mag"][f"{mode}__vs__{out}"] = c["mag"]
    subject.free()
    return pairs


def _pairing(pairs: list[dict], mode: str, out: str) -> dict:
    key = f"{mode}__vs__{out}"

    def rel(quantity: str) -> list[float]:
        return [
            abs(p[quantity][f"cunibs_{mode}"] / p[quantity][f"simnibs_{out}"] - 1)
            for p in pairs
        ]

    return {
        "rdm": _fields.quantiles([p["rdm"][key] for p in pairs]),
        "mag": _fields.quantiles([p["mag"][key] for p in pairs]),
        "roi_mean_rel_diff": _fields.quantiles(rel("roi_mean")),
        "p99_rel_diff": _fields.quantiles(rel("p99")),
    }


def run(ctx: _run.Context) -> dict | None:
    from cunibs import RECOVERY_MODES

    subjects = ctx.data.subjects[:1] if ctx.quick else ctx.data.subjects
    pairs = []
    for sid in subjects:
        pairs += _subject(ctx, sid) or []
    if not pairs:
        return None
    return {
        "coil": _data.COIL,
        "simnibs_solver": "hypre",
        "roi": {"depth_mm": _data.ROI_DEPTH_MM, "radius_mm": _data.ROI_RADIUS_MM},
        "n_subjects": len({p["subject"] for p in pairs}),
        "n_pairs": len(pairs),
        "pairings": {
            f"{mode}__vs__{out}": _pairing(pairs, mode, out)
            for mode in RECOVERY_MODES
            for out in OUTPUTS
        },
        "pairs": pairs,
    }
