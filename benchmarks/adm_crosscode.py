"""cuNIBS's placement search against SimNIBS's ADM, both scored against forward solves.

Backs tab:adm-crosscode and the cross-code paragraph of the cuNIBS paper, and supplies the SimNIBS
search time that ``adm_throughput`` divides by. SimNIBS builds its ADM candidate grid and its ROI
on cuNIBS's own mesh arrays and returns every candidate as a 4x4 frame, so cuNIBS evaluates
exactly those poses. Both codes resolve the ROI independently, and the two element sets are
compared. Each candidate is also solved directly, which scores both surrogates against the same
reference rather than against each other.

ADMlib replaces the coil with a 17x17x2 grid of auxiliary dipoles. A third surrogate runs
cuNIBS's pipeline on that auxiliary coil: if it reproduces SimNIBS's error, the difference between
the codes is the coil approximation and not either finite-element solve. Both coils run at the
same head, target, ROI and candidate grid, so the coil is the only thing that changes.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _measure, _objective, _run
from benchmarks._simnibs import mesh_arrays

COILS = ("MagVenture_C-B70", "MagStim_D70")
SEARCH = {"search_radius_mm": 20.0, "resolution_pos_mm": 5.0, "resolution_angle_deg": 30.0}
QUICK_SEARCH = {
    "search_radius_mm": 10.0,
    "resolution_pos_mm": 10.0,
    "resolution_angle_deg": 90.0,
}


def _versus(test: np.ndarray, ref: np.ndarray) -> dict:
    diff = np.abs(test - ref) / ref.max()
    return {"median": float(np.median(diff)), "max": float(diff.max())}


def _auxiliary(subject, roi, centers, frames, positions_m, moments) -> np.ndarray:
    """cuNIBS's objective with ADMlib's auxiliary coil, one moment set per rotation.

    ADMlib places the auxiliary grid with the unrotated position frame and takes the rotation from
    the moment set, so each rotation is a coil of its own evaluated at every position.
    """
    from cunibs import Coil
    from cunibs.adm import build_reciprocity

    coils = [Coil(positions_m, np.ascontiguousarray(m)) for m in moments]
    recip = build_reciprocity(subject, coils[0], roi, centers, distance_mm=_data.DISTANCE_MM)
    by_rotation = [_objective.surrogate_objective(recip, c, frames) for c in coils]
    return np.stack(by_rotation, axis=1).reshape(-1)


def _compare(subject, coil, roi, response) -> dict:
    import cupy as cp

    from cunibs.adm import build_reciprocity

    meta, a = response
    frames, position_frames = a["frames"], a["position_frames"]
    centers = position_frames[:, :3, 3]
    simnibs = a["objective"]
    recip = build_reciprocity(subject, coil, roi, centers, distance_mm=_data.DISTANCE_MM)
    cunibs = _objective.surrogate_objective(recip, coil, frames)
    auxiliary = _auxiliary(
        subject, roi, centers, position_frames, a["aux_positions_m"], a["aux_moments"]
    )
    forward = _objective.forward_objective(subject, coil, frames, roi)
    _measure.free_pools()

    ours = set(cp.asnumpy(roi.elem_idx).tolist())
    theirs = set(a["roi_tets"].tolist())
    same_dipoles = a["dipoles_m"].shape == coil.positions_m.shape
    timings = meta["timings_s"]
    return {
        "n_dipoles": coil.n_dipoles,
        "pairing": {
            "dipole_position_diff_m": float(np.abs(a["dipoles_m"] - coil.positions_m).max())
            if same_dipoles
            else None,
            "dipole_moment_diff": float(np.abs(a["moments"] - coil.moments).max())
            if same_dipoles
            else None,
            "n_tets": len(a["barycenters"]),
            "barycenter_diff_mm": float(
                np.abs(a["barycenters"] - cp.asnumpy(subject.context.tet_barycenters_mm)).max()
            ),
            "roi_elements": {
                "cunibs": len(ours),
                "simnibs": len(theirs),
                "only_cunibs": len(ours - theirs),
                "only_simnibs": len(theirs - ours),
            },
        },
        "n_positions": meta["n_positions"],
        "n_rotations": meta["n_rotations"],
        "vs_forward": {
            "cunibs": _objective.score(cunibs, forward),
            "simnibs": _objective.score(simnibs, forward),
            "cunibs_auxiliary_coil": _objective.score(auxiliary, forward),
        },
        "vs_simnibs": {
            "cunibs": _versus(cunibs, simnibs),
            "cunibs_auxiliary_coil": _versus(auxiliary, simnibs),
        },
        "simnibs": {
            "n_candidates": len(frames),
            "timings_s": timings,
            "total_s": sum(timings.values()),
            "host": meta["host"],
        },
    }


def run(ctx: _run.Context) -> dict | None:
    from cunibs import Coil, Subject
    from cunibs.adm import resolve_target

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    target = _objective.cortical_target(subject, placement)
    roi = resolve_target(subject, target)
    mesh = mesh_arrays(subject.mesh)

    values = {"subject": sid, "target": "C3", "n_roi": int(roi.elem_idx.size), "coils": {}}
    for name in COILS:
        params = {
            "coil": name,
            "didt": _data.DIDT,
            "target_mm": target.position_mm.tolist(),
            "roi_radius_mm": _data.ROI_RADIUS_MM,
            "distance_mm": _data.DISTANCE_MM,
            "solver": "pardiso",
            "threads": 0,
            **(QUICK_SEARCH if ctx.quick else SEARCH),
        }
        response = ctx.simnibs.call("adm", params, **mesh)
        if response is not None:
            values["coils"][name] = _compare(subject, Coil.load(name), roi, response)
    return values if values["coils"] else None
