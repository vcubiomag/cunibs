"""The reciprocity surrogate against a forward solve at every candidate placement.

Backs the placement-optimisation accuracy paragraph of the cuNIBS paper and the data of
fig_adm_decision. On 8 subjects at four targets, every candidate of a 49-position, 12-angle grid
is scored by the surrogate and by a forward solve: how well the surrogate ranks them, whether it
picks the same one, and its error. The continuous-angle optimum from ``optimize_placement`` is
then solved directly, which gives the surrogate's error at the pose it returns and what the
continuous angle gains over the best gridded one.

The figure shows the pair with the largest median error: the handle-angle sweep at its selected
position, and each candidate's shortfall from the best next to the surrogate's error on it.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields, _objective, _run

N_SUBJECTS = 8
# The continuous-angle search needs an odd count for a symmetric trigonometric interpolant.
N_ANGLE_SAMPLES = 9


def _pair(subject, path, coil, label, quick: bool) -> tuple[dict, dict]:
    from cunibs.adm import build_reciprocity, optimize_placement, resolve_target

    (placement,) = _data.placements(subject, path, labels=(label,))
    target = _objective.cortical_target(subject, placement)
    roi = resolve_target(subject, target)
    search = {"radius_mm": 10.0, "n_angles": 4} if quick else {}
    centers, frames, angles = _objective.candidate_frames(subject, target.position_mm, **search)
    recip = build_reciprocity(subject, coil, roi, centers, distance_mm=_data.DISTANCE_MM)
    surrogate = _objective.surrogate_objective(recip, coil, frames)
    forward = _objective.forward_objective(subject, coil, frames, roi)

    best = optimize_placement(recip, coil, n_samples=N_ANGLE_SAMPLES, didt=_data.DIDT)
    at_best = _objective.forward_objective(
        subject, coil, _data.frames(subject, [best.best_placement]), roi
    )[0]
    pair = {
        "target": label,
        **_objective.score(surrogate, forward),
        "choice_error": abs(best.best_objective - at_best) / at_best,
        "continuous_gain": (at_best - forward.max()) / forward.max(),
    }
    arrays = {"angles": angles, "surrogate": surrogate, "forward": forward, "at_best": at_best}
    return pair, arrays


def _map_case(pair: dict, a: dict) -> dict:
    n_angles = len(a["angles"])
    forward, surrogate = a["forward"], a["surrogate"]
    top = forward.max()
    position = int(np.argmax(forward)) // n_angles
    sweep = forward.reshape(-1, n_angles)[position]
    err = np.abs(surrogate - forward) / top
    order = np.argsort(-forward)
    shortfall = (top - forward[order]) / top
    return {
        "subject": pair["subject"],
        "target": pair["target"],
        "n_candidates": int(forward.size),
        "angle_variation": float((sweep.max() - sweep.min()) / sweep.max()),
        "median_error": float(np.median(err)),
        "max_error": float(err.max()),
        "n_within_max_error": int((shortfall[1:] <= err.max()).sum()),
        "top_two_margin": float(shortfall[1]),
        "angle_sweep": {
            "deg": a["angles"].tolist(),
            "forward": sweep.tolist(),
            "surrogate": surrogate.reshape(-1, n_angles)[position].tolist(),
            "continuous_optimum_forward": float(a["at_best"]),
        },
        "by_rank": {"shortfall": shortfall.tolist(), "error": err[order].tolist()},
    }


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, Subject

    coil = Coil.load(_data.COIL)
    subjects = ctx.data.subjects
    if ctx.quick:
        subjects, labels = subjects[:1], _data.TARGETS[:1]
    else:
        subjects = [subjects[i * len(subjects) // N_SUBJECTS] for i in range(N_SUBJECTS)]
        labels = _data.TARGETS

    pairs, arrays = [], []
    for sid in subjects:
        path = ctx.data.mesh(sid)
        subject = Subject.from_mesh(path)
        for label in labels:
            pair, a = _pair(subject, path, coil, label, ctx.quick)
            pairs.append({"subject": sid, **pair})
            arrays.append(a)
            print(f"{sid} {label}: median error {pair['median_error']:.2e}", flush=True)
        subject.free()

    worst = int(np.argmax([p["median_error"] for p in pairs]))
    choice = [p["choice_error"] for p in pairs]
    return {
        "coil": _data.COIL,
        "n_pairs": len(pairs),
        "n_solves": sum(p["n_candidates"] + 1 for p in pairs),
        "spearman": _fields.quantiles([p["spearman"] for p in pairs]),
        "n_same_pick": sum(p["same_pick"] for p in pairs),
        "median_error": float(np.median([p["median_error"] for p in pairs])),
        "choice_error": {"median": float(np.median(choice)), "max": float(np.max(choice))},
        "continuous_gain": float(np.median([p["continuous_gain"] for p in pairs])),
        "pairs": pairs,
        "map_case": _map_case(pairs[worst], arrays[worst]),
    }
