"""What a placement search costs: the per-target setup, and each candidate after it.

Backs fig_adm and the placement-optimisation rows of the cuNIBS cost table. Setup is the three
adjoint solves plus sampling the reciprocity field onto its grid, timed whole and as those two
parts. A candidate then costs an interpolation and a dipole sum, timed from 1 to a million
candidates evaluated together. Each call takes at most ``CHUNK`` candidates, as a sweep that size
would be fed.

The search against SimNIBS is whole against whole: from a mesh in memory to as many scored
candidates as ``adm_crosscode`` gave SimNIBS's ADM, against SimNIBS's time for that search from the
same result, which assembles and prepares its solver inside it. The two codes put different work on
either side of any internal boundary, so no phase is divided by another. A further target on a
subject that is already set up is timed beside it: SimNIBS keeps no solver between searches, so its
total is what such a target costs there too.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _measure, _objective, _run

COUNTS = (1, 100, 1_000, 10_000, 100_000, 1_000_000)
CHUNK = 1 << 17
# Distinct poses to tile the counts from: every raster position at many angles.
N_ANGLES = 360


def _per_candidate(recip, coil, frames, counts, repeats) -> list[dict]:
    import cupy as cp

    from cunibs.adm import evaluate_frames

    rows = []
    for n in counts:
        batch = cp.asarray(np.resize(frames, (min(n, CHUNK), 4, 4)))

        def sweep(n=n, batch=batch):
            for lo in range(0, n, CHUNK):
                evaluate_frames(recip, coil, batch[: min(CHUNK, n - lo)], _data.DIDT)

        ms = _measure.stats(_measure.timed(sweep, repeats))["median"]
        rows.append({"n": n, "ms": ms, "us_per_candidate": ms * 1e3 / n})
        print(f"{n:>9}: {rows[-1]['us_per_candidate']:.3f} us/candidate", flush=True)
    return rows


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, Subject, load_mesh
    from cunibs.adm import build_reciprocity, evaluate_frames, resolve_target

    # Internal: the setup split into its adjoint solves and its grid sampling.
    from cunibs.adm.reciprocity import sample_qfield, solve_adjoint

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    mesh = load_mesh(path)
    subject = Subject(mesh)
    coil = Coil.load(_data.COIL)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    target = _objective.cortical_target(subject, placement)
    roi = resolve_target(subject, target)
    centers, _, _ = _objective.candidate_frames(subject, target.position_mm)
    repeats = 3 if ctx.quick else 10

    def build():
        return build_reciprocity(subject, coil, roi, centers, distance_mm=_data.DISTANCE_MM)

    recip = build()
    adjoint = solve_adjoint(subject.context, roi)
    setup = {
        "total": _measure.stats(_measure.timed(build, repeats)),
        "adjoint_solves": _measure.stats(
            _measure.timed(lambda: solve_adjoint(subject.context, roi), repeats)
        ),
        "qfield_sample": _measure.stats(
            _measure.timed(lambda: sample_qfield(adjoint, recip.grid), repeats)
        ),
    }

    _, many, _ = _objective.candidate_frames(subject, target.position_mm, n_angles=N_ANGLES)
    counts = COUNTS[:4] if ctx.quick else COUNTS
    sweep = _per_candidate(recip, coil, many, counts, repeats)
    plateau = min(r["us_per_candidate"] for r in sweep)
    values = {
        "subject": sid,
        "target": "C3",
        "coil": _data.COIL,
        "n_centers": len(centers),
        "setup_ms": setup,
        "grid": {"shape": list(recip.grid.shape), "mib": recip.q.nbytes / 2**20},
        "per_candidate": sweep,
        "us_per_candidate_plateau": plateau,
        "within_5pct_of_plateau_from": next(
            r["n"]
            for i, r in enumerate(sweep)
            if all(s["us_per_candidate"] <= 1.05 * plateau for s in sweep[i:])
        ),
    }

    crosscode = ctx.result("adm_crosscode") or ctx.latest("adm_crosscode")
    simnibs = ((crosscode or {}).get("coils") or {}).get(_data.COIL, {}).get("simnibs")
    if simnibs:
        n = simnibs["n_candidates"]

        def search(on):
            resolved = resolve_target(on, target)
            at, candidates, _ = _objective.candidate_frames(on, target.position_mm)
            field = build_reciprocity(on, coil, resolved, at, distance_mm=_data.DISTANCE_MM)
            return evaluate_frames(field, coil, np.resize(candidates, (n, 4, 4)), _data.DIDT)

        def whole():
            with Subject(mesh) as fresh:
                return search(fresh)

        further = _measure.stats(_measure.timed(lambda: search(subject), repeats))
        ms = _measure.stats(_measure.timed(whole, repeats))
        total_ms = simnibs["total_s"] * 1e3
        values["search"] = {
            "n_candidates": n,
            "cunibs_ms": ms,
            "further_target_ms": further,
            "simnibs_s": simnibs["timings_s"] | {"total": simnibs["total_s"]},
            "simnibs_host": simnibs["host"],
            "speedup": total_ms / ms["median"],
            "speedup_further_target": total_ms / further["median"],
        }
    return values
