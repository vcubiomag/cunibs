"""What the reciprocity grid's spacing costs in accuracy and in device memory.

Backs the grid-spacing sentences of the cuNIBS paper's placement-optimisation results. The
reciprocity field is tabulated on a Cartesian grid and interpolated at the coil's dipoles; 3 mm
is the shipped spacing. On one head and target, every candidate is scored at spacings from 1 to
6 mm against one set of forward solves, which do not depend on the spacing. Memory is every array
the field holds on the device, of which the tabulated grid is only a part.
"""

from __future__ import annotations

import dataclasses

from benchmarks import _data, _measure, _objective, _run

SPACINGS_MM = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SHIPPED_MM = 3.0


def _field_mib(recip) -> float:
    import cupy as cp

    arrays = (getattr(recip, f.name) for f in dataclasses.fields(recip))
    return sum(a.nbytes for a in arrays if isinstance(a, cp.ndarray)) / 2**20


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, Subject
    from cunibs.adm import build_reciprocity, resolve_target

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    target = _objective.cortical_target(subject, placement)
    roi = resolve_target(subject, target)
    search = {"radius_mm": 10.0, "n_angles": 4} if ctx.quick else {}
    centers, frames, _ = _objective.candidate_frames(subject, target.position_mm, **search)
    forward = _objective.forward_objective(subject, coil, frames, roi)

    rows = []
    for spacing in (SHIPPED_MM, 6.0) if ctx.quick else SPACINGS_MM:
        recip = build_reciprocity(
            subject, coil, roi, centers, distance_mm=_data.DISTANCE_MM, spacing_mm=spacing
        )
        surrogate = _objective.surrogate_objective(recip, coil, frames)
        rows.append(
            {
                "spacing_mm": spacing,
                "grid_shape": list(recip.grid.shape),
                "grid_mib": recip.q.nbytes / 2**20,
                "field_mib": _field_mib(recip),
                **_objective.score(surrogate, forward),
            }
        )
        print(f"{spacing} mm: median error {rows[-1]['median_error']:.2e}", flush=True)
        del recip
        _measure.free_pools()

    shipped = next(r for r in rows if r["spacing_mm"] == SHIPPED_MM)
    finest = min(rows, key=lambda r: r["spacing_mm"])
    return {
        "subject": sid,
        "target": "C3",
        "coil": _data.COIL,
        "n_candidates": len(frames),
        "spacings": rows,
        "same_pick_at_every_spacing": all(r["same_pick"] for r in rows),
        "finest_vs_shipped": {
            "finest_mm": finest["spacing_mm"],
            "memory_ratio": finest["field_mib"] / shipped["field_mib"],
            "error_ratio": finest["median_error"] / shipped["median_error"],
        },
    }
