"""The cross-code difference beside other things that move the field: mesh, recovery, conductivity.

Backs the comparison of the cuNIBS paper (the cross-code ROI difference, and how many times smaller
it is than the mesh-resolution and conductivity effects) and the context table of the recovery
paper (code, mesh resolution, recovery rule and conductivity, and the ratios between them).

Every source is a relative change in the volume-weighted mean |E| over the cortical ROI. The
cross-code value is the median over field_agreement's pairs of the change between cuNIBS ``raw``
and the SimNIBS element field. The recovery value is the largest median change of a recovery rule
from ``harmonic``, from recovery_in_practice. The conductivity value is the median coefficient of
variation of the ROI mean over uq_cohort's pairs. A source whose benchmark has not run on this
machine is left out.

The mesh value is measured here, between the two finest ladder rungs that fit on the device, with
the ``harmonic`` field. The coil frame and the ROI centre are fixed once on the subject's own mesh
and reused on both rungs, because a pose re-derived per rung tilts with the remeshed scalp. The
robust peak (99th percentile of |E| in gray matter) is reported beside the ROI mean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks import _data, _fields, _measure, _run

if TYPE_CHECKING:
    import numpy as np


def _rung(rung: _data.Rung, placement, frame: np.ndarray, center: np.ndarray) -> dict:
    import cupy as cp

    from cunibs import Coil, Subject
    from cunibs.fem import ensure_recovery, solve_placements_block
    from cunibs.metrics import region_tag

    subject = Subject.from_mesh(rung.path)
    try:
        coil = Coil.load(_data.COIL)
        (result,) = solve_placements_block(
            subject.context,
            coil.positions_m,
            coil.moments,
            [(placement.center_mm, placement.handle_mm, placement.distance_mm)],
            _data.DIDT,
            recovery=ensure_recovery(subject.context, "harmonic"),
            transforms=frame[None],
        )
        magn = cp.asnumpy(result["magnE"])
        del result
        mesh = subject.mesh
        gm = mesh.tet_tags == region_tag("gray_matter")
        w = _fields.tet_volumes(mesh.nodes_mm, mesh.tet_nodes)
        roi = cp.asnumpy(subject.roi(center, radius_mm=_data.ROI_RADIUS_MM).elem_idx)
        return {
            "scale": rung.scale,
            "n_tets": mesh.n_tets,
            "roi_mean": _fields.roi_mean(magn, roi, w),
            "p99": _fields.weighted_percentile(magn[gm], w[gm], 99),
        }
    finally:
        subject.free()


def _mesh_resolution(ctx: _run.Context) -> dict | None:
    from cunibs import Subject
    from cunibs.head import project_to_scalp
    from cunibs.solver import clear_cuda_error

    path = ctx.data.mesh(ctx.data.ladder[0].subject)
    subject = Subject.from_mesh(path)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    (frame,) = _data.frames(subject, [placement])
    origin, inward = project_to_scalp(subject, placement.center_mm[None])
    center = origin[0] + _data.ROI_DEPTH_MM * inward[0]
    subject.free()

    rungs = ctx.data.ladder[:2] if ctx.quick else ctx.data.ladder
    solved, skipped = [], []
    for rung in reversed(rungs):
        try:
            solved.append(_rung(rung, placement, frame, center))
        except MemoryError:
            skipped.append(rung.scale)
            solved.clear()
            # A failed allocation leaves the thread's CUDA error set for the next launch to find.
            clear_cuda_error()
        _measure.free_pools()
        if len(solved) == 2:
            fine, coarse = solved
            return {
                "coarse": coarse,
                "fine": fine,
                "skipped_scales": skipped,
                "roi_mean": abs(coarse["roi_mean"] / fine["roi_mean"] - 1),
                "p99": abs(coarse["p99"] / fine["p99"] - 1),
            }
    return None


def run(ctx: _run.Context) -> dict:
    values: dict = {"roi": {"depth_mm": _data.ROI_DEPTH_MM, "radius_mm": _data.ROI_RADIUS_MM}}
    sources: dict[str, float] = {}

    agreement = ctx.result("field_agreement")
    if agreement:
        pairing = agreement["pairings"]["raw__vs__E"]
        values["cross_code"] = {
            "n_pairs": agreement["n_pairs"],
            "roi_mean": pairing["roi_mean_rel_diff"]["median"],
            "p99": pairing["p99_rel_diff"]["median"],
        }
        sources["cross_code"] = values["cross_code"]["roi_mean"]

    mesh = _mesh_resolution(ctx)
    if mesh:
        values["mesh_resolution"] = mesh
        sources["mesh_resolution"] = mesh["roi_mean"]

    practice = ctx.result("recovery_in_practice")
    if practice:
        by_mode = practice["roi_by_mode"]
        values["recovery_rule"] = by_mode["relative_change"]
        sources["recovery_rule"] = by_mode["largest_relative_change"]

    cohort = ctx.result("uq_cohort")
    if cohort:
        values["conductivity"] = cohort["cohort"]["roi_mean_cov"]
        sources["conductivity"] = cohort["cohort"]["roi_mean_cov"]["median"]

    values["sources"] = sources
    values["ratios"] = {
        f"{a}_over_{b}": sources[a] / sources[b]
        for a, b in (
            ("mesh_resolution", "cross_code"),
            ("recovery_rule", "cross_code"),
            ("conductivity", "cross_code"),
            ("conductivity", "recovery_rule"),
        )
        if a in sources and b in sources
    }
    return values
