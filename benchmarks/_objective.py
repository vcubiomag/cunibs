"""The placement objective the ADM and conductivity benchmarks share, and how it is scored.

The ROI is the one :func:`benchmarks._data.rois` resolves. The objective is the magnitude of the
volume-weighted ROI mean of the element field, which is what the reciprocity surrogate
approximates. Its reference value comes from a forward solve.

Candidates are a 5 mm raster in the tangent plane at the target, clipped to a 20 mm disc and
projected onto the scalp, times a set of handle angles.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields

SEARCH_RADIUS_MM = 20.0
SEARCH_SPACING_MM = 5.0
N_ANGLES = 12


def cortical_target(subject, placement):
    """The :class:`~cunibs.adm.Target` under ``placement``'s centre."""
    from cunibs.adm import Target

    (center,) = _data.roi_centers(subject, [placement])
    return Target(center, region="gray_matter", radius_mm=_data.ROI_RADIUS_MM)


def candidate_frames(
    subject,
    center_mm,
    radius_mm: float = SEARCH_RADIUS_MM,
    spacing_mm: float = SEARCH_SPACING_MM,
    n_angles: int = N_ANGLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raster centres ``(C, 3)``, frames ``(C * A, 4, 4)`` with the angle inner, and the angles.

    The frames come from the optimiser's own projection and in-plane basis, so its
    continuous-angle search over the same centres measures angles from the same reference.
    """
    import cupy as cp

    # Internal: the projection and frame builder optimize_placement scores candidates with.
    from cunibs.adm.optimize import _frames, _project

    ctx = subject.context
    p, _, u, v = _project(ctx, cp.asarray(center_mm, dtype=cp.float64).reshape(1, 3), 0.0)
    steps = np.arange(-radius_mm, radius_mm + spacing_mm / 2, spacing_mm)
    du, dv = np.meshgrid(steps, steps, indexing="ij")
    keep = du**2 + dv**2 <= radius_mm**2
    centers = p + cp.asarray(du[keep])[:, None] * u + cp.asarray(dv[keep])[:, None] * v
    proj, normal, u, v = _project(ctx, centers, _data.DISTANCE_MM)
    angles = cp.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    frames = _frames(proj, normal, u, v, angles, _data.DISTANCE_MM)
    return cp.asnumpy(centers), cp.asnumpy(frames), np.degrees(cp.asnumpy(angles))


def forward_objective(subject, coil, frames: np.ndarray, roi) -> np.ndarray:
    """The objective at each frame from a full forward solve of the raw element field."""
    import cupy as cp

    from cunibs.fem import MAX_BLOCK, solve_placements_block

    out = np.empty(len(frames))
    for lo in range(0, len(frames), MAX_BLOCK):
        chunk = np.ascontiguousarray(frames[lo : lo + MAX_BLOCK])
        # The solver reads the transforms; the sites only set the block width.
        sites = [(f[:3, 3], f[:3, 3] + f[:3, 1], 0.0) for f in chunk]
        results = solve_placements_block(
            subject.context, coil.positions_m, coil.moments, sites, _data.DIDT, transforms=chunk
        )
        for i, r in enumerate(results):
            mean = roi.weights @ r["E"][roi.elem_idx].astype(cp.float64)
            out[lo + i] = float(cp.linalg.norm(mean))
    return out


def surrogate_objective(recip, coil, frames) -> np.ndarray:
    from cunibs.adm import evaluate_frames

    return np.linalg.norm(evaluate_frames(recip, coil, frames, _data.DIDT), axis=1)


def score(test: np.ndarray, ref: np.ndarray) -> dict:
    """How well ``test`` stands in for ``ref`` over a candidate set.

    Errors are relative to the best reference objective. ``loss`` is the reference objective
    given up by taking ``test``'s pick, relative to the best.
    """
    err = np.abs(test - ref) / ref.max()
    pick, best = int(np.argmax(test)), int(np.argmax(ref))
    order = np.argsort(-ref)
    return {
        "n_candidates": int(ref.size),
        "median_error": float(np.median(err)),
        "max_error": float(err.max()),
        "spearman": _fields.spearman(test, ref),
        "same_pick": pick == best,
        "loss": float((ref[best] - ref[pick]) / ref[best]),
        "top_two_margin": float((ref[order[0]] - ref[order[1]]) / ref[order[0]]),
    }
