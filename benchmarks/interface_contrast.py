"""The conductivity contrast at a head mesh's interfaces, and at the reported peak.

Backs the recovery-paper results that the median contrast at mixed nodes is 6.015 (CSF over gray
matter); that 93.0% of gray-matter tetrahedra have a contrast above 2 and 55% one in (5, 20]; that
152 of the 160 ``harmonic`` peaks sit at a contrast in (5, 20] and none inside a single tissue;
and that stratifying the peak displacements by that contrast moves them by at most 0.15 mm.

A node's contrast is the largest over the smallest conductivity among its incident tetrahedra,
the factor by which the normal component of E jumps there; 1 means the node is inside one tissue.
A tetrahedron takes the largest contrast among its nodes. Fractions are per subject, summarised
by their median over the cohort.

The peaks come from ``recovery_in_practice``, which must have run on this machine. A pair is filed
under the contrast at its ``harmonic`` peak element. The stratified figure is, for each rule
against ``harmonic``, the median displacement among pairs in the most populated bin less the
median over all pairs, and ``max_stratified_shift_mm`` is the largest of those in magnitude.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _fields, _measure, _run

GRAY_MATTER = 2
EDGES = (1.0, 2.0, 5.0, 20.0, 100.0)
BINS = ("1", "(1, 2]", "(2, 5]", "(5, 20]", "(20, 100]", "> 100")


def _bin(contrast: np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGES, contrast, side="left")


def _contrast(subject) -> tuple[np.ndarray, np.ndarray]:
    """Per-node and per-tetrahedron contrast."""
    import cupy as cp
    import cupyx

    from cunibs.fem import TISSUE_CONDUCTIVITY

    ctx = subject.context
    lut = cp.zeros(max(TISSUE_CONDUCTIVITY) + 1)
    lut[list(TISSUE_CONDUCTIVITY)] = cp.asarray(list(TISSUE_CONDUCTIVITY.values()))
    sigma = cp.repeat(lut[ctx.tet_tags], 4)
    lo = cp.full(ctx.n_nodes, cp.inf)
    hi = cp.zeros(ctx.n_nodes)
    cupyx.scatter_min(lo, ctx.tet_nodes.ravel(), sigma)
    cupyx.scatter_max(hi, ctx.tet_nodes.ravel(), sigma)
    node = hi / lo
    return cp.asnumpy(node), cp.asnumpy(node[ctx.tet_nodes].max(axis=1))


def _spectrum(node: np.ndarray, tet: np.ndarray, gm: np.ndarray) -> dict:
    gm_bins = np.bincount(_bin(tet[gm]), minlength=len(BINS)) / gm.sum()
    return {
        "nodes_mixed": float(np.mean(node > 1)),
        "median_contrast_mixed_nodes": float(np.median(node[node > 1])),
        "gm_tets_above_2": float(np.mean(tet[gm] > 2)),
        "gm_tets_by_bin": dict(zip(BINS, gm_bins.tolist(), strict=True)),
    }


def run(ctx: _run.Context) -> dict:
    from cunibs import Subject

    payoff = ctx.result("recovery_in_practice")
    if payoff is None:
        raise RuntimeError("run recovery_in_practice on this machine first")
    subjects = ctx.data.subjects[:2] if ctx.quick else ctx.data.subjects
    per_subject, peaks = {}, []
    for sid in subjects:
        subject = Subject.from_mesh(ctx.data.mesh(sid))
        node, tet = _contrast(subject)
        per_subject[sid] = _spectrum(node, tet, subject.mesh.tet_tags == GRAY_MATTER)
        for case in (c for c in payoff["cases"] if c["subject"] == sid):
            peak = case["modes"]["harmonic"]["peak_element"]
            peaks.append(
                {"case": case, "contrast": float(tet[peak]), "bin": BINS[_bin(tet[peak])]}
            )
        subject.free()
        _measure.free_pools()

    cohort = {
        key: _fields.quantiles([row[key] for row in per_subject.values()])
        for key in ("nodes_mixed", "median_contrast_mixed_nodes", "gm_tets_above_2")
    }
    cohort["gm_tets_by_bin"] = {
        b: _fields.quantiles([row["gm_tets_by_bin"][b] for row in per_subject.values()])
        for b in BINS
    }
    counts = {b: sum(p["bin"] == b for p in peaks) for b in BINS}
    dominant = max(counts, key=counts.get)
    stratified = {}
    for mode in ("raw", "spr_global", "spr_tissue"):
        shift = np.array(
            [
                np.linalg.norm(
                    np.subtract(
                        p["case"]["modes"][mode]["peak_mm"],
                        p["case"]["modes"]["harmonic"]["peak_mm"],
                    )
                )
                for p in peaks
            ]
        )
        in_bin = np.array([p["bin"] == dominant for p in peaks])
        stratified[f"{mode}__vs__harmonic"] = {
            "median_all_mm": float(np.median(shift)),
            "median_in_bin_mm": float(np.median(shift[in_bin])),
            "difference_mm": float(np.median(shift[in_bin]) - np.median(shift)),
        }
    return {
        "subjects": per_subject,
        "cohort": cohort,
        "harmonic_peaks": {
            "n": len(peaks),
            "by_bin": counts,
            "dominant_bin": dominant,
            "inside_one_tissue": counts["1"],
            "pairs": [
                {"subject": p["case"]["subject"], "target": p["case"]["target"]}
                | {"contrast": p["contrast"]}
                for p in peaks
            ],
        },
        "stratified": stratified,
        "max_stratified_shift_mm": max(abs(s["difference_mm"]) for s in stratified.values()),
    }
