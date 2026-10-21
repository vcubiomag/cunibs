"""How the recovery rule moves the quantities a TMS study reports, over the cohort and 25 coils.

Backs the recovery paper's payoff table and the results around it. Every subject-target pair is
solved under each of the four rules, and each rule is compared with ``harmonic``: the median
change in peak |E|, the 99th and 99.9th percentiles, the ROI mean and focality (+11.1% peak for
raw, -2.31% for spr_global, ...); the median peak displacement (5.13, 1.84 and 0.69 mm) with 95%
bootstrap intervals ([2.88, 7.02] mm for raw); the share of pairs whose peak stays in the same
element (8% to 34%) and the largest displacement (38.6 to 73.5 mm). Every pair of rules is
reported, not only pairs with ``harmonic``.

All quantities are over gray matter, volume weighted. The center of gravity is the volume- and
|E|-weighted centroid of elements above half the 99.9th percentile, and focality the volume above
that threshold. The ROI is a 10 mm ball centred 20 mm inward from the scalp projection of the
target. Intervals resample subjects, not pairs, because a subject's four targets share a mesh.

``roi_by_mode`` is the median ratio of each rule's ROI mean to ``harmonic``'s, which the
error-in-context table places beside the cross-code, mesh and conductivity terms.

``coils`` repeats the displacement comparison with each of the 25 bundled coils (skipped in quick
mode). A coil's footprint is the larger in-plane extent of its dipole set (65 to 222 mm). A coil
dependence is only shown when the interquartile range of the per-coil medians exceeds the width
of one coil's own bootstrap interval, which ``iqr_over_ci_width`` measures.

``ladder`` is how each rule's peak and percentiles move across the resolution ladder, with the
coil frame of the finest rung that fits used on every rung.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from benchmarks import _data, _fields, _measure, _run

MODES = ("raw", "spr_global", "spr_tissue", "harmonic")
PAIRS = tuple(combinations(MODES, 2))
QUANTITIES = ("peak", "p99", "p99.9", "roi_mean", "focality")
ROI_DEPTH_MM = 20.0
ROI_RADIUS_MM = 10.0
GRAY_MATTER = 2


def _metrics(result, roi: str) -> dict:
    s = result.summary
    gm = np.flatnonzero(result.tet_tags == GRAY_MATTER)
    m = result.magnE[gm].astype(np.float64)
    above = m >= 0.5 * s["distribution"]["p99.9"]
    w = (m * result.vols[gm])[above]
    return {
        "peak": s["peak_magnE"],
        "p99": s["distribution"]["p99"],
        "p99.9": s["distribution"]["p99.9"],
        "roi_mean": result.roi_means[roi],
        "focality": s["focality_m3"]["0.5"],
        "peak_mm": s["peak_location_mm"].tolist(),
        "peak_element": int(gm[np.argmax(m)]),
        "cog_mm": (w @ result.barycenters_mm[gm][above] / w.sum()).tolist(),
    }


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def _subject(sid, path, coils):
    from cunibs import Subject
    from cunibs.head import project_to_scalp

    subject = Subject.from_mesh(path)
    placements = _data.placements(subject, path)
    origins, inward = project_to_scalp(subject, [p.center_mm for p in placements])
    rois = {
        t: subject.roi(o + ROI_DEPTH_MM * n, ROI_RADIUS_MM)
        for t, o, n in zip(_data.TARGETS, origins, inward, strict=True)
    }
    primary = coils[_data.COIL]
    cases = [{"subject": sid, "target": t, "modes": {}} for t in _data.TARGETS]
    for mode in MODES:
        results = subject.iter_simulate(
            primary, placements, _data.DIDT, magnitude=True, recovery=mode, record_rois=rois
        )
        for case, r in zip(cases, results, strict=True):
            case["modes"][mode] = _metrics(r, case["target"])

    sweep = {_data.COIL: [{m: c["modes"][m]["peak_mm"] for m in MODES} for c in cases]}
    for name, coil in coils.items():
        if name == _data.COIL:
            continue
        peaks = [{} for _ in placements]
        for mode in MODES:
            results = subject.iter_simulate(coil, placements, recovery=mode)
            for p, r in zip(peaks, results, strict=True):
                p[mode] = r.summary["peak_location_mm"]
        sweep[name] = peaks

    subject.free()
    return cases, sweep


def _median_ci(values, groups, n: int = 10_000) -> dict:
    """Median, with a 95% interval from resampling whole groups."""
    by = {}
    for g, v in zip(groups, values, strict=True):
        by.setdefault(g, []).append(v)
    blocks = np.array(list(by.values()))
    picks = np.random.default_rng(0).integers(0, len(blocks), (n, len(blocks)))
    lo, hi = np.percentile(np.median(blocks[picks].reshape(n, -1), axis=1), [2.5, 97.5])
    return {"median": float(np.median(values)), "ci95": [float(lo), float(hi)]}


def _displacements(cases, groups) -> dict:
    out = {}
    for a, b in PAIRS:
        peak = np.array(
            [_dist(c["modes"][a]["peak_mm"], c["modes"][b]["peak_mm"]) for c in cases]
        )
        cog = np.array([_dist(c["modes"][a]["cog_mm"], c["modes"][b]["cog_mm"]) for c in cases])
        out[f"{a}__vs__{b}"] = {
            "peak_shift_mm": _fields.quantiles(peak) | _median_ci(peak, groups),
            "frac_same_element": float(np.mean(peak == 0)),
            "cog_shift_mm": _fields.quantiles(cog) | _median_ci(cog, groups),
        }
    return out


def _coil_invariance(sweep, groups, footprints) -> dict:
    pairs = {}
    for a, b in PAIRS:
        stats = {
            coil: _median_ci([_dist(p[a], p[b]) for p in peaks], groups)
            for coil, peaks in sweep.items()
        }
        medians = np.array([s["median"] for s in stats.values()])
        width = float(np.median([s["ci95"][1] - s["ci95"][0] for s in stats.values()]))
        iqr = float(np.subtract(*np.percentile(medians, [75, 25])))
        pairs[f"{a}__vs__{b}"] = {
            "per_coil": stats,
            "min": float(medians.min()),
            "max": float(medians.max()),
            "range_mm": float(np.ptp(medians)),
            "median_over_coils": float(np.median(medians)),
            "iqr_mm": iqr,
            "ci_width_mm": width,
            "iqr_over_ci_width": iqr / width,
        }
    raw = [pairs["raw__vs__harmonic"]["per_coil"][c]["median"] for c in footprints]
    return {
        "footprint_mm": footprints,
        "footprint_range_mm": [min(footprints.values()), max(footprints.values())],
        "pairs": pairs,
        "raw_vs_harmonic_spearman_with_footprint": _fields.spearman(
            list(footprints.values()), raw
        ),
    }


def _ladder(ctx, coil) -> dict:
    from cunibs import Subject
    from cunibs.fem import solve_placements_block
    from cunibs.fem.recovery import build_recovery_operator
    from cunibs.metrics import region_slice

    rungs = ctx.data.ladder[:2] if ctx.quick else ctx.data.ladder
    cap_path = ctx.data.mesh(rungs[0].subject)
    frame, rows, skipped = None, {}, []
    for rung in reversed(rungs):
        subject = None
        try:
            subject = Subject.from_mesh(rung.path)
            if frame is None:
                (placement,) = _data.placements(subject, cap_path, labels=("C3",))
                frame = _data.frames(subject, [placement])[0]
            c = subject.context
            gm = region_slice(c.vols, c.tet_barycenters_mm, c.tet_tags, "gray_matter")
            site = [(placement.center_mm, placement.handle_mm, placement.distance_mm)]
            rows[rung.path.stem] = {}
            for mode in MODES:
                # Built uncached, so one operator at a time is resident on the finest rungs.
                op = None if mode == "raw" else build_recovery_operator(c, mode)
                (out,) = solve_placements_block(
                    c,
                    coil.positions_m,
                    coil.moments,
                    site,
                    _data.DIDT,
                    recovery=op,
                    transforms=frame[None],
                )
                s = gm.summarize(out["magnE"])
                rows[rung.path.stem][mode] = {"peak": s["peak_magnE"]} | {
                    q: s["distribution"][q] for q in ("p99", "p99.9")
                }
                del op, out
        except MemoryError as exc:
            skipped.append({"rung": rung.path.stem, "error": str(exc)})
        finally:
            if subject is not None:
                subject.free()
            _measure.free_pools()
    names = list(reversed(rows))
    spread = {
        mode: {
            q: {
                "relative_to_finest": [
                    rows[n][mode][q] / rows[names[-1]][mode][q] for n in names
                ],
                "range_over_finest": float(
                    np.ptp([rows[n][mode][q] for n in names]) / rows[names[-1]][mode][q]
                ),
            }
            for q in ("peak", "p99", "p99.9")
        }
        for mode in MODES
    }
    return {"rungs": names, "skipped": skipped, "values": rows, "spread": spread}


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, bundled_coils

    subjects = ctx.data.subjects[:2] if ctx.quick else ctx.data.subjects
    coils = {n: Coil.load(n) for n in ([_data.COIL] if ctx.quick else bundled_coils())}
    cases, sweep = [], {name: [] for name in coils}
    for sid in subjects:
        c, s = _subject(sid, ctx.data.mesh(sid), coils)
        cases += c
        for name, peaks in s.items():
            sweep[name] += peaks
        _measure.free_pools()
    groups = [c["subject"] for c in cases]

    sensitivity = {
        mode: {
            q: 100
            * float(np.median([c["modes"][mode][q] / c["modes"]["harmonic"][q] for c in cases]))
            - 100
            for q in QUANTITIES
        }
        for mode in MODES[:3]
    }
    displacement = _displacements(cases, groups)
    recovered = [k for k in displacement if "raw" not in k]

    def span(key, among=tuple(displacement)):
        values = [key(displacement[k]) for k in among]
        return [min(values), max(values)]

    roi_ratio = {m: 1 + sensitivity[m]["roi_mean"] / 100 for m in MODES[:3]}
    largest = max(roi_ratio, key=lambda m: abs(roi_ratio[m] - 1))
    values = {
        "coil": _data.COIL,
        "subjects": subjects,
        "n_pairs": len(cases),
        "cases": cases,
        "sensitivity_pct": sensitivity,
        "displacement": displacement,
        "headline": {
            "peak_shift_median_mm_recovered": span(
                lambda d: d["peak_shift_mm"]["median"], recovered
            ),
            "peak_shift_median_mm_all": span(lambda d: d["peak_shift_mm"]["median"]),
            "peak_change_pct_recovered": max(abs(sensitivity[m]["peak"]) for m in MODES[1:3]),
            "p999_roi_mean_max_abs_change_pct": max(
                abs(sensitivity[m][q]) for m in MODES[:3] for q in ("p99.9", "roi_mean")
            ),
            "focality_abs_change_pct": [
                f(abs(sensitivity[m]["focality"]) for m in MODES[:3]) for f in (min, max)
            ],
            "frac_same_element": span(lambda d: d["frac_same_element"]),
            "max_peak_shift_mm": span(lambda d: d["peak_shift_mm"]["max"]),
            "cog_shift_median_mm": span(lambda d: d["cog_shift_mm"]["median"]),
            "no_peak_shift_ci_contains_zero": all(
                d["peak_shift_mm"]["ci95"][0] > 0 for d in displacement.values()
            ),
        },
        "roi_by_mode": {
            "median_ratio": roi_ratio,
            "relative_change": {m: abs(r - 1) for m, r in roi_ratio.items()},
            "largest_mode": largest,
            "largest_relative_change": abs(roi_ratio[largest] - 1),
        },
        "ladder": _ladder(ctx, coils[_data.COIL]),
    }
    if len(coils) > 1:
        footprints = {
            n: float((np.ptp(c.positions_m[:, :2], axis=0) * 1e3).max())
            for n, c in coils.items()
        }
        values["coils"] = _coil_invariance(sweep, groups, footprints)
    return values
