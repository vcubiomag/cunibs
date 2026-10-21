"""What the conductivity prior does to peak, ROI mean, focality and hotspot across a cohort.

Backs fig_uq(b) and the cohort paragraph of the cuNIBS paper's conductivity results, and the
conductivity rows the recovery paper compares against: the ROI-mean coefficient of variation in
tab:context and the peak coefficient of variation in its text. Each of 10 subjects at four targets
gets a 500-draw ensemble with harmonic recovery.

Per pair: the coefficient of variation of each quantity over the draws, each tissue's first-order
share of the peak's log variance, and the largest distance of a draw's peak from the ensemble-mean
peak location. On most pairs the peak sits in one element in every draw and in a few it jumps
gyri, so the hotspot distance is reported only as quantiles across pairs and never in a ratio.
"""

from __future__ import annotations

from typing import get_args

import numpy as np

from benchmarks import _data, _fields, _objective, _run

N_SUBJECTS = 10
N_DRAWS = 500


def _spread(x) -> dict:
    return _fields.quantiles(x) | {"p95": float(np.percentile(x, 95))}


def _cov(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / x.mean())


def _by_tissue(pairs) -> dict[str, list[float]]:
    """Each tissue's shares over the pairs whose mesh holds it; not every head has every tissue."""
    out: dict[str, list[float]] = {}
    for p in pairs:
        for name, share in p["tissue_share"].items():
            out.setdefault(name, []).append(share)
    return out


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, Subject
    from cunibs.adm import resolve_target
    from cunibs.mesh import TissueLabel
    from cunibs.uq import DEFAULT_TISSUE_COV, ConductivityUQConfig

    names = dict(enumerate(get_args(TissueLabel.__value__), start=1))
    coil = Coil.load(_data.COIL)
    subjects = ctx.data.subjects
    if ctx.quick:
        subjects, labels, n = subjects[:1], _data.TARGETS[:1], 20
    else:
        subjects = [subjects[i * len(subjects) // N_SUBJECTS] for i in range(N_SUBJECTS)]
        labels, n = _data.TARGETS, N_DRAWS
    config = ConductivityUQConfig(n_samples=n)

    pairs = []
    for sid in subjects:
        path = ctx.data.mesh(sid)
        subject = Subject.from_mesh(path)
        for placement, label in zip(
            _data.placements(subject, path, labels), labels, strict=True
        ):
            roi = resolve_target(subject, _objective.cortical_target(subject, placement))
            r = subject.simulate_conductivity_uq(
                coil, placement, config, _data.DIDT, record_rois={"roi": roi}
            )
            loc = r.peak_location_samples
            pairs.append(
                {
                    "subject": sid,
                    "target": label,
                    "peak_cov": _cov(r.peak_samples),
                    "roi_mean_cov": _cov(r.roi_samples["roi"]),
                    "focality_cov": _cov(r.focality_samples),
                    "hotspot_max_mm": float(np.linalg.norm(loc - loc.mean(0), axis=1).max()),
                    "peak_unique": len(np.unique(loc, axis=0)) == 1,
                    "tissue_share": {
                        names[t]: s for t, s in r.tissue_sensitivity("peak").items()
                    },
                }
            )
            print(f"{sid} {label}: peak CoV {pairs[-1]['peak_cov']:.4f}", flush=True)
        subject.free()

    return {
        "coil": _data.COIL,
        "n_draws": n,
        "prior_cov": {names[t]: c for t, c in DEFAULT_TISSUE_COV.items()},
        "pairs": pairs,
        "cohort": {
            "n_pairs": len(pairs),
            "peak_cov": _spread([p["peak_cov"] for p in pairs]),
            "roi_mean_cov": _spread([p["roi_mean_cov"] for p in pairs]),
            "focality_cov": _spread([p["focality_cov"] for p in pairs]),
            "hotspot_max_mm": _spread([p["hotspot_max_mm"] for p in pairs]),
            "n_pairs_peak_not_unique": sum(not p["peak_unique"] for p in pairs),
            "tissue_share_median": {
                name: float(np.median(v)) for name, v in _by_tissue(pairs).items()
            },
        },
    }
