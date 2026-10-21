"""Whether the production stopping tolerance moves a conductivity ensemble.

Backs the first paragraph of the cuNIBS paper's conductivity-uncertainty results. The same 500
draws are solved at the production tolerance and again at 1e-9, and paired draw by draw: the
gray-matter peak, the ROI mean, the ensemble-mean field, and whether the peak moved. A two-sample
Kolmogorov-Smirnov test compares the two peak distributions, and the standard error of the
ensemble-mean peak is the Monte Carlo error the ensemble size leaves.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from benchmarks import _data, _objective, _run

N_DRAWS = 500
REFERENCE_TOLERANCE = 1e-9


def _ks_p(a: np.ndarray, b: np.ndarray) -> float:
    """Asymptotic two-sample Kolmogorov-Smirnov p-value."""
    x = np.concatenate([a, b])
    d = np.abs(
        np.searchsorted(np.sort(a), x, "right") / a.size
        - np.searchsorted(np.sort(b), x, "right") / b.size
    ).max()
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    # The alternating series does not converge near zero, where the p-value is 1.
    if lam < 0.3:
        return 1.0
    j = np.arange(1, 101)
    return float(np.clip(2 * np.sum((-1.0) ** (j - 1) * np.exp(-2 * j**2 * lam**2)), 0, 1))


def _paired(fast: np.ndarray, ref: np.ndarray) -> dict:
    dev = np.abs(fast - ref) / np.abs(ref).max()
    return {"median": float(np.median(dev)), "max": float(dev.max()), "ks_p": _ks_p(fast, ref)}


def run(ctx: _run.Context) -> dict:
    import cupy as cp

    from cunibs import Coil, Subject
    from cunibs.adm import resolve_target
    from cunibs.fem import ensure_recovery
    from cunibs.uq import (
        ConductivityUQConfig,
        build_conductivity_uq_precompute,
        run_conductivity_uq,
    )

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    roi = resolve_target(subject, _objective.cortical_target(subject, placement))
    operator = ensure_recovery(subject.context, "harmonic")
    tags = tuple(int(t) for t in np.unique(subject.mesh.tet_tags))
    pre = build_conductivity_uq_precompute(subject.context, tags)
    n = 20 if ctx.quick else N_DRAWS
    config = ConductivityUQConfig(n_samples=n)

    def ensemble(precompute):
        r = run_conductivity_uq(
            subject.context, precompute, coil, placement, config,
            didt=_data.DIDT, record_rois={"roi": roi}, recovery=operator,
        )  # fmt: skip
        return r.peak_samples, r.roi_samples["roi"], r.peak_location_samples, r.mean_magnE

    peak, roi_mean, loc, mean = ensemble(pre)
    tight = dataclasses.replace(pre, tolerance=REFERENCE_TOLERANCE)
    peak_t, roi_mean_t, loc_t, mean_t = ensemble(tight)

    mean_field = float(cp.abs(mean - mean_t).max() / cp.abs(mean_t).max())
    paired = {"peak": _paired(peak, peak_t), "roi_mean": _paired(roi_mean, roi_mean_t)}
    peak_cov = float(peak_t.std(ddof=1) / peak_t.mean())
    return {
        "subject": sid,
        "target": "C3",
        "n_draws": n,
        "tolerance": {"production": pre.tolerance, "reference": REFERENCE_TOLERANCE},
        **paired,
        "mean_field_max": mean_field,
        "max_deviation": max(paired["peak"]["max"], paired["roi_mean"]["max"], mean_field),
        "n_draws_peak_moved": int((np.linalg.norm(loc - loc_t, axis=1) > 0).sum()),
        "peak_cov": peak_cov,
        "peak_mean_standard_error": peak_cov / np.sqrt(n),
    }
