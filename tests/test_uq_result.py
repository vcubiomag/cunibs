"""Host-side conductivity-UQ result logic: summaries and the tissue-sensitivity regression.

These paths are pure NumPy, so they are exercised without a device by building a
``ConductivityUQResult`` directly instead of running a Monte Carlo.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cunibs.simulation import Placement
from cunibs.uq.conductivity import ConductivityUQResult


def _result(sigma, y_peak=None, roi=None, focality=None) -> ConductivityUQResult:
    sigma = np.asarray(sigma, dtype=np.float64)
    n, p = sigma.shape
    rng = np.random.default_rng(0)
    m = 6
    mean = rng.random(m) + 0.5
    std = rng.random(m) * 0.1
    return ConductivityUQResult(
        mean_magnE=mean,
        std_magnE=std,
        cov_magnE=std / mean,
        n_samples=n,
        perturbed_tags=tuple(range(1, p + 1)),
        sigma_samples=sigma,
        vols=np.full(m, 0.5),
        tet_tags=np.array([2, 2, 2, 3, 3, 3], dtype=np.int32),
        barycenters_mm=rng.standard_normal((m, 3)),
        placement=Placement([1, 2, 3], [4, 5, 6]),
        coil_name="synthetic",
        didt=1e6,
        peak_samples=y_peak,
        focality_samples=focality,
        roi_samples=roi,
    )


def _lognormal_draws(n, p, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(0.0, 0.3, size=(n, p)))


def test_tissue_sensitivity_attributes_all_variance_to_the_driving_tissue():
    """y = σ₁³ exactly: the log-linear index must give tag 1 the whole variance share."""
    sigma = _lognormal_draws(4000, 3)
    peak = sigma[:, 0] ** 3
    s = _result(sigma, y_peak=peak).tissue_sensitivity("peak")
    assert s[1] == pytest.approx(1.0, abs=1e-6)
    assert abs(s[2]) < 1e-6 and abs(s[3]) < 1e-6


def test_tissue_sensitivity_splits_between_two_tissues():
    """Independent inputs: shares are βⱼ²Var(log σⱼ)/Var(log y).

    They sum to one only to the extent the sampled columns are uncorrelated — the residual
    sample covariance is why this is an approximate, first-order index rather than an exact
    variance decomposition.
    """
    sigma = _lognormal_draws(20000, 2, seed=3)
    peak = sigma[:, 0] ** 2 * sigma[:, 1]
    s = _result(sigma, y_peak=peak).tissue_sensitivity("peak")
    assert sum(s.values()) == pytest.approx(1.0, abs=5e-3)
    # β₁ = 2, β₂ = 1 over inputs with equal log-variance, so the split is 4:1.
    assert s[1] / s[2] == pytest.approx(4.0, rel=0.05)


def test_tissue_sensitivity_constant_output_is_all_zero():
    sigma = _lognormal_draws(64, 2)
    s = _result(sigma, y_peak=np.full(64, 3.0)).tissue_sensitivity("peak")
    assert s == {1: 0.0, 2: 0.0}


def test_tissue_sensitivity_reads_focality_and_roi_outputs():
    sigma = _lognormal_draws(2000, 2, seed=7)
    r = _result(
        sigma,
        y_peak=sigma[:, 0] ** 2,
        focality=sigma[:, 1] ** 2,
        roi={"m1": sigma[:, 1] ** 3},
    )
    assert r.tissue_sensitivity("peak")[1] == pytest.approx(1.0, abs=1e-6)
    assert r.tissue_sensitivity("focality")[2] == pytest.approx(1.0, abs=1e-6)
    assert r.tissue_sensitivity("m1")[2] == pytest.approx(1.0, abs=1e-6)


def test_tissue_sensitivity_without_record_rois_raises():
    r = _result(_lognormal_draws(16, 2))
    with pytest.raises(ValueError, match="No per-draw samples"):
        r.tissue_sensitivity("peak")


def test_tissue_sensitivity_unknown_output_raises():
    sigma = _lognormal_draws(16, 2)
    r = _result(sigma, y_peak=sigma[:, 0], roi={"m1": sigma[:, 1]})
    with pytest.raises(ValueError, match=r"available: \['peak', 'focality', 'm1'\]"):
        r.tissue_sensitivity("nope")


def test_per_draw_samples_and_sensitivity_live_on_the_result():
    sigma = _lognormal_draws(128, 2)
    r = _result(sigma, y_peak=sigma[:, 0] ** 2, roi={"m1": sigma[:, 1]})
    assert r.n_samples == 128
    assert r.perturbed_tags == (1, 2)
    assert r.tissue_sensitivity("peak")[1] == pytest.approx(1.0, abs=1e-6)


def test_non_default_region_is_computed_from_the_retained_moments():
    r = _result(_lognormal_draws(16, 2))
    assert r.peak_mean_magnE("csf") > 0
    assert r.peak_cov("csf") >= 0


def test_non_default_region_needs_the_moments():
    """Dropping the moments is what makes a region unanswerable, not the summary itself."""
    r = _result(_lognormal_draws(16, 2))
    bare = dataclasses.replace(
        r, summary=r.compute_summary(), mean_magnE=None, std_magnE=None, cov_magnE=None
    )
    assert bare.peak_mean_magnE() == bare.summary.mean_field["peak_magnE"]  # precomputed
    with pytest.raises(ValueError, match="moments=True"):
        bare.peak_mean_magnE("csf")


def test_result_peak_and_cov_are_region_masked():
    from cunibs import metrics

    r = _result(_lognormal_draws(16, 2))
    for region in ("gray_matter", "csf", "all"):
        mask = metrics.region_mask(r.tet_tags, region)
        assert r.peak_mean_magnE(region) == r.mean_magnE[mask].max()
        assert r.peak_cov(region) == r.cov_magnE[mask].max()
