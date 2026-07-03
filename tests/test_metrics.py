from __future__ import annotations

import numpy as np
import pytest

from cunibs import metrics

# Unit volumes make the expected metrics direct sums.
MAGNE = np.array([1.0, 2.0, 3.0, 100.0])
VOLS = np.array([1.0, 1.0, 1.0, 1.0])
TAGS = np.array([2, 2, 2, 5], dtype=np.int32)
BARY = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 9, 9]])


def test_region_mask():
    np.testing.assert_array_equal(
        metrics.region_mask(TAGS, "gray_matter"), [True, True, True, False]
    )
    assert metrics.region_mask(TAGS, "all").all()
    with pytest.raises(ValueError):
        metrics.region_mask(TAGS, "bone")


def test_peak_excludes_other_regions():
    mask = metrics.region_mask(TAGS, "gray_matter")
    assert metrics.peak_magnitude(MAGNE, mask) == 3.0
    np.testing.assert_allclose(metrics.peak_location_mm(MAGNE, BARY, mask), [2, 0, 0])


def test_focality_volume_above_half_peak():
    mask = metrics.region_mask(TAGS, "gray_matter")
    assert metrics.focality(MAGNE, VOLS, mask, 0.5) == 2.0
    assert metrics.stimulated_volume(MAGNE, VOLS, mask, 2.5) == 1.0


def test_distribution_volume_weighted_mean():
    mask = metrics.region_mask(TAGS, "gray_matter")
    d = metrics.distribution(MAGNE, VOLS, mask, percentiles=(50.0,))
    assert d["mean"] == pytest.approx(2.0)
    assert d["std"] == pytest.approx(np.sqrt(2 / 3))
    assert d["p50"] == pytest.approx(2.0)


def test_compute_metrics_shape():
    m = metrics.compute_metrics(MAGNE, VOLS, BARY, TAGS, region="gray_matter")
    assert m["peak_magnE"] == 3.0
    assert m["region_volume_m3"] == 3.0
    assert set(m["focality_m3"]) == {"0.5"}
    assert "mean" in m["distribution"]
    assert m["center_of_gravity_mm"].shape == (3,)
