from __future__ import annotations

import numpy as np
import pytest

from cunibs import metrics
from cunibs.mesh import VOLUME_KEY_TO_LABEL

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
        metrics.region_mask(TAGS, "bone")  # ty:ignore[invalid-argument-type]


def test_peak_excludes_other_regions():
    mask = metrics.region_mask(TAGS, "gray_matter")
    assert metrics.peak_magnitude(MAGNE, mask) == 3.0
    np.testing.assert_allclose(metrics.peak_location_mm(MAGNE, BARY, mask), [2, 0, 0])


def test_focality_volume_above_half_the_anchor():
    mask = metrics.region_mask(TAGS, "gray_matter")
    # Three elements: p99.9 saturates to the max, so both anchors agree here.
    assert metrics.focality(MAGNE, VOLS, mask, 0.5) == 2.0
    assert metrics.stimulated_volume(MAGNE, VOLS, mask, 2.5) == 1.0


def test_focality_anchors_to_a_percentile_not_the_max():
    """One hot sliver must not drag the focality threshold up with it."""
    magn = np.concatenate([np.ones(9999), [100.0]])
    vols = np.ones(10000)
    mask = np.ones(10000, dtype=bool)

    assert metrics.peak_magnitude(magn, mask) == 100.0
    assert metrics.percentile_magnitude(magn, vols, mask, 99.9) == pytest.approx(1.0)

    assert metrics.focality(magn, vols, mask, 0.5) == pytest.approx(10000.0)
    assert metrics.focality(magn, vols, mask, 0.5, anchor_percentile=100.0) == pytest.approx(
        1.0
    )


def test_compute_metrics_focality_matches_the_focality_function():
    m = metrics.compute_metrics(
        MAGNE, VOLS, BARY, TAGS, region="gray_matter", focality_fracs=(0.25, 0.5, 0.9)
    )
    assert set(m["focality_m3"]) == {"0.25", "0.5", "0.9"}
    mask = metrics.region_mask(TAGS, "gray_matter")
    for frac in (0.25, 0.5, 0.9):
        assert m["focality_m3"][f"{frac:g}"] == pytest.approx(
            metrics.focality(MAGNE, VOLS, mask, frac)
        )


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
    assert set(m["focality_m3"]) == {"0.5", "0.75"}
    assert "mean" in m["distribution"]
    assert m["center_of_gravity_mm"].shape == (3,)


def test_region_mask_covers_every_volume_label():
    tags = np.array(sorted(VOLUME_KEY_TO_LABEL), dtype=np.int32)
    for tag, label in VOLUME_KEY_TO_LABEL.items():
        mask = metrics.region_mask(tags, label)
        np.testing.assert_array_equal(tags[mask], [tag])


def test_weighted_quantiles_matches_hazen_under_uniform_weights():
    """Equal weights reduce the estimator to the Hazen plotting position, (i + 0.5) / n."""
    rng = np.random.default_rng(0)
    values = rng.random(1001)
    qs = np.array([0.01, 0.1, 0.5, 0.95, 0.99, 0.999])
    got = metrics.weighted_quantiles(values, np.ones_like(values), qs)
    np.testing.assert_allclose(got, np.quantile(values, qs, method="hazen"), atol=1e-9)


def test_weighted_quantiles_respects_weights():
    """A 9:1 weight split pulls the median almost all the way onto the heavy value."""
    values = np.array([1.0, 2.0])
    light = metrics.weighted_quantiles(values, np.array([1.0, 9.0]), np.array([0.5]))
    heavy = metrics.weighted_quantiles(values, np.array([9.0, 1.0]), np.array([0.5]))
    assert float(light[0]) > 1.8
    assert float(heavy[0]) < 1.2


def test_weighted_quantiles_is_scale_invariant_in_weights():
    rng = np.random.default_rng(1)
    values, weights = rng.random(50), rng.random(50) + 0.1
    qs = np.array([0.25, 0.5, 0.9])
    np.testing.assert_allclose(
        metrics.weighted_quantiles(values, weights, qs),
        metrics.weighted_quantiles(values, weights * 1e6, qs),
        rtol=1e-12,
    )


def test_center_of_gravity_analytic():
    """Equal volume·|E| weights put the centre of gravity exactly at the midpoint."""
    magn = np.array([2.0, 1.0])
    vols = np.array([1.0, 2.0])
    bary = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    mask = np.ones(2, dtype=bool)
    np.testing.assert_allclose(
        metrics.center_of_gravity_mm(magn, vols, bary, mask), [2.0, 0.0, 0.0]
    )


def test_center_of_gravity_ignores_masked_elements():
    magn = np.array([1.0, 1.0, 1e6])
    vols = np.ones(3)
    bary = np.array([[0.0, 0, 0], [2.0, 0, 0], [1e3, 0, 0]])
    mask = np.array([True, True, False])
    np.testing.assert_allclose(
        metrics.center_of_gravity_mm(magn, vols, bary, mask), [1.0, 0.0, 0.0]
    )


def test_stimulated_volume_threshold_is_inclusive():
    magn = np.array([1.0, 2.0, 3.0])
    vols = np.array([1.0, 10.0, 100.0])
    mask = np.ones(3, dtype=bool)
    assert metrics.stimulated_volume(magn, vols, mask, 2.0) == 110.0
    assert metrics.stimulated_volume(magn, vols, mask, 2.0 + 1e-12) == 100.0


def test_focality_full_field_equals_region_volume():
    mask = metrics.region_mask(TAGS, "gray_matter")
    assert metrics.focality(MAGNE, VOLS, mask, 0.0) == float(VOLS[mask].sum())


def test_focality_at_peak_is_the_peak_element_volume():
    mask = metrics.region_mask(TAGS, "gray_matter")
    assert metrics.focality(MAGNE, VOLS, mask, 1.0) == 1.0


def test_distribution_single_element():
    d = metrics.distribution(
        np.array([7.0]), np.array([3.0]), np.array([True]), percentiles=(50.0, 99.0)
    )
    assert d["mean"] == pytest.approx(7.0)
    assert d["std"] == pytest.approx(0.0)
    assert d["p50"] == pytest.approx(7.0)
    assert d["p99"] == pytest.approx(7.0)


def test_distribution_percentiles_are_monotone():
    rng = np.random.default_rng(2)
    magn, vols = rng.random(500), rng.random(500) + 0.1
    d = metrics.distribution(magn, vols, np.ones(500, dtype=bool))
    assert d["p50"] <= d["p95"] <= d["p99"] <= d["p99.9"]


@pytest.mark.gpu
def test_metrics_cupy_numpy_parity(cp):
    """``cp.get_array_module`` dispatch must give identical results on host and device."""
    rng = np.random.default_rng(3)
    magn = rng.random(400) + 0.1
    vols = rng.random(400) + 0.1
    bary = rng.standard_normal((400, 3))
    tags = rng.choice(np.array([2, 3, 5], np.int32), 400)

    host = metrics.compute_metrics(magn, vols, bary, tags, region="gray_matter")
    dev = metrics.compute_metrics(
        cp.asarray(magn),
        cp.asarray(vols),
        cp.asarray(bary),
        cp.asarray(tags),
        region="gray_matter",
    )
    assert host["region"] == dev["region"]
    for key in ("peak_magnE", "region_volume_m3"):
        assert host[key] == pytest.approx(dev[key], rel=1e-12)
    for key in ("peak_location_mm", "center_of_gravity_mm"):
        np.testing.assert_allclose(host[key], dev[key], rtol=1e-12)
    for key, value in host["distribution"].items():
        assert value == pytest.approx(dev["distribution"][key], rel=1e-12), key
    for key, value in host["focality_m3"].items():
        assert value == pytest.approx(dev["focality_m3"][key], rel=1e-12), key


@pytest.mark.gpu
def test_weighted_quantiles_are_repeatable(cp):
    """The same call must return the same number, every time.

    A decoupled-lookback scan associates its float adds in an order that follows block
    scheduling rather than the data, and ties decide which weight lands in which prefix-sum
    slot, so the array here carries many of both.
    """
    rng = cp.random.default_rng(0)
    values = cp.round(cp.asarray(rng.random(200_000), dtype=cp.float32), 3)
    weights = cp.asarray(rng.random(200_000), dtype=cp.float32)
    qs = cp.asarray([0.5, 0.95, 0.999], dtype=cp.float64)
    assert int(cp.unique(values).size) < values.size, "the tie path must be exercised"

    first = cp.asnumpy(metrics.weighted_quantiles(values, weights, qs))
    for _ in range(6):
        np.testing.assert_array_equal(
            cp.asnumpy(metrics.weighted_quantiles(values, weights, qs)), first
        )


@pytest.mark.gpu
def test_reused_region_slice_matches_a_per_call_summary(cp):
    """Reusing one RegionSlice across fields is an optimisation, so it may not move a value."""
    rng = np.random.default_rng(5)
    n = 50_000
    vols = cp.asarray(rng.random(n) + 0.1, dtype=cp.float32)
    bary = cp.asarray(rng.standard_normal((n, 3)))
    tags = cp.asarray(rng.choice(np.array([2, 3, 5], np.int32), n))
    fields = [cp.asarray(rng.random(n), dtype=cp.float32) for _ in range(3)]

    region = metrics.region_slice(vols, bary, tags, "gray_matter")
    for magn in fields:
        reused = region.summarize(magn)
        direct = metrics.compute_metrics(magn, vols, bary, tags, region="gray_matter")

        assert reused["region"] == direct["region"] == "gray_matter"
        assert reused["peak_magnE"] == direct["peak_magnE"]
        assert reused["region_volume_m3"] == direct["region_volume_m3"]
        np.testing.assert_array_equal(reused["peak_location_mm"], direct["peak_location_mm"])
        np.testing.assert_array_equal(
            reused["center_of_gravity_mm"], direct["center_of_gravity_mm"]
        )
        assert reused["distribution"] == direct["distribution"]
        assert reused["focality_m3"] == direct["focality_m3"]
