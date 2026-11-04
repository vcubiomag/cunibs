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
    m = metrics.compute_metrics(MAGNE, VOLS, BARY, TAGS)
    assert m["peak_magnE"] == 3.0
    np.testing.assert_allclose(m["peak_location_mm"], [2, 0, 0])


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

    assert metrics.percentile_magnitude(magn, vols, mask, 99.9) == pytest.approx(1.0)

    assert metrics.focality(magn, vols, mask, 0.5) == pytest.approx(10000.0)
    assert metrics.stimulated_volume(magn, vols, mask, 0.5 * magn.max()) == pytest.approx(1.0)


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
    d = metrics.compute_metrics(MAGNE, VOLS, BARY, TAGS, percentiles=(50.0,))["distribution"]
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


@pytest.mark.parametrize("percentile", [101.0, -1.0, float("nan")])
def test_summary_rejects_a_percentile_outside_its_range(percentile):
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        metrics.compute_metrics(MAGNE, VOLS, BARY, TAGS, percentiles=(percentile,))


@pytest.mark.gpu
def test_device_summary_rejects_a_percentile_outside_its_range(cp):
    region = metrics.region_slice(cp.asarray(VOLS), cp.asarray(BARY), cp.asarray(TAGS))
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        region.summarize(cp.asarray(MAGNE), percentiles=(101.0,))


@pytest.mark.parametrize("length", [2, 5])
def test_summary_rejects_a_field_of_the_wrong_length(length):
    region = metrics.region_slice(VOLS, BARY, TAGS)
    with pytest.raises(ValueError, match="one value per element"):
        region.summarize(np.ones(length))


@pytest.mark.gpu
@pytest.mark.parametrize("length", [2, 5])
def test_device_summary_rejects_a_field_of_the_wrong_length(cp, length):
    """Too short a field reads past its array on the device, with no error and plausible values."""
    region = metrics.region_slice(cp.asarray(VOLS), cp.asarray(BARY), cp.asarray(TAGS))
    with pytest.raises(ValueError, match="one value per element"):
        region.summarize(cp.ones(length))


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
    tags = np.full(2, 2, np.int32)
    m = metrics.compute_metrics(magn, vols, bary, tags)
    np.testing.assert_allclose(m["center_of_gravity_mm"], [2.0, 0.0, 0.0])


def test_center_of_gravity_ignores_masked_elements():
    magn = np.array([1.0, 1.0, 1e6])
    vols = np.ones(3)
    bary = np.array([[0.0, 0, 0], [2.0, 0, 0], [1e3, 0, 0]])
    tags = np.array([2, 2, 5], np.int32)
    m = metrics.compute_metrics(magn, vols, bary, tags)
    np.testing.assert_allclose(m["center_of_gravity_mm"], [1.0, 0.0, 0.0])


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
    d = metrics.compute_metrics(
        np.array([7.0]),
        np.array([3.0]),
        np.zeros((1, 3)),
        np.array([2], np.int32),
        percentiles=(50.0, 99.0),
    )["distribution"]
    assert d["mean"] == pytest.approx(7.0)
    assert d["std"] == pytest.approx(0.0)
    assert d["p50"] == pytest.approx(7.0)
    assert d["p99"] == pytest.approx(7.0)


def test_distribution_percentiles_are_monotone():
    rng = np.random.default_rng(2)
    magn, vols = rng.random(500), rng.random(500) + 0.1
    tags = np.full(500, 2, np.int32)
    d = metrics.compute_metrics(magn, vols, np.zeros((500, 3)), tags)["distribution"]
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


def test_tail_percentiles_use_the_exact_probability_for_float32_fields():
    """p99.9 is the quantile at 0.999, not at float32(0.999), which sits 1.3e-8 above it."""
    rng = np.random.default_rng(7)
    magn = np.sort(rng.pareto(1.5, 200_000)).astype(np.float32)
    vols = rng.random(200_000).astype(np.float32) + 0.1
    mask = np.ones(magn.size, dtype=bool)
    exact = metrics.weighted_quantiles(magn, vols, np.array([99.9 / 100.0]))[0]
    float32_q = np.array([np.float32(99.9 / 100.0)], dtype=np.float64)
    rounded = metrics.weighted_quantiles(magn, vols, float32_q)[0]
    assert exact != rounded, "the tail must be steep enough to tell the two apart"
    tags = np.full(magn.size, 2, np.int32)
    d = metrics.compute_metrics(magn, vols, np.zeros((magn.size, 3)), tags)["distribution"]
    assert d["p99.9"] == exact
    assert metrics.percentile_magnitude(magn, vols, mask, 99.9) == exact


def _same_summary(a, b):
    for key, value in a.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, b[key], err_msg=key)
        else:
            assert value == b[key], key


@pytest.mark.gpu
def test_a_block_summary_is_each_field_summarised_alone(cp):
    """Batching is a throughput choice: no field's numbers may depend on its batch mates."""
    rng = np.random.default_rng(8)
    n = 60_000
    vols = cp.asarray(rng.random(n) + 0.1, dtype=cp.float32)
    bary = cp.asarray(rng.standard_normal((n, 3)))
    tags = cp.asarray(rng.choice(np.array([2, 3], np.int32), n))
    # Rounded so that ties, which decide where weight lands in the scan, are common.
    fields = [cp.round(cp.asarray(rng.pareto(2.0, n), dtype=cp.float32), 2) for _ in range(10)]
    region = metrics.region_slice(vols, bary, tags, "gray_matter")

    block = region.summarize_block(fields)
    for field, batched in zip(fields, block, strict=True):
        _same_summary(batched, region.summarize(field))
    for again in (region.summarize_block(fields), region.summarize_block(fields[::-1])[::-1]):
        for a, b in zip(block, again, strict=True):
            _same_summary(a, b)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "options",
    [
        {},
        # Without 99.9 among the percentiles, the focality anchor has to be probed on its own.
        {"percentiles": (50.0,), "focality_fracs": (0.3,)},
        {"percentiles": (100.0, 0.0), "focality_fracs": (0.2, 1.0)},
    ],
    ids=["default", "no_anchor", "extremes"],
)
def test_a_float32_device_summary_matches_the_host_path(cp, options):
    rng = np.random.default_rng(9)
    n = 20_000
    magn = (rng.pareto(2.0, n) + 0.01).astype(np.float32)
    vols = (rng.random(n) + 0.1).astype(np.float32)
    bary = rng.standard_normal((n, 3))
    tags = rng.choice(np.array([2, 3], np.int32), n)
    host = metrics.compute_metrics(magn, vols, bary, tags, **options)
    dev = metrics.compute_metrics(
        cp.asarray(magn), cp.asarray(vols), cp.asarray(bary), cp.asarray(tags), **options
    )
    assert dev["distribution"].keys() == host["distribution"].keys()
    assert dev["focality_m3"].keys() == host["focality_m3"].keys()
    assert dev["peak_magnE"] == host["peak_magnE"]
    np.testing.assert_array_equal(dev["peak_location_mm"], host["peak_location_mm"])
    np.testing.assert_allclose(
        dev["center_of_gravity_mm"], host["center_of_gravity_mm"], rtol=1e-6
    )
    for key, value in host["distribution"].items():
        assert dev["distribution"][key] == pytest.approx(value, rel=1e-6), key
    for key, value in host["focality_m3"].items():
        assert dev["focality_m3"][key] == pytest.approx(value, rel=1e-6), key


@pytest.mark.parametrize("percentile", [-1.0, 101.0, float("nan")])
def test_percentile_outside_its_range_is_rejected(percentile):
    values, vols = np.ones(4), np.ones(4)
    with pytest.raises(ValueError, match="percentile must be"):
        metrics.percentile_magnitude(values, vols, np.ones(4, dtype=bool), percentile)


@pytest.mark.gpu
@pytest.mark.parametrize("percentile", [-1.0, 101.0, float("nan")])
def test_a_summary_rejects_a_percentile_outside_its_range(cp, percentile):
    """Clamped to the extreme element, it would report the peak under the name it was asked for."""
    magn, vols = np.ones(8), np.ones(8)
    bary, tags = np.zeros((8, 3)), np.full(8, 2, np.int32)
    with pytest.raises(ValueError, match="percentile must be"):
        metrics.percentile_magnitude(magn, vols, np.ones(8, dtype=bool), percentile)
    with pytest.raises(ValueError, match="percentile must be"):
        metrics.compute_metrics(magn, vols, bary, tags, percentiles=(percentile,))
    device = [cp.asarray(a) for a in (magn, vols, bary, tags)]
    with pytest.raises(ValueError, match="percentile must be"):
        metrics.compute_metrics(*device, percentiles=(percentile,))


def test_a_percentile_of_no_elements_is_rejected():
    values, vols = np.ones(4), np.ones(4)
    with pytest.raises(ValueError, match="no elements"):
        metrics.percentile_magnitude(values, vols, np.zeros(4, dtype=bool), 50.0)


def test_device_summary_names_its_probe_limit(cp):
    magn, vols, bary = np.ones(8), np.ones(8), np.zeros((8, 3))
    tags = np.full(8, 2, dtype=np.int32)
    with pytest.raises(ValueError, match="at most 16 quantiles"):
        metrics.compute_metrics(
            cp.asarray(magn),
            cp.asarray(vols),
            cp.asarray(bary),
            cp.asarray(tags),
            percentiles=tuple(float(p) for p in range(1, 21)),
        )


@pytest.mark.parametrize("frac", [float("nan"), float("inf"), -0.5])
def test_focality_rejects_a_fraction_that_is_not_a_fraction(frac):
    values, vols = np.ones(4), np.ones(4)
    with pytest.raises(ValueError, match="focality fraction"):
        metrics.focality(values, vols, np.ones(4, dtype=bool), frac)
