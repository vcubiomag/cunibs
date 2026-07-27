"""Public API contracts: return shapes, dispatch rules, and every documented error path."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import cunibs
from cunibs import metrics
from cunibs.simulation import FieldResult, Placement


def _field_result(**overrides) -> FieldResult:
    rng = np.random.default_rng(0)
    m, n = 12, 8
    magnE = rng.random(m) + 0.1
    vols = rng.random(m) + 0.1
    tet_tags = np.array([2] * 8 + [3] * 4, dtype=np.int32)
    barycenters_mm = rng.standard_normal((m, 3))
    kwargs = dict(
        summary=metrics.compute_metrics(
            magnE, vols, barycenters_mm, tet_tags, region="gray_matter"
        ),
        magnE=magnE,
        E=rng.standard_normal((m, 3)),
        v=rng.standard_normal(n),
        transform=np.eye(4),
        placement=Placement([1, 2, 3], [4, 5, 6]),
        coil_name="synthetic",
        didt=1e6,
        vols=vols,
        tet_tags=tet_tags,
        barycenters_mm=barycenters_mm,
    )
    return FieldResult(**{**kwargs, **overrides})


def test_all_exports_importable():
    assert cunibs.__all__
    for name in cunibs.__all__:
        assert hasattr(cunibs, name), name
    assert isinstance(cunibs.__version__, str) and cunibs.__version__


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, 3, 4], [[1, 2], [3, 4]], []])
def test_placement_rejects_non_3vector(bad):
    with pytest.raises(ValueError):
        Placement(bad, [0, 0, 1])
    with pytest.raises(ValueError):
        Placement([0, 0, 1], bad)


def test_placement_accepts_any_shape_holding_three_values():
    p = Placement(np.array([[1.0], [2.0], [3.0]]), (4, 5, 6))
    np.testing.assert_array_equal(p.center_mm, [1, 2, 3])
    np.testing.assert_array_equal(p.handle_mm, [4, 5, 6])


def test_placement_is_frozen_and_float64():
    p = Placement((1, 2, 3), [4.0, 5.0, 6.0])
    assert p.center_mm.dtype == np.float64
    assert p.center_mm.flags.c_contiguous
    assert p.distance_mm == 4.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.distance_mm = 9.0  # ty:ignore[invalid-assignment]


def test_precomputed_summary_answers_the_default_region_without_arrays():
    """The device-side summary is what makes a bare result useful; no magnE needed."""
    r = _field_result(magnE=None, E=None, v=None, vols=None, tet_tags=None, barycenters_mm=None)
    assert r.peak_magnE() == r.summary["peak_magnE"]
    assert r.focality(0.5) == r.summary["focality_m3"]["0.5"]
    np.testing.assert_array_equal(r.peak_location_mm(), r.summary["peak_location_mm"])
    np.testing.assert_array_equal(r.center_of_gravity_mm(), r.summary["center_of_gravity_mm"])


@pytest.mark.parametrize("region", ["csf", "all"])  # the tissues _field_result() contains
def test_non_default_region_needs_the_magnitude(region):
    bare = _field_result(magnE=None, vols=None, tet_tags=None, barycenters_mm=None)
    with pytest.raises(ValueError, match="magnitude=True"):
        bare.peak_magnE(region)
    # With magnE retained the same call is computed on demand.
    full = _field_result()
    mask = metrics.region_mask(full.tet_tags, region)
    assert full.peak_magnE(region) == metrics.peak_magnitude(full.magnE, mask)


def test_non_default_focality_fraction_needs_the_magnitude():
    bare = _field_result(magnE=None, vols=None, tet_tags=None, barycenters_mm=None)
    assert bare.focality(0.5) == bare.summary["focality_m3"]["0.5"]  # precomputed
    with pytest.raises(ValueError, match="magnitude=True"):
        bare.focality(0.3)


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.9])
def test_field_result_focality_matches_metrics(frac):
    r = _field_result()
    mask = metrics.region_mask(r.tet_tags, "gray_matter")
    assert r.focality(frac) == pytest.approx(metrics.focality(r.magnE, r.vols, mask, frac))


def test_field_result_peak_matches_metrics():
    r = _field_result()
    mask = metrics.region_mask(r.tet_tags, "csf")
    assert r.peak_magnE("csf") == metrics.peak_magnitude(r.magnE, mask)
    np.testing.assert_array_equal(
        r.peak_location_mm("csf"), metrics.peak_location_mm(r.magnE, r.barycenters_mm, mask)
    )


def test_summary_for_is_cached_per_region():
    r = _field_result()
    assert r.summary_for("gray_matter") is r.summary  # the precomputed one, not recomputed
    assert r.summary_for("csf") is r.summary_for("csf")
    assert set(r._summaries) == {"csf"}


def test_field_result_unknown_region_raises():
    r = _field_result()
    with pytest.raises(ValueError, match="Unknown region"):
        r.peak_magnE("bone")  # ty:ignore[invalid-argument-type]


@pytest.mark.gpu
def test_iter_simulate_empty_sequence_yields_nothing(cube_subject, synthetic_coil):
    assert list(cube_subject.iter_simulate(synthetic_coil, [])) == []


@pytest.mark.gpu
def test_simulate_matches_iter_simulate_of_one(cube_subject, synthetic_coil):
    pl = Placement([50, 50, 100], [50, 100, 100])
    single = cube_subject.simulate(synthetic_coil, pl)
    (streamed,) = cube_subject.iter_simulate(synthetic_coil, [pl])
    assert isinstance(single, FieldResult) and isinstance(streamed, FieldResult)
    assert single.peak_magnE() == streamed.peak_magnE()


@pytest.mark.gpu
def test_simulate_rejects_a_sequence(cube_subject, synthetic_coil):
    """The eager N-result list is the OOM footgun; sequences must go through the generator."""
    pl = Placement([50, 50, 100], [50, 100, 100])
    for placements in ([pl], (pl,), []):
        with pytest.raises(TypeError, match="iter_simulate"):
            cube_subject.simulate(synthetic_coil, placements)


@pytest.mark.gpu
def test_iter_simulate_validates_eagerly(cube_subject, synthetic_coil):
    """Bad arguments raise at the call, not at the first next().

    Both generators are thin validating wrappers around an inner generator function, so a
    caller sees the error where they made the mistake rather than at the consumption site.
    Note there is no next() below.
    """
    pl = Placement([50, 50, 100], [50, 100, 100])
    cfg = cunibs.ConductivityUQConfig(n_samples=2, seed=0)

    # A bare Placement: list(placements) alone would say only 'not iterable'.
    with pytest.raises(TypeError, match="sequence of placements"):
        cube_subject.iter_simulate(synthetic_coil, pl)
    with pytest.raises(TypeError, match="sequence of placements"):
        cube_subject.iter_simulate_conductivity_uq(synthetic_coil, pl, cfg)


@pytest.mark.gpu
def test_simulate_conductivity_uq_rejects_a_sequence(cube_subject, synthetic_coil):
    cfg = cunibs.ConductivityUQConfig(n_samples=2, seed=0)
    pl = Placement([50, 50, 100], [50, 100, 100])
    with pytest.raises(TypeError, match="iter_simulate_conductivity_uq"):
        cube_subject.simulate_conductivity_uq(synthetic_coil, [pl], cfg)
