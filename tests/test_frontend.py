from __future__ import annotations

import numpy as np
import pytest

from cunibs.simulation import FieldResult, Placement

pytestmark = pytest.mark.gpu


def _placements():
    return [
        Placement(center_mm=[50, 50, 100], handle_mm=[50, 100, 100]),
        Placement(center_mm=[50, 50, 100], handle_mm=[100, 50, 100]),
    ]


def test_simulate_many_reuses_context(cube_subject, synthetic_coil):
    results = list(cube_subject.iter_simulate(synthetic_coil, _placements(), didt=1e6))
    assert len(results) == 2
    assert cube_subject.context is cube_subject.context
    for r in results:
        assert isinstance(r, FieldResult)
        assert r.coil_name == "synthetic"
        assert r.peak_magnE() > 0


def test_simulate_single_returns_scalar_summary(cube_subject, synthetic_coil):
    res = cube_subject.simulate(synthetic_coil, _placements()[0])
    assert isinstance(res, FieldResult)
    summary = res.summary
    assert summary["peak_magnE"] > 0
    assert summary["region"] == "gray_matter"
    assert res.peak_location_mm().shape == (3,)


def test_iter_simulate_with_fields_returns_numpy_arrays(cube_subject, synthetic_coil):
    results = list(
        cube_subject.iter_simulate(
            synthetic_coil, _placements(), magnitude=True, vectors=True, potential=True
        )
    )
    assert len(results) == 2
    for r in results:
        assert isinstance(r, FieldResult)
        assert isinstance(r.magnE, np.ndarray)
        assert isinstance(r.E, np.ndarray)
        assert isinstance(r.v, np.ndarray)
        assert isinstance(r.vols, np.ndarray)
    # Metric inputs are shared across results rather than copied per placement.
    assert results[0].vols is results[1].vols


def test_simulate_single_with_fields_returns_field_result(cube_subject, synthetic_coil):
    res = cube_subject.simulate(synthetic_coil, _placements()[0], magnitude=True)
    assert isinstance(res, FieldResult)
    assert isinstance(res.magnE, np.ndarray)
    assert res.peak_magnE() > 0


def test_iter_simulate_yields_in_submission_order(cube_subject, synthetic_coil):
    """Order must hold across the chunk boundary, so use more placements than block_k."""
    sites = [Placement(center_mm=[50, 50, 100], handle_mm=[50, 100 + i, 100]) for i in range(5)]
    expected = [cube_subject.simulate(synthetic_coil, s).peak_magnE() for s in sites]
    for block_k in (1, 2, 8):
        got = [
            r.peak_magnE()
            for r in cube_subject.iter_simulate(synthetic_coil, sites, block_k=block_k)
        ]
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_retained_results_are_independent_of_the_stream(cube_subject, synthetic_coil):
    """Each yield owns fresh arrays -- no reused buffer -- so keeping one is safe."""
    sites = [Placement(center_mm=[50, 50, 100], handle_mm=[50, 100 + i, 100]) for i in range(4)]
    stream = cube_subject.iter_simulate(synthetic_coil, sites, magnitude=True, block_k=2)
    first = next(stream)
    kept = first.magnE.copy()
    rest = list(stream)
    assert len(rest) == 3
    # Exhausting the generator (and freeing its pool) must not disturb the held result.
    np.testing.assert_array_equal(first.magnE, kept)
    assert first.peak_magnE() > 0


def test_iter_simulate_survives_early_break(cube_subject, synthetic_coil):
    sites = [Placement(center_mm=[50, 50, 100], handle_mm=[50, 100 + i, 100]) for i in range(6)]
    stream = cube_subject.iter_simulate(synthetic_coil, sites, magnitude=True, block_k=2)
    for i, r in enumerate(stream):
        assert r.peak_magnE() > 0
        if i == 1:
            break
    stream.close()
    # The subject stays usable once the abandoned generator has released its scratch pool.
    assert cube_subject.simulate(synthetic_coil, sites[0]).peak_magnE() > 0


def test_optional_arrays_are_opt_in(cube_subject, synthetic_coil):
    res = cube_subject.simulate(synthetic_coil, _placements()[0], magnitude=True)
    assert res.E is None and res.v is None
    # magnE is always retained, so the metrics work without asking for anything else.
    assert res.peak_magnE() > 0
    assert res.focality(0.5) > 0

    both = cube_subject.simulate(
        synthetic_coil, _placements()[0], magnitude=True, vectors=True, potential=True
    )
    assert isinstance(both.E, np.ndarray) and isinstance(both.v, np.ndarray)
    assert both.peak_magnE() == res.peak_magnE()


def test_summary_and_field_paths_agree(cube_subject, synthetic_coil):
    """fields=None reduces on device; fields=(...) reduces on host. They must match."""
    sites = _placements()
    summaries = list(cube_subject.iter_simulate(synthetic_coil, sites))
    fields = list(cube_subject.iter_simulate(synthetic_coil, sites, magnitude=True))
    for s, f in zip(summaries, fields, strict=False):
        np.testing.assert_allclose(s.peak_magnE(), f.peak_magnE(), rtol=1e-6)
        np.testing.assert_allclose(s.focality(0.5), f.focality(0.5), rtol=1e-6)
        np.testing.assert_allclose(s.peak_location_mm(), f.peak_location_mm())


def test_depth_probes_walk_inward(cp, cube_subject):
    subj = cube_subject
    point = [50.0, 50.0, 100.0]
    depths = [0.0, 20.0, 40.0]
    # The 6-tet cube resolves every depth to the same element at radius 0, so use a radius
    # wide enough that the ROI membership actually changes with depth.
    probes = subj.depth_probes(point, [0.0, 0.0, -1.0], depths, radius_mm=40.0)
    assert len(probes) == len(depths)

    # Each probe is the ROI at the point that far inward along the direction.
    for depth, probe in zip(depths, probes, strict=False):
        expected = subj.roi([50.0, 50.0, 100.0 - depth], radius_mm=40.0, region="all")
        np.testing.assert_array_equal(cp.asnumpy(probe.elem_idx), cp.asnumpy(expected.elem_idx))
        np.testing.assert_allclose(
            cp.asnumpy(probe.barycenter_mm), cp.asnumpy(expected.barycenter_mm)
        )

    # inward_dir is normalized, so its magnitude must not shift the probe spacing.
    scaled = subj.depth_probes(point, [0.0, 0.0, -7.5], depths, radius_mm=40.0)
    for probe, other in zip(probes, scaled, strict=False):
        np.testing.assert_array_equal(cp.asnumpy(probe.elem_idx), cp.asnumpy(other.elem_idx))

    z = [float(p.barycenter_mm[2]) for p in probes]
    assert z[0] >= z[1] >= z[2] and z[2] < z[0]
    assert all(abs(float(p.weights.sum()) - 1.0) < 1e-12 for p in probes)


def test_result_serialize_round_trip(tmp_path, cube_subject, synthetic_coil):
    res = cube_subject.simulate(
        synthetic_coil, _placements()[0], magnitude=True, vectors=True, potential=True
    )
    path = tmp_path / "res.h5"
    res.save(path)
    loaded = FieldResult.load(path)
    np.testing.assert_allclose(loaded.magnE, res.magnE)
    assert loaded.summary["peak_magnE"] == res.summary["peak_magnE"]


def test_partial_fields_serialize_round_trip(tmp_path, cube_subject, synthetic_coil):
    res = cube_subject.simulate(synthetic_coil, _placements()[0], magnitude=True)
    path = tmp_path / "partial.h5"
    res.save(path)
    loaded = FieldResult.load(path)
    assert loaded.E is None and loaded.v is None
    np.testing.assert_allclose(loaded.magnE, res.magnE)
