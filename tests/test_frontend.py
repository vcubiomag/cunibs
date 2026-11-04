from __future__ import annotations

import numpy as np
import pytest

from cunibs import metrics
from cunibs.simulation import FieldResult, Placement

pytestmark = pytest.mark.gpu


def _placements():
    return [
        Placement(center_mm=[50, 50, 100], handle_mm=[50, 100, 100]),
        Placement(center_mm=[50, 50, 100], handle_mm=[100, 50, 100]),
    ]


def _fan(n):
    """``n`` placements at one centre whose handles point different ways, so their fields differ."""
    return [
        Placement([50, 50, 100], [50 + 50 * np.cos(a), 50 + 50 * np.sin(a), 100])
        for a in np.linspace(0.2, 2.8, n)
    ]


def test_simulate_many_reuses_context(cube_subject, synthetic_coil):
    context = cube_subject.context
    results = list(cube_subject.iter_simulate(synthetic_coil, _placements(), didt=1e6))
    assert len(results) == 2
    assert cube_subject.context is context
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
    """Order must hold across the chunk boundary, so use more placements than a chunk holds."""
    from cunibs.simulation import _block_width

    sites = _fan(5)
    expected = [cube_subject.simulate(synthetic_coil, s).peak_magnE() for s in sites]
    assert len(set(expected)) == len(sites), "the placements must be told apart by their field"
    for width in (1, 2, 8):
        with _block_width(width):
            got = [r.peak_magnE() for r in cube_subject.iter_simulate(synthetic_coil, sites)]
        assert got == expected, f"width={width}"


def test_retained_results_are_independent_of_the_stream(cube_subject, synthetic_coil):
    """Each yield owns fresh arrays, so keeping one is safe."""
    from cunibs.simulation import _block_width

    sites = _fan(4)
    with _block_width(2):
        stream = cube_subject.iter_simulate(synthetic_coil, sites, magnitude=True)
        first = next(stream)
        kept = first.magnE.copy()
        rest = list(stream)
    assert len(rest) == 3
    assert not np.array_equal(rest[0].magnE, kept), "a reused buffer has to be visible"
    # Exhausting the generator must not disturb the held result.
    np.testing.assert_array_equal(first.magnE, kept)


def test_iter_simulate_survives_early_break(cube_subject, synthetic_coil):
    from cunibs.simulation import _block_width

    sites = _fan(6)
    want = cube_subject.simulate(synthetic_coil, sites[0], magnitude=True).magnE
    with _block_width(2):
        stream = cube_subject.iter_simulate(synthetic_coil, sites, magnitude=True)
        for i, r in enumerate(stream):
            assert r.peak_magnE() > 0
            if i == 1:
                break
        stream.close()
    # The subject stays usable after an abandoned sweep, whose copies were still in flight.
    again = cube_subject.simulate(synthetic_coil, sites[0], magnitude=True).magnE
    np.testing.assert_array_equal(again, want)


def test_optional_arrays_are_opt_in(cube_subject, synthetic_coil):
    res = cube_subject.simulate(synthetic_coil, _placements()[0], magnitude=True)
    assert res.E is None and res.v is None
    assert res.peak_magnE() > 0
    assert res.focality(0.5) > 0

    both = cube_subject.simulate(
        synthetic_coil, _placements()[0], magnitude=True, vectors=True, potential=True
    )
    assert isinstance(both.E, np.ndarray) and isinstance(both.v, np.ndarray)
    assert both.peak_magnE() == res.peak_magnE()


def test_summary_and_field_paths_agree(two_tissue_subject, synthetic_coil):
    """A summary-only run recovers just its summary's tissue; keeping |E| recovers them all.

    Neither may move the summary. It must also match the host metrics of the array kept.
    """
    sites = _placements()
    summaries = list(two_tissue_subject.iter_simulate(synthetic_coil, sites))
    fields = list(two_tissue_subject.iter_simulate(synthetic_coil, sites, magnitude=True))
    assert len(summaries) == len(fields) == len(sites)
    for s, f in zip(summaries, fields, strict=True):
        assert s.magnE is None
        assert s.peak_magnE() == f.peak_magnE()
        assert s.summary["focality_m3"] == f.summary["focality_m3"]
        assert s.summary["distribution"] == f.summary["distribution"]
        np.testing.assert_array_equal(s.peak_location_mm(), f.peak_location_mm())
        np.testing.assert_array_equal(s.center_of_gravity_mm(), f.center_of_gravity_mm())

        host = metrics.compute_metrics(f.magnE, f.vols, f.barycenters_mm, f.tet_tags)
        assert host["peak_magnE"] == f.peak_magnE()
        assert host["focality_m3"]["0.5"] == pytest.approx(f.focality(0.5), rel=1e-6)


def test_depth_probes_walk_inward(cp, cube_subject):
    subj = cube_subject
    point = [50.0, 50.0, 100.0]
    depths = [0.0, 20.0, 40.0]
    # The 6-tet cube resolves every depth to the same element at radius 0, so the radius has to be
    # wide enough for the ROI to change with depth.
    probes = subj.depth_probes(point, [0.0, 0.0, -1.0], depths, radius_mm=40.0)
    assert len(probes) == len(depths)

    # Each probe is the ROI at the point that far inward along the direction.
    for depth, probe in zip(depths, probes, strict=True):
        expected = subj.roi([50.0, 50.0, 100.0 - depth], radius_mm=40.0, region="all")
        np.testing.assert_array_equal(cp.asnumpy(probe.elem_idx), cp.asnumpy(expected.elem_idx))
        np.testing.assert_allclose(
            cp.asnumpy(probe.barycenter_mm), cp.asnumpy(expected.barycenter_mm)
        )

    # inward_dir is normalized, so its magnitude must not shift the probe spacing.
    scaled = subj.depth_probes(point, [0.0, 0.0, -7.5], depths, radius_mm=40.0)
    for probe, other in zip(probes, scaled, strict=True):
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
    np.testing.assert_array_equal(loaded.magnE, res.magnE)
    np.testing.assert_array_equal(loaded.E, res.E)
    np.testing.assert_array_equal(loaded.v, res.v)
    assert loaded.summary["peak_magnE"] == res.summary["peak_magnE"]


def test_partial_fields_serialize_round_trip(tmp_path, cube_subject, synthetic_coil):
    res = cube_subject.simulate(synthetic_coil, _placements()[0], magnitude=True)
    path = tmp_path / "partial.h5"
    res.save(path)
    loaded = FieldResult.load(path)
    assert loaded.E is None and loaded.v is None
    np.testing.assert_array_equal(loaded.magnE, res.magnE)
