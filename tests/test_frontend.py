from __future__ import annotations

import numpy as np

from conftest import synthetic_coil
from cunibs.simulation import FieldResult, FieldSummary, Placement, Subject
from gpu import requires_gpu

pytestmark = requires_gpu


def _placements():
    return [
        Placement(center_mm=[50, 50, 100], handle_mm=[50, 100, 100]),
        Placement(center_mm=[50, 50, 100], handle_mm=[100, 50, 100]),
    ]


def test_simulate_many_reuses_context(cube_mesh):
    subj = Subject(cube_mesh)
    coil = synthetic_coil()
    results = subj.simulate(coil, _placements(), didt=1e6)
    assert isinstance(results, list) and len(results) == 2
    assert subj.context is subj.context
    for r in results:
        assert isinstance(r, FieldSummary)
        assert r.coil_name == "synthetic"
        assert r.peak_magnE() > 0


def test_simulate_single_returns_scalar_summary(cube_mesh):
    subj = Subject(cube_mesh)
    res = subj.simulate(synthetic_coil(), _placements()[0])
    assert isinstance(res, FieldSummary)
    summary = res.summary
    assert summary["peak_magnE"] > 0
    assert summary["region"] == "gray_matter"
    assert res.peak_location_mm().shape == (3,)


def test_retain_fields_cpu_returns_numpy_arrays(cube_mesh):
    subj = Subject(cube_mesh)
    results = subj.simulate(synthetic_coil(), _placements(), retain_fields=True)
    assert isinstance(results, list) and len(results) == 2
    for r in results:
        assert isinstance(r, FieldResult)
        assert isinstance(r.magnE, np.ndarray)
        assert isinstance(r.vols, np.ndarray)
    assert results[0].vols is results[1].vols


def test_retain_fields_gpu_returns_cupy_arrays(cube_mesh):
    import cupy as cp

    subj = Subject(cube_mesh)
    res = subj.simulate(synthetic_coil(), _placements()[0], retain_fields=True, device="gpu")
    assert isinstance(res, FieldResult)
    assert isinstance(res.magnE, cp.ndarray)


def test_result_to_numpy_and_serialize(tmp_path, cube_mesh):
    import cupy as cp

    subj = Subject(cube_mesh)
    res = subj.simulate(synthetic_coil(), _placements()[0], retain_fields=True, device="gpu")
    host = res.to_numpy()
    assert isinstance(host.magnE, np.ndarray)

    path = tmp_path / "res.h5"
    res.save(path)
    loaded = FieldResult.load(path)
    np.testing.assert_allclose(loaded.magnE, cp.asnumpy(res.magnE))
    assert loaded.summary()["peak_magnE"] == host.summary()["peak_magnE"]
