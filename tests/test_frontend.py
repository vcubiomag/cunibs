from __future__ import annotations

import numpy as np

from conftest import synthetic_coil
from cunibs.simulation import FieldResult, Placement, Subject
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
        assert r.coil_name == "synthetic"
        assert r.peak_magnE() > 0


def test_simulate_single_returns_scalar_result(cube_mesh):
    subj = Subject(cube_mesh)
    res = subj.simulate(synthetic_coil(), _placements()[0])
    assert isinstance(res, FieldResult)
    summary = res.summary()
    assert summary["peak_magnE"] > 0
    assert summary["region"] == "gray_matter"
    assert res.peak_location_mm().shape == (3,)


def test_result_to_numpy_and_serialize(tmp_path, cube_mesh):
    import cupy as cp

    subj = Subject(cube_mesh)
    res = subj.simulate(synthetic_coil(), _placements()[0])
    host = res.to_numpy()
    assert isinstance(host.magnE, np.ndarray)

    path = tmp_path / "res.h5"
    res.save(path)
    loaded = FieldResult.load(path)
    np.testing.assert_allclose(loaded.magnE, cp.asnumpy(res.magnE))
    assert loaded.summary()["peak_magnE"] == host.summary()["peak_magnE"]
