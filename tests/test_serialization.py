from __future__ import annotations

import numpy as np

from cunibs.simulation import FieldResult, Placement


def _make_result() -> FieldResult:
    rng = np.random.default_rng(0)
    m, n = 6, 8
    return FieldResult(
        E=rng.standard_normal((m, 3)),
        magnE=rng.random(m) + 0.1,
        v=rng.standard_normal(n),
        transform=np.eye(4),
        placement=Placement(center_mm=[1, 2, 3], handle_mm=[4, 5, 6], distance_mm=4.0),
        coil_name="synthetic",
        didt=1.5e6,
        vols=rng.random(m) + 0.1,
        tet_tags=np.array([2, 2, 2, 2, 5, 5], dtype=np.int32),
        barycenters_mm=rng.standard_normal((m, 3)),
    )


def test_fieldresult_hdf5_roundtrip(tmp_path):
    r = _make_result()
    path = tmp_path / "result.h5"
    r.save(path)
    loaded = FieldResult.load(path)

    for name in ("E", "magnE", "v", "transform", "vols", "tet_tags", "barycenters_mm"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(r, name), err_msg=name)
    assert loaded.coil_name == "synthetic"
    assert loaded.didt == 1.5e6
    np.testing.assert_allclose(loaded.placement.center_mm, [1, 2, 3])
    np.testing.assert_allclose(loaded.placement.handle_mm, [4, 5, 6])
    assert loaded.placement.distance_mm == 4.0


def test_metrics_survive_roundtrip(tmp_path):
    r = _make_result()
    path = tmp_path / "result.h5"
    r.save(path)
    loaded = FieldResult.load(path)
    assert loaded.summary()["peak_magnE"] == r.summary()["peak_magnE"]


def test_placement_normalizes_inputs():
    p = Placement(center_mm=(1, 2, 3), handle_mm=[4.0, 5.0, 6.0])
    assert p.center_mm.shape == (3,)
    assert p.center_mm.dtype == np.float64
    assert p.distance_mm == 4.0
