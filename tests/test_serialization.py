from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest

from cunibs import metrics
from cunibs.simulation import FieldResult, Placement
from cunibs.uq.conductivity import ConductivityUQResult


def _make_uq_result(*, with_rois: bool) -> ConductivityUQResult:
    rng = np.random.default_rng(1)
    m, n = 6, 4
    mean = rng.random(m) + 0.5
    std = rng.random(m) * 0.1
    r = ConductivityUQResult(
        mean_magnE=mean,
        std_magnE=std,
        cov_magnE=std / mean,
        n_samples=n,
        perturbed_tags=(2, 3),
        sigma_samples=rng.random((n, 2)) + 0.5,
        vols=rng.random(m) + 0.1,
        tet_tags=np.array([2, 2, 2, 3, 3, 3], dtype=np.int32),
        barycenters_mm=rng.standard_normal((m, 3)),
        placement=Placement([1, 2, 3], [4, 5, 6], 7.5),
        coil_name="synthetic",
        didt=1e6,
        peak_samples=rng.random(n) + 1.0,
        focality_samples=rng.random(n),
        peak_location_samples=rng.standard_normal((n, 3)),
        roi_samples={"m1": rng.random(n), "sma": rng.random(n)} if with_rois else {},
    )
    # Subject attaches this on the device; a hand-built result reduces on the host.
    return dataclasses.replace(r, summary=r.compute_summary())


def _make_result() -> FieldResult:
    rng = np.random.default_rng(2)
    m, n = 6, 4
    magnE = rng.random(m) + 0.5
    vols = rng.random(m) + 0.1
    tet_tags = np.array([2, 2, 2, 3, 3, 3], dtype=np.int32)
    barycenters_mm = rng.standard_normal((m, 3))
    return FieldResult(
        summary=metrics.compute_metrics(
            magnE, vols, barycenters_mm, tet_tags, region="gray_matter"
        ),
        magnE=magnE,
        E=rng.standard_normal((m, 3)),
        v=rng.standard_normal(n),
        transform=np.eye(4),
        placement=Placement([1, 2, 3], [4, 5, 6]),
        coil_name="synthetic",
        didt=1.5e6,
        vols=vols,
        tet_tags=tet_tags,
        barycenters_mm=barycenters_mm,
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


def _make_nodal_result() -> FieldResult:
    """A tissue-slot result: 4 nodes, one of which carries both a gray-matter and a CSF slot."""
    r = _make_result()
    return dataclasses.replace(
        r,
        recovery="harmonic",
        n_nodes=4,
        E_slots=np.arange(15, dtype=np.float32).reshape(5, 3),
        slot_node=np.array([0, 1, 1, 2, 3], dtype=np.int32),
        slot_tag=np.array([2, 2, 3, 3, 2], dtype=np.int32),
    )


def test_nodal_fields_survive_roundtrip(tmp_path):
    r = _make_nodal_result()
    path = tmp_path / "nodal.h5"
    r.save(path)
    loaded = FieldResult.load(path)

    for name in ("E_slots", "slot_node", "slot_tag"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(r, name), err_msg=name)
    assert loaded.recovery == "harmonic"
    assert loaded.n_nodes == 4
    # The whole point of carrying the slot maps: the field survives as a per-node answer.
    np.testing.assert_array_equal(
        loaded.nodal_field("gray_matter"), r.nodal_field("gray_matter")
    )


def test_nodal_field_resolves_each_tissue_to_its_own_slot():
    """Node 1 has a slot in each tissue, so the two regions must disagree there."""
    r = _make_nodal_result()
    gray = r.nodal_field("gray_matter")
    csf = r.nodal_field("csf")
    assert gray.shape == (4, 3)
    np.testing.assert_array_equal(gray[1], r.E_slots[1])
    np.testing.assert_array_equal(csf[1], r.E_slots[2])
    # Node 0 is gray only, so CSF cannot reach it.
    assert np.isnan(csf[0]).all()
    assert not np.isnan(gray[0]).any()


def test_recovery_defaults_to_raw_for_a_file_without_the_attribute(tmp_path):
    """A file with no recovery attribute holds a raw field, whatever the current default is."""
    r = _make_result()
    path = tmp_path / "old.h5"
    r.save(path)
    with h5py.File(path, "r+") as h5f:
        del h5f.attrs["recovery"]
    assert FieldResult.load(path).recovery == "raw"


def test_metrics_survive_roundtrip(tmp_path):
    r = _make_result()
    path = tmp_path / "result.h5"
    r.save(path)
    loaded = FieldResult.load(path)
    assert loaded.summary["peak_magnE"] == r.summary["peak_magnE"]


def test_placement_normalizes_inputs():
    p = Placement(center_mm=(1, 2, 3), handle_mm=[4.0, 5.0, 6.0])
    assert p.center_mm.shape == (3,)
    assert p.center_mm.dtype == np.float64
    assert p.distance_mm == 4.0


def test_placement_roundtrip_non_default_distance(tmp_path):
    r = _make_result()
    object.__setattr__(r.placement, "distance_mm", 7.5)
    path = tmp_path / "result.h5"
    r.save(path)
    assert FieldResult.load(path).placement.distance_mm == 7.5


def test_field_result_datasets_are_gzipped(tmp_path):
    path = tmp_path / "result.h5"
    _make_result().save(path)
    with h5py.File(path, "r") as f:
        for name in ("E", "magnE", "v", "vols", "tet_tags", "barycenters_mm"):
            assert f[name].compression == "gzip", name
        assert f.attrs["format_version"] == 1


def test_field_result_format_version_mismatch_raises(tmp_path):
    path = tmp_path / "result.h5"
    _make_result().save(path)
    with h5py.File(path, "r+") as f:
        f.attrs["format_version"] = 99
    with pytest.raises(ValueError, match=r"99.*expected 1"):
        FieldResult.load(path)


def test_field_result_missing_format_version_raises(tmp_path):
    """Files written before the version attribute existed read back as version 0."""
    path = tmp_path / "result.h5"
    _make_result().save(path)
    with h5py.File(path, "r+") as f:
        del f.attrs["format_version"]
    with pytest.raises(ValueError, match="version 0"):
        FieldResult.load(path)


def test_uq_result_roundtrip_without_record_rois(tmp_path):
    r = _make_uq_result(with_rois=False)
    path = tmp_path / "uq.h5"
    r.save(path)
    loaded = ConductivityUQResult.load(path)
    for name in ("mean_magnE", "std_magnE", "cov_magnE", "vols", "tet_tags", "sigma_samples"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(r, name), err_msg=name)
    assert loaded.n_samples == r.n_samples
    assert loaded.perturbed_tags == (2, 3)
    assert loaded.roi_samples == {}
    # Per-draw arrays are not tied to record_rois; they survive a round trip regardless.
    for name in ("peak_samples", "focality_samples", "peak_location_samples"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(r, name), err_msg=name)
    assert loaded.placement.distance_mm == 7.5


def test_uq_result_roundtrip_with_record_rois(tmp_path):
    r = _make_uq_result(with_rois=True)
    path = tmp_path / "uq.h5"
    r.save(path)
    loaded = ConductivityUQResult.load(path)
    assert set(loaded.roi_samples) == {"m1", "sma"}
    for key, value in r.roi_samples.items():
        np.testing.assert_array_equal(loaded.roi_samples[key], value)
    for name in ("peak_samples", "focality_samples", "peak_location_samples"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(r, name), err_msg=name)


def test_uq_result_format_version_mismatch_raises(tmp_path):
    """The UQ format version is tracked independently of the FieldResult one."""
    path = tmp_path / "uq.h5"
    _make_uq_result(with_rois=False).save(path)
    with h5py.File(path, "r+") as f:
        f.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="version 2 is not readable"):
        ConductivityUQResult.load(path)


def test_unretained_arrays_are_absent_from_the_file(tmp_path):
    factory, cls, kept, dropped = _make_result, FieldResult, "magnE", ("E", "v")
    r = dataclasses.replace(factory(), **dict.fromkeys(dropped))
    path = tmp_path / "partial.h5"
    r.save(path)
    with h5py.File(path, "r") as f:
        assert kept in f
        for name in dropped:
            assert name not in f, name
    loaded = cls.load(path)
    np.testing.assert_array_equal(getattr(loaded, kept), getattr(r, kept))
    for name in dropped:
        assert getattr(loaded, name) is None, name


def test_uq_summary_from_loaded_result(tmp_path):
    r = _make_uq_result(with_rois=True)
    path = tmp_path / "uq.h5"
    r.save(path)
    loaded = ConductivityUQResult.load(path)
    before, after = r.summary, loaded.summary
    assert before.mean_field["peak_magnE"] == after.mean_field["peak_magnE"]
    assert before.max_local_cov == after.max_local_cov
    assert before.mean_field["region_volume_m3"] == after.mean_field["region_volume_m3"]
    np.testing.assert_array_equal(
        before.mean_field["peak_location_mm"], after.mean_field["peak_location_mm"]
    )
