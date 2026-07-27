from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest

from cunibs import coil as coil_registry
from cunibs.coil import Coil, _decode_attr, encode_ccd

BUNDLED = sorted(
    (name, value)
    for name, value in vars(coil_registry).items()
    if not name.startswith("_") and name.isupper() and isinstance(value, Path)
)

_CCD = """\
# brand=Test;coilname=Synth;dIdtmax=100.0
2
# x y z mx my mz
0.0 0.0 0.0 0.0 0.0 1.0
0.01 0.0 0.0 0.0 0.0 -1.0
"""


def test_encode_ccd_roundtrips_to_coil(tmp_path):
    ccd = tmp_path / "synth.ccd"
    ccd.write_text(_CCD)
    h5 = tmp_path / "synth.h5"
    encode_ccd(ccd, h5)

    c = Coil.load(h5)
    assert c.name == "Synth"
    assert c.didt_max == 100.0
    np.testing.assert_allclose(c.positions_m, [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    np.testing.assert_allclose(c.moments, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert c.positions_m.shape == (2, 3) and c.moments.shape == (2, 3)


def test_registry_constants_resolve_to_files():
    paths = [
        coil_registry.MAGSTIM_D70,
        coil_registry.MAGVENTURE_MCF_B65,
        coil_registry.DEYMED_70BF,
    ]
    for p in paths:
        assert p.exists(), p


def test_load_bundled_coil():
    c = Coil.load(coil_registry.MAGSTIM_D70)
    assert c.positions_m.shape[1] == 3
    assert c.moments.shape == c.positions_m.shape
    assert c.name


def test_registry_is_not_empty():
    assert len(BUNDLED) >= 25


@pytest.mark.parametrize("path", [p for _, p in BUNDLED], ids=[n for n, _ in BUNDLED])
def test_all_bundled_coils_load(path):
    assert path.exists(), path
    c = Coil.load(path)
    n = c.positions_m.shape[0]
    assert n > 0
    assert c.positions_m.shape == (n, 3) and c.moments.shape == (n, 3)
    assert c.positions_m.dtype == np.float64 and c.moments.dtype == np.float64
    assert np.isfinite(c.positions_m).all() and np.isfinite(c.moments).all()
    # Dipoles live within a metre of the coil origin; a unit slip would be visible here.
    assert np.abs(c.positions_m).max() < 1.0
    assert np.abs(c.moments).max() > 0.0
    assert c.name
    assert c.didt_max is None or c.didt_max > 0
    assert c.metadata is not None and "coilname" in c.metadata


def test_coil_is_frozen():
    c = Coil.load(coil_registry.MAGSTIM_D70)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.name = "other"  # ty:ignore[invalid-assignment]


def test_encode_ccd_preserves_metadata(tmp_path):
    ccd = tmp_path / "synth.ccd"
    ccd.write_text(_CCD)
    h5 = tmp_path / "synth.h5"
    encode_ccd(ccd, h5)
    with h5py.File(h5, "r") as f:
        attrs = dict(f["dipoles"].attrs)
        assert f["dipoles"].shape == (2, 6)
    assert int(attrs["num_elements"]) == 2
    assert _decode_attr(attrs["brand"]) == "Test"
    assert Coil.load(h5).metadata["brand"] == "Test"


def test_coil_load_without_didtmax(tmp_path):
    """A file with no dIdtmax/coilname falls back to None and the file stem."""
    h5 = tmp_path / "Nameless.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("dipoles", data=np.zeros((3, 6)))
    c = Coil.load(h5)
    assert c.didt_max is None
    assert c.name == "Nameless"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b"bytes", "bytes"),
        ("plain", "plain"),
        (np.float64(1.5), 1.5),
        (np.int64(7), 7),
        (np.array([1, 2]), "[1 2]"),
    ],
)
def test_decode_attr_handles_bytes_and_np_scalars(raw, expected):
    assert _decode_attr(raw) == expected
