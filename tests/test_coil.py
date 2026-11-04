from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest

from cunibs._common import plain
from cunibs.coil import Coil, bundled_coils, encode_ccd

BUNDLED = sorted(bundled_coils().items())

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
    # The header is A/us; the field is A/s. The raw string survives in metadata.
    assert c.didt_max == 100.0e6
    assert c.metadata is not None and c.metadata["dIdtmax"] == "100.0"
    np.testing.assert_allclose(c.positions_m, [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    np.testing.assert_allclose(c.moments, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert c.positions_m.shape == (2, 3) and c.moments.shape == (2, 3)


def test_bundled_didt_max_is_in_a_per_s():
    """dIdtmax is stored in A/us and exposed in A/s, so didt=coil.didt_max is a real pulse."""
    c = Coil.load("MagStim_D70")
    assert c.metadata is not None and c.metadata["dIdtmax"] == "114.73"
    assert c.didt_max == pytest.approx(114.73e6)


def test_registry_is_not_empty():
    assert len(BUNDLED) >= 25


def test_bundled_coils_is_reachable_from_the_top_level():
    """Enumerating the models is the discovery path, so it has to be visible on `cunibs`."""
    import cunibs

    assert cunibs.bundled_coils() == bundled_coils()
    assert "bundled_coils" in cunibs.__all__


def test_repr_names_the_coil_rather_than_dumping_its_dipoles():
    """A dataclass repr prints both tables in full, which is what enumerating the set would show."""
    c = Coil.load("MagStim_D70")
    text = repr(c)
    assert text == f"Coil(name={c.name!r}, n_dipoles={c.n_dipoles}, didt_max={c.didt_max!r})"
    assert c.n_dipoles == c.positions_m.shape[0]
    assert "array" not in text


@pytest.mark.parametrize("path", [p for _, p in BUNDLED], ids=[n for n, _ in BUNDLED])
def test_all_bundled_coils_load(path):
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
    # A/s, not the A/us the header carries: every bundled coil rates >= 84 A/us.
    assert c.didt_max is None or c.didt_max > 1e7
    assert c.metadata is not None and "coilname" in c.metadata


def test_coil_is_frozen():
    c = Coil.load("MagStim_D70")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.name = "other"  # ty:ignore[invalid-assignment]


def test_encode_ccd_keeps_every_dipole_without_a_comment_line(tmp_path):
    ccd = tmp_path / "synth.ccd"
    ccd.write_text("".join(line for line in _CCD.splitlines(True) if "x y z" not in line))
    h5 = tmp_path / "synth.h5"
    encode_ccd(ccd, h5)
    assert Coil.load(h5).n_dipoles == 2


def test_load_reads_a_ccd_with_a_comment_where_the_count_would_be(tmp_path):
    """Older SimNIBS files state the count only in the header's prose."""
    ccd = tmp_path / "legacy.ccd"
    ccd.write_text("".join(line for line in _CCD.splitlines(True) if line.strip() != "2"))
    assert Coil.load(ccd).n_dipoles == 2


def test_load_reads_a_ccd_that_rates_no_didt(tmp_path):
    ccd = tmp_path / "unrated.ccd"
    ccd.write_text(_CCD.replace("dIdtmax=100.0", "dIdtmax=none"))
    assert Coil.load(ccd).didt_max is None


def test_load_rejects_more_dipoles_than_the_ccd_declares(tmp_path):
    ccd = tmp_path / "long.ccd"
    ccd.write_text(_CCD + "0.02 0.0 0.0 0.0 0.0 1.0\n")
    with pytest.raises(ValueError, match="header says 2 dipoles, found 3"):
        Coil.load(ccd)


def test_encode_ccd_rejects_a_dipole_count_the_file_does_not_hold(tmp_path):
    ccd = tmp_path / "synth.ccd"
    ccd.write_text(_CCD.replace("\n2\n", "\n3\n", 1))
    with pytest.raises(ValueError, match="header says 3 dipoles, found 2"):
        encode_ccd(ccd, tmp_path / "synth.h5")


def test_encode_ccd_preserves_metadata(tmp_path):
    ccd = tmp_path / "synth.ccd"
    ccd.write_text(_CCD)
    h5 = tmp_path / "synth.h5"
    encode_ccd(ccd, h5)
    with h5py.File(h5, "r") as f:
        attrs = dict(f["dipoles"].attrs)
        assert f["dipoles"].shape == (2, 6)
    assert int(attrs["num_elements"]) == 2
    assert plain(attrs["brand"]) == "Test"
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
def test_plain_handles_bytes_and_np_scalars(raw, expected):
    assert plain(raw) == expected


def test_load_takes_a_bundled_name():
    by_name = Coil.load("MagStim_D70")
    by_path = Coil.load(bundled_coils()["MagStim_D70"])
    np.testing.assert_array_equal(by_name.positions_m, by_path.positions_m)
    assert by_name.name == by_path.name


def test_load_reads_a_ccd_as_its_encoding_would(tmp_path):
    ccd = tmp_path / "synth.ccd"
    ccd.write_text(_CCD)
    h5 = tmp_path / "synth.h5"
    encode_ccd(ccd, h5)
    direct, encoded = Coil.load(ccd), Coil.load(h5)
    np.testing.assert_array_equal(direct.moments, encoded.moments)
    assert direct.didt_max == encoded.didt_max
    assert direct.name == encoded.name == "Synth"


def test_load_names_the_bundled_models_when_nothing_matches():
    with pytest.raises(FileNotFoundError, match="MagStim_D70"):
        Coil.load("MagStim_D71")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_coil_rejects_non_finite_dipoles(bad):
    positions, moments = np.zeros((2, 3)), np.ones((2, 3))
    positions[1, 0] = bad
    with pytest.raises(ValueError, match="must be finite"):
        Coil(positions, moments)
    with pytest.raises(ValueError, match="must be finite"):
        Coil(moments, positions)


@pytest.mark.parametrize(
    ("positions", "moments"),
    [
        (np.zeros((2, 3)), np.ones((3, 3))),
        (np.zeros((2, 2)), np.ones((2, 2))),
        (np.zeros(3), np.ones(3)),
    ],
)
def test_coil_rejects_dipoles_that_are_not_paired_three_vectors(positions, moments):
    with pytest.raises(ValueError, match=r"must both be \(n, 3\)"):
        Coil(positions, moments)


def test_coil_takes_any_array_like_as_float64():
    c = Coil([[0, 0, 0]], np.ones((1, 3), np.float32))
    assert c.positions_m.dtype == c.moments.dtype == np.float64
    assert c.n_dipoles == 1


def test_load_rejects_an_hdf5_file_with_no_dipoles(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f["x"] = 1
    with pytest.raises(ValueError, match="no dipoles dataset"):
        Coil.load(path)


def test_load_finds_a_bundled_name_despite_a_directory_called_that(tmp_path, monkeypatch):
    (tmp_path / "MagStim_D70").mkdir()
    monkeypatch.chdir(tmp_path)
    assert Coil.load("MagStim_D70").n_dipoles > 0


def test_coils_compare_by_identity_and_hash():
    a, b = Coil.load("MagStim_D70"), Coil.load("MagStim_D70")
    assert a != b
    assert len({a, b, a}) == 2


def test_coil_load_rejects_malformed_dipoles(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f["dipoles"] = np.zeros(6)
    with pytest.raises(ValueError, match=r"must be \(n, 6\)"):
        Coil.load(path)


def test_encode_ccd_rejects_rows_that_are_not_six_columns(tmp_path):
    ccd = tmp_path / "short.ccd"
    ccd.write_text("# name=Bad\n1\n0.0 0.0 0.0 1.0 0.0\n")
    with pytest.raises(ValueError, match="6 columns"):
        encode_ccd(ccd, tmp_path / "short.h5")
