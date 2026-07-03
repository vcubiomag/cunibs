from __future__ import annotations

import numpy as np

from cunibs import coil as coil_registry
from cunibs.coil import Coil, encode_ccd

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
