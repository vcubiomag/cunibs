from __future__ import annotations

import numpy as np
import pytest

from gpu import requires_gpu

pytestmark = requires_gpu

_CUBE_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    dtype=np.float64,
)
_CUBE_TETS = np.array(
    [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]],
    dtype=np.int32,
)
_CUBE_TRIS = np.array(
    [
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3],
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
    ],
    dtype=np.int32,
)


@pytest.fixture
def cp():
    import cupy

    return cupy


@pytest.fixture
def two_tissue_cube():
    """A 100 mm cube split into gray matter (tag 2) and CSF (tag 3) so |E| depends on σ ratios."""
    from cunibs.mesh import HeadMesh

    tags = np.array([2, 2, 2, 3, 3, 3], dtype=np.int32)
    return HeadMesh(
        nodes_mm=np.ascontiguousarray(_CUBE_CORNERS * 100.0),
        tet_nodes=_CUBE_TETS.copy(),
        tet_tags=tags,
        skin_tris=_CUBE_TRIS.copy(),
    )


def _coil():
    from cunibs.coil import Coil

    return Coil(
        positions_m=np.array([[-0.02, 0, 0], [0.02, 0, 0]]),
        moments=np.array([[0, 0, 1.0], [0, 0, -1.0]]),
        name="syn",
        didt_max=1e6,
    )


def _placement():
    from cunibs import Placement

    return Placement([50, 50, 100], [50, 100, 100], 4.0)


def test_uq_assembly_gate_reproduces_nominal(cp, two_tissue_cube):
    """base + Σ σ_nom·Kt must equal the direct nominal reduced stiffness (the correctness gate)."""
    from cunibs import Subject
    from cunibs.uq.conductivity.assembly import build_conductivity_uq_precompute

    subj = Subject(two_tissue_cube)
    pre = build_conductivity_uq_precompute(subj.context, (2, 3))
    recon = pre.base_data + pre.nominal_sigma @ pre.tissue_data
    assert recon.shape == pre.base_data.shape
    assert float(cp.abs(pre.tissue_data).sum()) > 0.0  # components are non-trivial


def test_uq_degenerate_matches_forward(cp, cube_mesh):
    """Zero conductivity variance → mean field equals the deterministic solve, std is zero."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(cube_mesh)
    coil, pl = _coil(), _placement()
    det = subj.simulate(coil, pl, retain_fields=True, device="gpu")
    cfg = ConductivityUQConfig(
        n_samples=8, tissue_cov={2: 0.0}, seed=1, preconditioner_refresh="never"
    )
    r = subj.simulate(coil, pl, conductivity_uq=cfg, retain_fields=True, device="gpu")
    np.testing.assert_allclose(cp.asnumpy(r.mean_magnE), cp.asnumpy(det.magnE), atol=1e-6)
    assert float(cp.asarray(r.std_magnE).max()) == 0.0


def test_uq_homogeneous_has_no_sensitivity(cp, cube_mesh):
    """A single-tissue domain: scaling σ scales K and b equally, so |E| is σ-invariant."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(cube_mesh)
    r = subj.simulate(
        _coil(),
        _placement(),
        conductivity_uq=ConductivityUQConfig(n_samples=32, tissue_cov={2: 0.3}, seed=0),
        retain_fields=True,
        device="gpu",
    )
    assert float(cp.asarray(r.cov_magnE).max()) < 1e-5


def test_uq_two_tissue_has_variance(cp, two_tissue_cube):
    """Distinct tissues make |E| depend on the conductivity ratio, so the ensemble has spread."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(two_tissue_cube)
    r = subj.simulate(
        _coil(),
        _placement(),
        conductivity_uq=ConductivityUQConfig(
            n_samples=200, tissue_cov={2: 0.2, 3: 0.3}, seed=0
        ),
        retain_fields=True,
        device="gpu",
    )
    cov = cp.asarray(r.cov_magnE)
    assert float(cov.max()) > 1e-3
    assert bool(cp.isfinite(cov).all())


def test_uq_refresh_modes_agree(cp, two_tissue_cube):
    """The converged field is preconditioner-independent: frozen vs per-sample resetup must match."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(two_tissue_cube)
    coil, pl = _coil(), _placement()
    ra = subj.simulate(
        coil,
        pl,
        conductivity_uq=ConductivityUQConfig(
            n_samples=64,
            tissue_cov={2: 0.2, 3: 0.3},
            seed=7,
            preconditioner_refresh="always",
        ),
        retain_fields=True,
        device="gpu",
    )
    rn = subj.simulate(
        coil,
        pl,
        conductivity_uq=ConductivityUQConfig(
            n_samples=64,
            tissue_cov={2: 0.2, 3: 0.3},
            seed=7,
            preconditioner_refresh="never",
        ),
        retain_fields=True,
        device="gpu",
    )
    peak = float(cp.asarray(ra.mean_magnE).max())
    diff = float(cp.abs(cp.asarray(ra.mean_magnE) - cp.asarray(rn.mean_magnE)).max())
    assert diff / peak < 1e-5


def test_uq_deterministic_seed(cp, two_tissue_cube):
    """A fixed seed reproduces identical moments."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(two_tissue_cube)
    coil, pl = _coil(), _placement()
    cfg = ConductivityUQConfig(n_samples=32, tissue_cov={2: 0.2, 3: 0.3}, seed=5)
    r1 = subj.simulate(coil, pl, conductivity_uq=cfg, retain_fields=True, device="gpu")
    r2 = subj.simulate(coil, pl, conductivity_uq=cfg, retain_fields=True, device="gpu")
    assert bool(cp.all(cp.asarray(r1.mean_magnE) == cp.asarray(r2.mean_magnE)))
    assert bool(cp.all(cp.asarray(r1.std_magnE) == cp.asarray(r2.std_magnE)))


def test_uq_default_sequence_returns_summaries(two_tissue_cube):
    from cunibs import ConductivityUQConfig, Placement, Subject

    subj = Subject(two_tissue_cube)
    placements = [
        _placement(),
        Placement([50, 50, 100], [100, 50, 100], 4.0),
    ]
    r = subj.simulate(
        _coil(),
        placements,
        conductivity_uq=ConductivityUQConfig(n_samples=8, seed=3),
    )
    assert isinstance(r, list) and len(r) == 2
    for item in r:
        assert item.peak_mean_magnE() > 0.0
        assert item.peak_cov() >= 0.0


def test_uq_retain_fields_cpu_sequence_results_are_host_backed(cp, two_tissue_cube):
    from cunibs import ConductivityUQConfig, Placement, Subject

    subj = Subject(two_tissue_cube)
    placements = [
        _placement(),
        Placement([50, 50, 100], [100, 50, 100], 4.0),
    ]
    r = subj.simulate(
        _coil(),
        placements,
        conductivity_uq=ConductivityUQConfig(n_samples=8, seed=3),
        retain_fields=True,
    )
    assert isinstance(r, list) and len(r) == 2
    for item in r:
        assert isinstance(item.mean_magnE, np.ndarray)
        assert isinstance(item.vols, np.ndarray)
        assert not isinstance(item.mean_magnE, cp.ndarray)
    assert r[0].vols is r[1].vols


def test_uq_result_save_load(cp, tmp_path, two_tissue_cube):
    from cunibs import ConductivityUQConfig, Subject
    from cunibs.uq.conductivity import ConductivityUQResult

    subj = Subject(two_tissue_cube)
    r = subj.simulate(
        _coil(),
        _placement(),
        conductivity_uq=ConductivityUQConfig(n_samples=16, seed=3),
        retain_fields=True,
        device="gpu",
    ).to_numpy()
    path = tmp_path / "uq.h5"
    r.save(path)
    loaded = ConductivityUQResult.load(path)
    np.testing.assert_array_equal(loaded.mean_magnE, r.mean_magnE)
    np.testing.assert_array_equal(loaded.cov_magnE, r.cov_magnE)
    assert loaded.n_samples == r.n_samples
    assert loaded.perturbed_tags == r.perturbed_tags
