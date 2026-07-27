from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def two_tissue_cube(two_tissue_cube_mesh):
    return two_tissue_cube_mesh


def _coil():
    from cunibs.coil import Coil

    return Coil(
        positions_m=np.array([[-0.02, 0, 0], [0.02, 0, 0]]),
        moments=np.array([[0, 0, 1.0], [0, 0, -1.0]]),
        name="syn",
        didt_max=100e6,
    )


def _placement():
    from cunibs import Placement

    return Placement([50, 50, 100], [50, 100, 100], 4.0)


def _direct_reduced_stiffness(ctx, sigma_by_tag):
    """Assemble the grounded stiffness from scratch at explicit per-tag conductivities."""
    import cupy as cp

    from cunibs.fem.assembly import assemble_stiffness, gradient_operator
    from cunibs.fem.solve import ground_node_of, grounded_index, reduce_matrix

    g64, vols = gradient_operator(ctx.nodes_mm * 1e-3, ctx.tet_nodes)
    cond = cp.zeros(int(ctx.tet_nodes.shape[0]), dtype=cp.float64)
    for tag, value in sigma_by_tag.items():
        cond[ctx.tet_tags == tag] = value
    k = assemble_stiffness(g64, vols, cond, ctx.n_nodes, ctx.tet_nodes)
    return reduce_matrix(k, grounded_index(ctx.n_nodes, ground_node_of(ctx.nodes_mm)))


def test_uq_precompute_matches_direct_nominal_assembly(cp, two_tissue_subject):
    """base + Σ σ_nom·K_t must equal an independently assembled nominal stiffness.

    The library's own gate (assembly.py's ``rel > 1e-10`` RuntimeError) compares the
    decomposition against the same ``k_ref`` it was built from, so it cannot catch a shared
    error; this rebuilds the reference from ``gradient_operator``/``assemble_stiffness``.
    """
    from cunibs.fem.assembly import TISSUE_CONDUCTIVITY
    from cunibs.uq.conductivity.assembly import build_conductivity_uq_precompute

    ctx = two_tissue_subject.context
    pre = build_conductivity_uq_precompute(ctx, (2, 3))
    ref = _direct_reduced_stiffness(ctx, {t: TISSUE_CONDUCTIVITY[t] for t in (2, 3)})

    np.testing.assert_array_equal(cp.asnumpy(pre.indptr), cp.asnumpy(ref.indptr))
    np.testing.assert_array_equal(cp.asnumpy(pre.indices), cp.asnumpy(ref.indices))
    recon = pre.base_data + pre.nominal_sigma @ pre.tissue_data
    rel = float(cp.linalg.norm(recon - ref.data) / cp.linalg.norm(ref.data))
    assert rel <= 1e-12
    assert float(cp.abs(pre.tissue_data).sum()) > 0.0


@pytest.mark.parametrize("sigma", [(0.4, 2.1), (1e-3, 5.0), (1.0, 1.0)])
def test_uq_combine_matches_direct_assembly_at_nonnominal_sigma(cp, two_tissue_subject, sigma):
    """``combine`` away from nominal — the point the library's internal gate never checks."""
    from cunibs.uq.conductivity.assembly import build_conductivity_uq_precompute

    ctx = two_tissue_subject.context
    pre = build_conductivity_uq_precompute(ctx, (2, 3))
    got = pre.combine(cp.asarray(sigma, dtype=cp.float64))
    ref = _direct_reduced_stiffness(ctx, {2: sigma[0], 3: sigma[1]})
    rel = float(cp.linalg.norm(got - ref.data) / cp.linalg.norm(ref.data))
    assert rel <= 1e-12


def test_uq_combine_is_affine_in_sigma(cp, two_tissue_subject):
    """K(σ) is affine, so combine must satisfy K(a+b) - K(a) = K(b) - K(0) exactly."""
    from cunibs.uq.conductivity.assembly import build_conductivity_uq_precompute

    pre = build_conductivity_uq_precompute(two_tissue_subject.context, (2, 3))
    a = cp.asarray([0.3, 1.7])
    b = cp.asarray([1.1, 0.2])
    zero = cp.zeros(2)
    lhs = pre.combine(a + b) - pre.combine(a)
    rhs = pre.combine(b) - pre.combine(zero)
    np.testing.assert_allclose(cp.asnumpy(lhs), cp.asnumpy(rhs), rtol=1e-12, atol=1e-18)


def test_uq_precompute_gate_fires_on_corruption(monkeypatch, two_tissue_subject):
    """Corrupting one component must trip the built-in decomposition gate."""
    import cunibs.uq.conductivity.assembly as uq_assembly

    real = uq_assembly.assemble_stiffness
    calls = {"n": 0}

    def flaky(g, vols, cond, n_nodes, tet_nodes):
        calls["n"] += 1
        out = real(g, vols, cond, n_nodes, tet_nodes)
        if calls["n"] == 2:  # the first per-tissue component
            out.data *= 1.05
        return out

    monkeypatch.setattr(uq_assembly, "assemble_stiffness", flaky)
    with pytest.raises(RuntimeError, match="decomposition mismatch"):
        uq_assembly.build_conductivity_uq_precompute(two_tissue_subject.context, (2, 3))


def test_uq_precompute_cached_per_tagset(two_tissue_subject):
    from cunibs import ConductivityUQConfig

    subj = two_tissue_subject
    both = subj._conductivity_uq_precompute(ConductivityUQConfig(perturbed_tags=(2, 3)))
    again = subj._conductivity_uq_precompute(ConductivityUQConfig(perturbed_tags=(2, 3)))
    csf_only = subj._conductivity_uq_precompute(ConductivityUQConfig(perturbed_tags=(3,)))
    assert both is again
    assert csf_only is not both
    assert csf_only.perturbed_tags == (3,)


def test_uq_perturbed_tags_subset(two_tissue_subject):
    """Perturbing only CSF must leave the gray-matter elements' conductivity fixed."""
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(
            n_samples=32, tissue_cov={3: 0.3}, perturbed_tags=(3,), seed=0
        ),
        moments=True,
    )
    assert r.perturbed_tags == (3,)
    assert r.sigma_samples.shape == (32, 1)


def test_uq_degenerate_matches_forward(cp, cube_mesh):
    """Zero conductivity variance → mean field equals the deterministic solve, std is zero."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(cube_mesh)
    coil, pl = _coil(), _placement()
    det = subj.simulate(coil, pl, magnitude=True, vectors=True, potential=True)
    cfg = ConductivityUQConfig(n_samples=8, tissue_cov={2: 0.0}, seed=1)
    r = subj.simulate_conductivity_uq(coil, pl, config=cfg, moments=True)
    np.testing.assert_allclose(cp.asnumpy(r.mean_magnE), cp.asnumpy(det.magnE), atol=1e-6)
    assert float(cp.asarray(r.std_magnE).max()) == 0.0


@pytest.mark.realmesh
def test_per_draw_focality_uses_the_same_anchor_as_the_summary(
    patch_subject, patch_placement, d70_coil
):
    """With no conductivity spread every draw is the mean field, so the two must agree.

    Needs the patch: the cube's six gray-matter elements make p99.9 saturate to the max, so
    both anchors would agree there whether or not they are the same.
    """
    from cunibs import ConductivityUQConfig

    # Every tissue pinned: unlisted tags fall back to DEFAULT_TISSUE_COV and would vary.
    cfg = ConductivityUQConfig(n_samples=4, tissue_cov={2: 0.0}, perturbed_tags=(2,), seed=1)
    r = patch_subject.simulate_conductivity_uq(
        d70_coil, patch_placement, config=cfg, moments=True, record_rois={}
    )

    assert r.summary is not None
    expected = r.summary.mean_field["focality_m3"]["0.5"]
    np.testing.assert_allclose(r.focality_samples, expected, rtol=1e-6)


def test_uq_homogeneous_has_no_sensitivity(cp, cube_mesh):
    """A single-tissue domain: scaling σ scales K and b equally, so |E| is σ-invariant."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(cube_mesh)
    r = subj.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=32, tissue_cov={2: 0.3}, seed=0),
        moments=True,
    )
    assert float(cp.asarray(r.cov_magnE).max()) < 1e-5


def test_uq_two_tissue_has_variance(cp, two_tissue_cube):
    """Distinct tissues make |E| depend on the conductivity ratio, so the ensemble has spread."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(two_tissue_cube)
    r = subj.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=200, tissue_cov={2: 0.2, 3: 0.3}, seed=0),
        moments=True,
    )
    cov = cp.asarray(r.cov_magnE)
    assert float(cov.max()) > 1e-3
    assert bool(cp.isfinite(cov).all())


def test_uq_deterministic_seed(cp, two_tissue_cube):
    """A fixed seed reproduces identical moments."""
    from cunibs import ConductivityUQConfig, Subject

    subj = Subject(two_tissue_cube)
    coil, pl = _coil(), _placement()
    cfg = ConductivityUQConfig(n_samples=32, tissue_cov={2: 0.2, 3: 0.3}, seed=5)
    r1 = subj.simulate_conductivity_uq(coil, pl, config=cfg, moments=True)
    r2 = subj.simulate_conductivity_uq(coil, pl, config=cfg, moments=True)
    assert bool(cp.all(cp.asarray(r1.mean_magnE) == cp.asarray(r2.mean_magnE)))
    assert bool(cp.all(cp.asarray(r1.std_magnE) == cp.asarray(r2.std_magnE)))


def test_uq_default_sequence_returns_summaries(two_tissue_cube):
    from cunibs import ConductivityUQConfig, Placement, Subject

    subj = Subject(two_tissue_cube)
    placements = [
        _placement(),
        Placement([50, 50, 100], [100, 50, 100], 4.0),
    ]
    r = list(
        subj.iter_simulate_conductivity_uq(
            _coil(),
            placements,
            config=ConductivityUQConfig(n_samples=8, seed=3),
        )
    )
    assert len(r) == 2
    for item in r:
        assert item.peak_mean_magnE() > 0.0
        assert item.peak_cov() >= 0.0


def test_uq_streamed_sequence_results_are_host_backed(cp, two_tissue_cube):
    from cunibs import ConductivityUQConfig, Placement, Subject

    subj = Subject(two_tissue_cube)
    placements = [
        _placement(),
        Placement([50, 50, 100], [100, 50, 100], 4.0),
    ]
    r = list(
        subj.iter_simulate_conductivity_uq(
            _coil(),
            placements,
            config=ConductivityUQConfig(n_samples=8, seed=3),
            moments=True,
        )
    )
    assert len(r) == 2
    for item in r:
        assert isinstance(item.mean_magnE, np.ndarray)
        assert isinstance(item.vols, np.ndarray)
        assert not isinstance(item.mean_magnE, cp.ndarray)
    assert r[0].vols is r[1].vols


def test_record_rois_per_draw_arrays(cp, two_tissue_subject):
    """record_rois adds the per-draw distributional data a metric of the mean field cannot give."""
    from cunibs import ConductivityUQConfig

    subj = two_tissue_subject
    rois = {
        "centre": subj.roi([50, 50, 50], radius_mm=40.0),
        "top": subj.roi([50, 50, 90], radius_mm=40.0),
    }
    n = 64
    r = subj.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=n, tissue_cov={2: 0.2, 3: 0.3}, seed=2),
        moments=True,
        record_rois=rois,
    )
    assert set(r.roi_samples) == {"centre", "top"}
    for arr in r.roi_samples.values():
        assert arr.shape == (n,) and np.isfinite(arr).all() and (arr > 0).all()
    assert r.peak_samples.shape == (n,)
    assert r.focality_samples.shape == (n,)
    assert r.peak_location_samples.shape == (n, 3)
    assert np.isfinite(r.peak_samples).all()

    # The ROI mean over draws must track the same ROI on the mean field, to Monte Carlo error.
    idx = cp.asnumpy(rois["centre"].elem_idx)
    w = cp.asnumpy(rois["centre"].weights)
    on_mean_field = float(w @ cp.asnumpy(r.mean_magnE)[idx])
    samples = r.roi_samples["centre"]
    assert abs(samples.mean() - on_mean_field) < 5 * samples.std() / np.sqrt(n) + 1e-12


def test_record_rois_empty_mapping_records_whole_field_only(two_tissue_subject):
    """``record_rois={}`` is meaningful: no ROIs, but the whole-field per-draw metrics appear."""
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=16, tissue_cov={2: 0.2, 3: 0.3}, seed=4),
        moments=True,
        record_rois={},
    )
    assert r.roi_samples == {}
    assert r.peak_samples is not None and r.peak_samples.shape == (16,)


def test_no_record_rois_leaves_per_draw_arrays_unset(two_tissue_subject):
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=8, seed=4),
        moments=True,
    )
    assert r.roi_samples is None
    assert r.peak_samples is None and r.focality_samples is None


def test_tissue_sensitivity_identifies_the_varying_tissue(two_tissue_subject):
    """With only CSF perturbed, the first-order index must put the variance on CSF."""
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=256, tissue_cov={2: 0.0, 3: 0.3}, seed=6),
        moments=True,
        record_rois={},
    )
    s = r.tissue_sensitivity("peak")
    assert s[3] > 0.9
    assert abs(s[2]) < 0.05


def test_tissue_sensitivity_requires_recorded_samples(two_tissue_subject):
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=8, seed=6),
        moments=True,
    )
    with pytest.raises(ValueError, match="No per-draw samples"):
        r.tissue_sensitivity("peak")


def test_uq_recycling_path_does_not_change_the_answer(two_tissue_subject):
    """n_samples ≥ 2·RECYCLE_BUILD enables subspace recycling; the seeded guess must not bias it.

    Both runs share a seed, so the first 8 draws are the same conductivities either way — only
    the initial guess differs.
    """
    from cunibs import ConductivityUQConfig

    def run(n):
        return two_tissue_subject.simulate_conductivity_uq(
            _coil(),
            _placement(),
            config=ConductivityUQConfig(n_samples=n, tissue_cov={2: 0.2, 3: 0.3}, seed=9),
            moments=True,
            record_rois={},
        )

    small, large = run(8), run(64)  # 8 < 32 → no recycling; 64 ≥ 32 → recycling
    np.testing.assert_allclose(small.sigma_samples, large.sigma_samples[:8], rtol=0, atol=0)
    np.testing.assert_allclose(small.peak_samples, large.peak_samples[:8], rtol=1e-6)


def test_uq_fp64_fallback_matches(cp, fresh_subject, two_tissue_cube_mesh):
    """An unreachable tolerance sends every draw through the lazily built fp64 AMGX solver."""
    from cunibs import ConductivityUQConfig

    subj = fresh_subject(two_tissue_cube_mesh)
    cfg = ConductivityUQConfig(n_samples=16, tissue_cov={2: 0.2, 3: 0.3}, seed=10)
    kw = {"config": cfg, "moments": True}

    reference = subj.simulate_conductivity_uq(_coil(), _placement(), **kw)
    pre = subj._conductivity_uq_precompute(cfg)
    assert pre.solver is None

    pre.tolerance = 0.0
    pre.max_iters = 5
    fallback = subj.simulate_conductivity_uq(_coil(), _placement(), **kw)
    assert pre.solver is not None

    ref = cp.asnumpy(reference.mean_magnE).astype(np.float64)
    got = cp.asnumpy(fallback.mean_magnE).astype(np.float64)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) <= 1e-5


def test_uq_result_save_load(tmp_path, two_tissue_cube):
    from cunibs import ConductivityUQConfig, Subject
    from cunibs.uq.conductivity import ConductivityUQResult

    subj = Subject(two_tissue_cube)
    r = subj.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=16, seed=3),
        moments=True,
    ).to_numpy()
    path = tmp_path / "uq.h5"
    r.save(path)
    loaded = ConductivityUQResult.load(path)
    np.testing.assert_array_equal(loaded.mean_magnE, r.mean_magnE)
    np.testing.assert_array_equal(loaded.cov_magnE, r.cov_magnE)
    assert loaded.n_samples == r.n_samples
    assert loaded.perturbed_tags == r.perturbed_tags
