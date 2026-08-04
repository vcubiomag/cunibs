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

    real = uq_assembly.fill_stiffness_values
    calls = {"n": 0}

    def flaky(a, g, vols, cond, tet_nodes, ptr, idx):
        calls["n"] += 1
        real(a, g, vols, cond, tet_nodes, ptr, idx)
        if calls["n"] == 1:  # the first per-tissue component
            a.data *= 1.05

    monkeypatch.setattr(uq_assembly, "fill_stiffness_values", flaky)
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
        d70_coil, patch_placement, config=cfg, moments=True
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
        assert item.max_local_cov() >= 0.0


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


def test_per_draw_arrays_are_always_recorded(two_tissue_subject):
    """No flag gates them: a plain run still yields the whole-field per-draw metrics."""
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=16, tissue_cov={2: 0.2, 3: 0.3}, seed=4),
        moments=True,
    )
    assert r.roi_samples == {}
    assert r.peak_samples.shape == (16,)
    assert r.focality_samples.shape == (16,)
    assert r.peak_location_samples.shape == (16, 3)


def test_record_rois_adds_named_probes(two_tissue_subject):
    """record_rois only adds ROI columns; the whole-field arrays are there regardless."""
    from cunibs import ConductivityUQConfig

    subj = two_tissue_subject
    r = subj.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=8, tissue_cov={2: 0.2, 3: 0.3}, seed=4),
        moments=True,
        record_rois={"centre": subj.roi([50.0, 50.0, 60.0], radius_mm=8.0)},
    )
    assert set(r.roi_samples) == {"centre"}
    assert r.roi_samples["centre"].shape == (8,)
    assert r.peak_samples.shape == (8,)


def test_tissue_sensitivity_identifies_the_varying_tissue(two_tissue_subject):
    """With only CSF perturbed, the first-order index must put the variance on CSF."""
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=256, tissue_cov={2: 0.0, 3: 0.3}, seed=6),
        moments=True,
    )
    s = r.tissue_sensitivity("peak")
    assert s[3] > 0.9
    assert abs(s[2]) < 0.05


def test_tissue_sensitivity_rejects_an_unrecorded_roi(two_tissue_subject):
    from cunibs import ConductivityUQConfig

    r = two_tissue_subject.simulate_conductivity_uq(
        _coil(),
        _placement(),
        config=ConductivityUQConfig(n_samples=8, seed=6),
        moments=True,
    )
    r.tissue_sensitivity("peak")  # always available
    with pytest.raises(ValueError, match="Unknown output 'm1'"):
        r.tissue_sensitivity("m1")


@pytest.mark.parametrize(("small_n", "large_n"), [(8, 64), (32, 200)])
def test_uq_draws_do_not_depend_on_the_ensemble_size(two_tissue_subject, small_n, large_n):
    """A draw's field must be the same however many draws were asked for, to the last bit.

    Both runs share a seed, so the shorter one's draws are the same conductivities either way and
    only the initial guess could differ. It must not: the guess is built from the mesh, the
    placement and the nominal conductivities, and applied per draw from that draw's own σ, so
    ``n_samples`` reaches it nowhere.

    Bitwise, not ``allclose``. Anything that reaches x0 from the ensemble moves a draw by about
    the solve tolerance, 6e-07 on sub-004, which an ``rtol=1e-6`` would wave through.
    """
    from cunibs import ConductivityUQConfig

    def run(n):
        return two_tissue_subject.simulate_conductivity_uq(
            _coil(),
            _placement(),
            config=ConductivityUQConfig(n_samples=n, tissue_cov={2: 0.2, 3: 0.3}, seed=9),
            moments=True,
        )

    small, large = run(small_n), run(large_n)
    np.testing.assert_array_equal(small.sigma_samples, large.sigma_samples[:small_n])
    np.testing.assert_array_equal(small.peak_samples, large.peak_samples[:small_n])


def test_uq_sensitivity_basis_predicts_a_small_perturbation(cp, two_tissue_subject):
    """The projection basis must actually be dx/dsigma at nominal sigma.

    Nothing downstream would notice a sign slip or a mismatched tissue index: a wrong subspace
    only makes the initial guess worse, and every draw still converges to the right answer, just
    slower. This pins the derivation against the thing it claims to be — for a small step in one
    tissue's conductivity, ``x(sigma_nom + d) ≈ x_nom + d · dx/dsigma_t``, with the error second
    order in d.
    """
    import cupyx.scipy.sparse as csp

    from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform
    from cunibs.uq.conductivity.assembly import build_conductivity_uq_precompute
    from cunibs.uq.conductivity.run import _dadt_node_to_elm, _placement_rhs, _sensitivity_basis

    ctx = two_tissue_subject.context
    pre = build_conductivity_uq_precompute(ctx, (2, 3))
    n_pert = len(pre.perturbed_tags)
    n_red = int(pre.idx.shape[0])
    stream = cp.cuda.get_current_stream().ptr

    # The real per-tissue RHS decomposition, so both terms of b_t - A_t x_nom are exercised.
    placement = _placement()
    transform = compute_coil_transform(
        ctx, placement.center_mm, placement.handle_mm, placement.distance_mm
    )
    coil = _coil()
    dadt_elm = _dadt_node_to_elm(
        coil_dadt_at_nodes(coil.positions_m, coil.moments, transform, 1e6, ctx.nodes_mm),
        ctx.tet_nodes,
    )
    b_base, b_tissue = _placement_rhs(ctx, pre, dadt_elm)

    def solve_at(sigma):
        b = cp.ascontiguousarray(
            (b_base + sigma.astype(cp.float32) @ b_tissue)[pre.idx], dtype=cp.float64
        )
        pre.pcg.update_values(cp.ascontiguousarray(pre.combine(sigma)), stream)
        x = cp.empty(n_red, dtype=cp.float64)
        pre.pcg.solve_mixed(pre.precond, b, x, 1e-11, pre.max_iters, stream)
        return x

    x_nom = solve_at(pre.nominal_sigma)
    ax_tissue = cp.stack(
        [
            csp.csr_matrix((data, pre.indices, pre.indptr), shape=(n_red, n_red)) @ x_nom
            for data in pre.tissue_data
        ]
    )

    w, einv = _sensitivity_basis(pre, b_tissue, ax_tissue, stream)
    assert w.shape == (n_red, n_pert), "one basis direction per perturbed tissue"
    assert einv.shape == (n_pert, n_pert)
    # Orthonormal columns, so the frozen Galerkin operator is well conditioned by construction.
    np.testing.assert_allclose(cp.asnumpy(w.T @ w), np.eye(n_pert), atol=1e-10)

    # Step one tissue and check the move lands in the span the basis claims. A sign slip or a
    # transposed tissue index leaves it pointing elsewhere and this fails outright.
    for t in range(n_pert):
        sigma = pre.nominal_sigma.copy()
        sigma[t] *= 1.01
        delta = solve_at(sigma) - x_nom
        moved = float(cp.linalg.norm(delta))
        out_of_span = float(cp.linalg.norm(delta - w @ (w.T @ delta)))
        assert moved > 0, f"tissue {pre.perturbed_tags[t]} had no effect on the solution"
        assert out_of_span <= 0.02 * moved, (
            f"tissue {pre.perturbed_tags[t]}: {out_of_span:.3e} of a {moved:.3e} move fell "
            f"outside the sensitivity basis"
        )


def test_uq_unreachable_tolerance_rebuilds_then_raises(fresh_subject, two_tissue_cube_mesh):
    """An unreachable tolerance makes each draw build a matched preconditioner, then fail loudly."""
    from cunibs import ConductivityUQConfig
    from cunibs.fem import SolverConvergenceError

    subj = fresh_subject(two_tissue_cube_mesh)
    cfg = ConductivityUQConfig(n_samples=16, tissue_cov={2: 0.2, 3: 0.3}, seed=10)
    kw = {"config": cfg, "moments": True}

    subj.simulate_conductivity_uq(_coil(), _placement(), **kw)
    pre = subj._conductivity_uq_precompute(cfg)

    pre.tolerance = 0.0
    pre.max_iters = 5
    with pytest.raises(SolverConvergenceError):
        subj.simulate_conductivity_uq(_coil(), _placement(), **kw)


def test_uq_draw_preconditioner_solves_its_own_sample(cp, fresh_subject, two_tissue_cube_mesh):
    """``preconditioner_for`` must produce a working hierarchy for an off-nominal draw.

    This is the retry path's payload, and it is never exercised by a converging ensemble.
    """
    from cunibs import ConductivityUQConfig

    subj = fresh_subject(two_tissue_cube_mesh)
    cfg = ConductivityUQConfig(n_samples=4, tissue_cov={2: 0.2, 3: 0.3}, seed=10)
    subj.simulate_conductivity_uq(_coil(), _placement(), config=cfg, moments=True)
    pre = subj._conductivity_uq_precompute(cfg)

    extreme = cp.ascontiguousarray(pre.nominal_sigma * 5.0)
    sample_data = cp.ascontiguousarray(pre.combine(extreme))
    precond = pre.preconditioner_for(sample_data)

    n_red = int(pre.idx.shape[0])
    rng = cp.random.default_rng(0)
    b = cp.ascontiguousarray(rng.standard_normal(n_red, dtype=cp.float64))
    x = cp.empty(n_red, dtype=cp.float64)
    pre.pcg.update_values(sample_data, cp.cuda.get_current_stream().ptr)
    _, rel = pre.pcg.solve_mixed(
        precond, b, x, pre.tolerance, pre.max_iters, cp.cuda.get_current_stream().ptr
    )
    assert rel <= pre.tolerance


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
