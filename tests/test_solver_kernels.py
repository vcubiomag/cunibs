"""Direct unit tests for the native ``cunibs.solver`` kernels.

Every kernel here was previously reached only through a full solve, where a wrong answer
only shows up if it is large enough to move a field norm. These call each binding on small
hand-built inputs and compare against a NumPy transcription of the kernel's own documented
formula, so an indexing or scaling error is caught at the kernel rather than at the pipeline.

The block variants are additionally required to agree column-for-column with their
single-RHS counterparts, and the node-centric reductions to be bit-reproducible (they are
atomic-free with a fixed summation order, which is what makes the solver deterministic).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

N_TET = 37
N_NODES = 23


@pytest.fixture(scope="module")
def mesh_arrays():
    """Random connectivity and per-element data, plus the matching node2corner map."""
    rng = np.random.default_rng(0)
    tet_nodes = rng.integers(0, N_NODES, size=(N_TET, 4), dtype=np.int32)
    # Every node must own at least one corner, or ptr/idx has empty segments that hide bugs.
    tet_nodes[: N_NODES // 4 * 4].reshape(-1)[:N_NODES] = np.arange(N_NODES, dtype=np.int32)

    flat = tet_nodes.ravel()
    idx = np.argsort(flat, kind="stable").astype(np.int32)  # corner ids c = 4e + i
    ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=N_NODES))]).astype(
        np.int32
    )

    return {
        "tet_nodes": tet_nodes,
        "ptr": ptr,
        "idx": idx,
        "g": rng.standard_normal((N_TET, 4, 3)).astype(np.float32),
        "neg_vc": (-rng.random(N_TET) - 0.1).astype(np.float32),
        "dadt_elm": rng.standard_normal((N_TET, 3)).astype(np.float32),
        "v": rng.standard_normal(N_NODES),
    }


@pytest.fixture
def dev(cp, mesh_arrays):
    return {k: cp.ascontiguousarray(cp.asarray(v)) for k, v in mesh_arrays.items()}


@pytest.fixture
def stream(cp):
    return cp.cuda.get_current_stream().ptr


# --- weighted_gradient / dadt_node_to_element --------------------------------------------


def test_weighted_gradient_scales_each_corner_by_its_element(cp, dev, mesh_arrays, stream):
    """wg[e,i,k] = g[e,i,k] · neg_vc[e] — the flat kernel indexes neg_vc[i/12]."""
    from cunibs.solver import weighted_gradient

    wg = cp.empty_like(dev["g"])
    weighted_gradient(dev["g"], dev["neg_vc"], wg, stream)
    expected = mesh_arrays["g"] * mesh_arrays["neg_vc"][:, None, None]
    np.testing.assert_allclose(cp.asnumpy(wg), expected, rtol=1e-6)


def test_dadt_node_to_element_averages_the_four_corners(cp, dev, mesh_arrays, stream):
    """dadt_elm[e] = ¼ Σᵢ dadt_nodes[tet_nodes[e,i]]."""
    from cunibs.solver import dadt_node_to_element

    rng = np.random.default_rng(1)
    dadt_nodes = rng.standard_normal((N_NODES, 3)).astype(np.float32)
    out = cp.empty((N_TET, 3), dtype=cp.float32)
    dadt_node_to_element(cp.asarray(dadt_nodes), dev["tet_nodes"], out, stream)

    expected = dadt_nodes[mesh_arrays["tet_nodes"]].mean(axis=1)
    np.testing.assert_allclose(cp.asnumpy(out), expected, rtol=1e-6)


def test_dadt_node_to_element_is_exact_on_a_constant_field(cp, dev, stream):
    """A spatially constant dA/dt must average to itself on every element."""
    from cunibs.solver import dadt_node_to_element

    dadt_nodes = cp.tile(cp.asarray([[1.5, -2.5, 3.5]], dtype=cp.float32), (N_NODES, 1))
    out = cp.empty((N_TET, 3), dtype=cp.float32)
    dadt_node_to_element(cp.ascontiguousarray(dadt_nodes), dev["tet_nodes"], out, stream)
    np.testing.assert_array_equal(cp.asnumpy(out), np.tile([1.5, -2.5, 3.5], (N_TET, 1)))


# --- RHS assembly ------------------------------------------------------------------------


def rhs_reference(mesh_arrays, wg=None):
    """b[node] = Σ_{corners c of node} dot(dadt_elm[c>>2], wg[c]), with wg = g·neg_vc."""
    tet_nodes = mesh_arrays["tet_nodes"]
    corner_g = (
        mesh_arrays["g"] * mesh_arrays["neg_vc"][:, None, None] if wg is None else wg
    ).reshape(-1, 3)
    dadt_per_corner = np.repeat(mesh_arrays["dadt_elm"], 4, axis=0)
    q = (dadt_per_corner.astype(np.float64) * corner_g.astype(np.float64)).sum(axis=1)
    b = np.zeros(N_NODES)
    np.add.at(b, tet_nodes.ravel(), q)
    return b


def test_rhs_assemble_matches_numpy(cp, dev, mesh_arrays, stream):
    """The unweighted variant folds neg_vc in itself; nothing else in the suite calls it."""
    from cunibs.solver import rhs_assemble

    b = cp.empty(N_NODES, dtype=cp.float32)
    rhs_assemble(dev["dadt_elm"], dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], b, stream)
    np.testing.assert_allclose(cp.asnumpy(b), rhs_reference(mesh_arrays), rtol=1e-4)


def test_rhs_assemble_weighted_matches_numpy(cp, dev, mesh_arrays, stream):
    from cunibs.solver import rhs_assemble_weighted, weighted_gradient

    wg = cp.empty_like(dev["g"])
    weighted_gradient(dev["g"], dev["neg_vc"], wg, stream)

    b = cp.empty(N_NODES, dtype=cp.float32)
    rhs_assemble_weighted(dev["dadt_elm"], wg, dev["ptr"], dev["idx"], b, stream)
    np.testing.assert_allclose(cp.asnumpy(b), rhs_reference(mesh_arrays), rtol=1e-4)


def test_rhs_assemble_weighted_equals_unweighted(cp, dev, stream):
    """Both spellings must produce the same RHS; only their memory traffic differs."""
    from cunibs.solver import rhs_assemble, rhs_assemble_weighted, weighted_gradient

    wg = cp.empty_like(dev["g"])
    weighted_gradient(dev["g"], dev["neg_vc"], wg, stream)

    plain = cp.empty(N_NODES, dtype=cp.float32)
    weighted = cp.empty(N_NODES, dtype=cp.float32)
    rhs_assemble(
        dev["dadt_elm"], dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], plain, stream
    )
    rhs_assemble_weighted(dev["dadt_elm"], wg, dev["ptr"], dev["idx"], weighted, stream)
    np.testing.assert_allclose(cp.asnumpy(plain), cp.asnumpy(weighted), rtol=1e-5)


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_rhs_assemble_weighted_block_matches_single(cp, dev, stream, k):
    """Every column of the block RHS must equal the single-RHS kernel on that column."""
    from cunibs.solver import (
        rhs_assemble_weighted,
        rhs_assemble_weighted_block,
        weighted_gradient,
    )

    wg = cp.empty_like(dev["g"])
    weighted_gradient(dev["g"], dev["neg_vc"], wg, stream)

    rng = np.random.default_rng(2)
    columns = [
        cp.ascontiguousarray(cp.asarray(rng.standard_normal((N_TET, 3)).astype(np.float32)))
        for _ in range(k)
    ]

    b_block = cp.empty((N_NODES, k), dtype=cp.float32)
    rhs_assemble_weighted_block(columns, wg, dev["ptr"], dev["idx"], b_block, stream)

    for c, dadt in enumerate(columns):
        single = cp.empty(N_NODES, dtype=cp.float32)
        rhs_assemble_weighted(dadt, wg, dev["ptr"], dev["idx"], single, stream)
        np.testing.assert_array_equal(
            cp.asnumpy(b_block[:, c]), cp.asnumpy(single), err_msg=f"column {c}"
        )


def test_rhs_assembly_is_bit_reproducible(cp, dev, stream):
    """Node-centric, atomic-free, fixed summation order — repeated calls must be identical."""
    from cunibs.solver import rhs_assemble_weighted, weighted_gradient

    wg = cp.empty_like(dev["g"])
    weighted_gradient(dev["g"], dev["neg_vc"], wg, stream)

    runs = []
    for _ in range(3):
        b = cp.empty(N_NODES, dtype=cp.float32)
        rhs_assemble_weighted(dev["dadt_elm"], wg, dev["ptr"], dev["idx"], b, stream)
        runs.append(cp.asnumpy(b))
    np.testing.assert_array_equal(runs[0], runs[1])
    np.testing.assert_array_equal(runs[0], runs[2])


# --- E reconstruction --------------------------------------------------------------------


def reconstruct_reference(mesh_arrays, v=None, dadt_elm=None):
    """E[e] = −Σᵢ v[tet_nodes[e,i]]·g[e,i] − dadt_elm[e]."""
    v = mesh_arrays["v"] if v is None else v
    dadt_elm = mesh_arrays["dadt_elm"] if dadt_elm is None else dadt_elm
    grad_v = np.einsum(
        "ei,eik->ek", v[mesh_arrays["tet_nodes"]], mesh_arrays["g"].astype(float)
    )
    e = -grad_v - dadt_elm.astype(np.float64)
    return e, np.linalg.norm(e, axis=1)


def test_reconstruct_e_matches_numpy(cp, dev, mesh_arrays, stream):
    from cunibs.solver import reconstruct_e

    e_out = cp.empty((N_TET, 3), dtype=cp.float32)
    magn_out = cp.empty(N_TET, dtype=cp.float32)
    reconstruct_e(
        dev["v"], dev["tet_nodes"], dev["g"], dev["dadt_elm"], e_out, magn_out, stream
    )

    e_ref, magn_ref = reconstruct_reference(mesh_arrays)
    np.testing.assert_allclose(cp.asnumpy(e_out), e_ref, rtol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(magn_out), magn_ref, rtol=1e-5)


def test_reconstruct_e_magnitude_is_the_norm_of_e(cp, dev, stream):
    """magn_out must be ‖e_out‖ computed from the same fp64 accumulation, not a re-norm."""
    from cunibs.solver import reconstruct_e

    e_out = cp.empty((N_TET, 3), dtype=cp.float32)
    magn_out = cp.empty(N_TET, dtype=cp.float32)
    reconstruct_e(
        dev["v"], dev["tet_nodes"], dev["g"], dev["dadt_elm"], e_out, magn_out, stream
    )
    np.testing.assert_allclose(
        cp.asnumpy(magn_out),
        np.linalg.norm(cp.asnumpy(e_out).astype(np.float64), axis=1),
        rtol=1e-6,
    )


def test_reconstruct_e_zero_potential_is_minus_dadt(cp, dev, mesh_arrays, stream):
    """With v ≡ 0 the field reduces to −dA/dt exactly."""
    from cunibs.solver import reconstruct_e

    e_out = cp.empty((N_TET, 3), dtype=cp.float32)
    magn_out = cp.empty(N_TET, dtype=cp.float32)
    reconstruct_e(
        cp.zeros(N_NODES), dev["tet_nodes"], dev["g"], dev["dadt_elm"], e_out, magn_out, stream
    )
    np.testing.assert_array_equal(cp.asnumpy(e_out), -mesh_arrays["dadt_elm"])


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_reconstruct_e_block_matches_single(cp, dev, stream, k):
    from cunibs.solver import reconstruct_e, reconstruct_e_block

    rng = np.random.default_rng(3)
    v_block = cp.ascontiguousarray(cp.asarray(rng.standard_normal((N_NODES, k))))
    dadts = [
        cp.ascontiguousarray(cp.asarray(rng.standard_normal((N_TET, 3)).astype(np.float32)))
        for _ in range(k)
    ]
    es = [cp.empty((N_TET, 3), dtype=cp.float32) for _ in range(k)]
    magns = [cp.empty(N_TET, dtype=cp.float32) for _ in range(k)]
    reconstruct_e_block(v_block, dev["tet_nodes"], dev["g"], dadts, es, magns, stream)

    for c in range(k):
        e_ref = cp.empty((N_TET, 3), dtype=cp.float32)
        magn_ref = cp.empty(N_TET, dtype=cp.float32)
        reconstruct_e(
            cp.ascontiguousarray(v_block[:, c]),
            dev["tet_nodes"],
            dev["g"],
            dadts[c],
            e_ref,
            magn_ref,
            stream,
        )
        np.testing.assert_array_equal(cp.asnumpy(es[c]), cp.asnumpy(e_ref), err_msg=f"col {c}")
        np.testing.assert_array_equal(
            cp.asnumpy(magns[c]), cp.asnumpy(magn_ref), err_msg=f"col {c}"
        )


# --- ADM reciprocity weights -------------------------------------------------------------


def test_element_weight_matches_numpy(cp, dev, mesh_arrays, stream):
    """w_e[e,k] = (−neg_vc[e]) · Σᵢ values[tet_nodes[e,i]]·g[e,i,k] — i.e. vol·σ·(Gλ)."""
    from cunibs.solver import element_weight

    w_e = cp.empty((N_TET, 3), dtype=cp.float64)
    element_weight(dev["v"], dev["tet_nodes"], dev["g"], dev["neg_vc"], w_e, stream)

    grad = np.einsum(
        "ei,eik->ek", mesh_arrays["v"][mesh_arrays["tet_nodes"]], mesh_arrays["g"].astype(float)
    )
    expected = -mesh_arrays["neg_vc"].astype(np.float64)[:, None] * grad
    np.testing.assert_allclose(cp.asnumpy(w_e), expected, rtol=1e-9)


def test_element_weight_sign_is_positive_for_positive_conductivity(cp, dev, stream):
    """neg_vc is −vol·σ, so the kernel's leading −neg_vc must come out positive."""
    from cunibs.solver import element_weight

    ones = cp.ones(N_NODES, dtype=cp.float64)
    w_e = cp.empty((N_TET, 3), dtype=cp.float64)
    element_weight(ones, dev["tet_nodes"], dev["g"], dev["neg_vc"], w_e, stream)
    # Σᵢ g[e,i,:] is the P1 partition-of-unity gradient, which is zero for a real element;
    # here g is random, so just check the scaling factor is applied with the right sign.
    grad = cp.asnumpy(dev["g"]).astype(np.float64).sum(axis=1)
    scale = -cp.asnumpy(dev["neg_vc"]).astype(np.float64)
    assert np.all(scale > 0)
    np.testing.assert_allclose(cp.asnumpy(w_e), scale[:, None] * grad, rtol=1e-9)


def test_element_weight_is_linear_in_values(cp, dev, stream):
    from cunibs.solver import element_weight

    def run(values):
        out = cp.empty((N_TET, 3), dtype=cp.float64)
        element_weight(values, dev["tet_nodes"], dev["g"], dev["neg_vc"], out, stream)
        return cp.asnumpy(out)

    a = run(dev["v"])
    b = run(cp.ascontiguousarray(dev["v"] * 3.0))
    np.testing.assert_allclose(b, 3.0 * a, rtol=1e-12)


def test_node_scatter3_matches_numpy(cp, dev, mesh_arrays, stream):
    """node_w[n,k] = ¼ Σ_{corners c of n} w_e[c>>2, k]."""
    from cunibs.solver import node_scatter3

    rng = np.random.default_rng(4)
    w_e = rng.standard_normal((N_TET, 3))
    node_w = cp.empty((N_NODES, 3), dtype=cp.float64)
    node_scatter3(cp.asarray(w_e), dev["ptr"], dev["idx"], node_w, stream)

    expected = np.zeros((N_NODES, 3))
    np.add.at(expected, mesh_arrays["tet_nodes"].ravel(), np.repeat(w_e, 4, axis=0))
    np.testing.assert_allclose(cp.asnumpy(node_w), 0.25 * expected, rtol=1e-12)


def test_node_scatter3_counts_each_incident_corner(cp, dev, mesh_arrays, stream):
    """With w_e ≡ 1 each node gets ¼ × (number of corners it owns)."""
    from cunibs.solver import node_scatter3

    node_w = cp.empty((N_NODES, 3), dtype=cp.float64)
    node_scatter3(cp.ones((N_TET, 3), dtype=cp.float64), dev["ptr"], dev["idx"], node_w, stream)
    counts = np.bincount(mesh_arrays["tet_nodes"].ravel(), minlength=N_NODES)
    np.testing.assert_allclose(
        cp.asnumpy(node_w), 0.25 * np.tile(counts[:, None], 3), rtol=1e-12
    )


# --- streaming moments -------------------------------------------------------------------


def test_accumulate_moments_adds_to_the_running_sums(cp, stream):
    """sum_e += |E|, sumsq_e += |E|² — it accumulates, it does not overwrite."""
    from cunibs.solver import accumulate_moments

    rng = np.random.default_rng(5)
    draws = rng.random((6, N_TET)).astype(np.float32) + 0.1
    sum_e = cp.zeros(N_TET, dtype=cp.float64)
    sumsq_e = cp.zeros(N_TET, dtype=cp.float64)
    for magn in draws:
        accumulate_moments(cp.asarray(magn), sum_e, sumsq_e, stream)

    ref = draws.astype(np.float64)
    np.testing.assert_allclose(cp.asnumpy(sum_e), ref.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(sumsq_e), (ref**2).sum(axis=0), rtol=1e-12)


def test_accumulate_moments_respects_a_nonzero_starting_value(cp, stream):
    from cunibs.solver import accumulate_moments

    magn = cp.full(N_TET, 2.0, dtype=cp.float32)
    sum_e = cp.full(N_TET, 10.0, dtype=cp.float64)
    sumsq_e = cp.full(N_TET, 100.0, dtype=cp.float64)
    accumulate_moments(magn, sum_e, sumsq_e, stream)
    np.testing.assert_allclose(cp.asnumpy(sum_e), 12.0)
    np.testing.assert_allclose(cp.asnumpy(sumsq_e), 104.0)


def test_accumulate_moments_recovers_mean_and_variance(cp, stream):
    """The moments must reproduce the ensemble mean and variance the UQ path derives."""
    from cunibs.solver import accumulate_moments

    rng = np.random.default_rng(6)
    draws = rng.random((64, N_TET)).astype(np.float32) + 0.5
    sum_e = cp.zeros(N_TET, dtype=cp.float64)
    sumsq_e = cp.zeros(N_TET, dtype=cp.float64)
    for magn in draws:
        accumulate_moments(cp.asarray(magn), sum_e, sumsq_e, stream)

    n = draws.shape[0]
    mean = cp.asnumpy(sum_e) / n
    var = cp.asnumpy(sumsq_e) / n - mean**2
    ref = draws.astype(np.float64)
    np.testing.assert_allclose(mean, ref.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(var, ref.var(axis=0), rtol=1e-6, atol=1e-12)


# --- N-body and placement, called directly ------------------------------------------------


def test_dadt_nbody_matches_analytic_dipoles(cp, stream):
    """The raw kernel, with the (m | m×s) packing its caller builds, against the closed form."""
    from cunibs.fem.placement import MU0_OVER_4PI
    from cunibs.solver import dadt_nbody

    rng = np.random.default_rng(7)
    s = rng.uniform(-0.05, 0.05, size=(16, 3))
    m = rng.standard_normal((16, 3))
    r = rng.uniform(0.1, 0.2, size=(40, 3))  # kept well away from the sources
    didt = 1e6

    mp = np.concatenate([m, np.cross(m, s)], axis=1).astype(np.float32)
    out = cp.empty((40, 3), dtype=cp.float32)
    dadt_nbody(
        cp.ascontiguousarray(cp.asarray(s.astype(np.float32))),
        cp.ascontiguousarray(cp.asarray(mp)),
        cp.ascontiguousarray(cp.asarray((s.astype(np.float32) ** 2).sum(1))),
        cp.ascontiguousarray(cp.asarray(r.astype(np.float32))),
        out,
        didt,
        MU0_OVER_4PI,
        stream,
    )

    diff = r[:, None, :] - s[None, :, :]
    dist3 = np.linalg.norm(diff, axis=2)[:, :, None] ** 3
    ref = didt * MU0_OVER_4PI * (np.cross(m[None, :, :], diff) / dist3).sum(axis=1)
    assert np.linalg.norm(cp.asnumpy(out) - ref) / np.linalg.norm(ref) <= 1e-5


def test_place_transforms_batches_independently(cp, cube_subject, stream):
    """The raw batched kernel: each placement's 4×4 must not depend on its neighbours."""
    from cunibs.solver import place_transforms

    ctx = cube_subject.context
    centers = np.array([[50.0, 50, 120], [20, 80, 120], [80, 20, 120], [50, 50, 130]])
    handles = centers + np.array([0.0, 50.0, 0.0])
    dists = np.array([4.0, 1.0, 8.0, 0.0])

    def run(sel):
        out = cp.empty((len(sel), 16), dtype=cp.float64)
        place_transforms(
            cp.ascontiguousarray(cp.asarray(centers[sel])),
            cp.ascontiguousarray(cp.asarray(handles[sel])),
            cp.ascontiguousarray(cp.asarray(dists[sel])),
            ctx.skin_a,
            ctx.skin_b,
            ctx.skin_c,
            cp.ascontiguousarray(ctx.skin_tri_normals, dtype=cp.float64),
            out,
            stream,
        )
        return cp.asnumpy(out).reshape(-1, 4, 4)

    everything = run(np.arange(4))
    for i in range(4):
        np.testing.assert_array_equal(run(np.array([i]))[0], everything[i], err_msg=f"site {i}")


# --- AMGX hierarchy export ----------------------------------------------------------------


@pytest.mark.realmesh
def test_amgx_float_solver_exports_a_usable_hierarchy(cp, patch_subject):
    """``build_native_vcycle`` reads these three accessors; nothing else tests them.

    The exported aggregate map must be a total function from fine rows onto the coarse rows
    the next level declares — that surjectivity is what makes the boolean-P reconstruction
    in ``build_native_vcycle`` valid.
    """
    from cunibs.fem.solve import AMGX_PRECONDITIONER_CONFIG
    from cunibs.solver import AMGXFloatSolver

    solver = patch_subject.context.solver
    amgx = AMGXFloatSolver(AMGX_PRECONDITIONER_CONFIG)
    amgx.setup(
        solver.row_ptr, solver.col_idx, cp.ascontiguousarray(solver.values.astype(cp.float32))
    )

    n_levels = amgx.amg_num_levels()
    assert n_levels >= 2, "the patch should coarsen at least once"

    rows, nnz, _ = amgx.amg_level_dims(0)
    assert rows == int(solver.idx.shape[0])
    assert nnz == int(solver.col_idx.shape[0])

    for level in range(n_levels - 1):
        level_rows, level_nnz, n_coarse = amgx.amg_level_dims(level)
        assert level_nnz >= level_rows
        assert 0 < n_coarse < level_rows, f"level {level} did not coarsen"
        # build_native_vcycle chains the levels by feeding n_coarse into the next Galerkin
        # product and guards the mismatch with a RuntimeError; this is that invariant.
        next_rows, _, _ = amgx.amg_level_dims(level + 1)
        assert next_rows == n_coarse

        agg = cp.empty(level_rows, dtype=cp.int32)
        amgx.download_aggregates(level, agg)
        agg_host = cp.asnumpy(agg)
        assert agg_host.min() >= 0 and agg_host.max() < n_coarse
        # Surjective onto the coarse rows: an empty aggregate would give P a zero column and
        # make the boolean-P reconstruction singular.
        assert np.unique(agg_host).size == n_coarse, "an aggregate is empty"
