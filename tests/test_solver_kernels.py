"""Direct unit tests for the native ``cunibs.solver`` kernels.

Every kernel here is otherwise reached only through a full solve, where a wrong answer
only shows up if it is large enough to move a field norm. These call each binding on small
hand-built inputs and compare against a NumPy transcription of the kernel's own documented
formula, so an indexing or scaling error is caught at the kernel rather than at the pipeline.

The block variants are additionally required to agree column-for-column with their
single-RHS counterparts, and the node-centric reductions to be bit-reproducible (they are
atomic-free with a fixed summation order, which is what makes the solver deterministic).
"""

from __future__ import annotations

import itertools

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


# --- dadt_node_to_element ----------------------------------------------------------------


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


def test_rhs_assemble_staged_matches_numpy(cp, dev, mesh_arrays, stream):
    from cunibs.solver import rhs_assemble_staged

    b = cp.empty(N_NODES, dtype=cp.float32)
    rhs_assemble_staged(
        dev["dadt_elm"], dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], b, stream
    )
    np.testing.assert_allclose(cp.asnumpy(b), rhs_reference(mesh_arrays), rtol=1e-4)


def test_rhs_assemble_staged_equals_fused(cp, dev, stream):
    """Both spellings must produce the same RHS; only their memory traffic differs."""
    from cunibs.solver import rhs_assemble, rhs_assemble_staged

    fused = cp.empty(N_NODES, dtype=cp.float32)
    staged = cp.empty(N_NODES, dtype=cp.float32)
    rhs_assemble(
        dev["dadt_elm"], dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], fused, stream
    )
    rhs_assemble_staged(
        dev["dadt_elm"], dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], staged, stream
    )
    np.testing.assert_allclose(cp.asnumpy(fused), cp.asnumpy(staged), rtol=1e-5)


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_rhs_assemble_staged_block_matches_single(cp, dev, stream, k):
    """Every column of the block RHS must equal the single-RHS kernel on that column."""
    from cunibs.solver import rhs_assemble_staged, rhs_assemble_staged_block

    rng = np.random.default_rng(2)
    columns = [
        cp.ascontiguousarray(cp.asarray(rng.standard_normal((N_TET, 3)).astype(np.float32)))
        for _ in range(k)
    ]

    b_block = cp.empty((N_NODES, k), dtype=cp.float32)
    rhs_assemble_staged_block(
        columns, dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], b_block, stream
    )

    for c, dadt in enumerate(columns):
        single = cp.empty(N_NODES, dtype=cp.float32)
        rhs_assemble_staged(
            dadt, dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], single, stream
        )
        np.testing.assert_array_equal(
            cp.asnumpy(b_block[:, c]), cp.asnumpy(single), err_msg=f"column {c}"
        )


def test_rhs_assembly_is_bit_reproducible(cp, dev, stream):
    """Node-centric, atomic-free, fixed summation order — repeated calls must be identical."""
    from cunibs.solver import rhs_assemble_staged

    runs = []
    for _ in range(3):
        b = cp.empty(N_NODES, dtype=cp.float32)
        rhs_assemble_staged(
            dev["dadt_elm"], dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], b, stream
        )
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
        degenerate = cp.empty(len(sel), dtype=cp.int32)
        place_transforms(
            cp.ascontiguousarray(cp.asarray(centers[sel])),
            cp.ascontiguousarray(cp.asarray(handles[sel])),
            cp.ascontiguousarray(cp.asarray(dists[sel])),
            ctx.skin_a,
            ctx.skin_b,
            ctx.skin_c,
            cp.ascontiguousarray(ctx.skin_tri_normals, dtype=cp.float64),
            out,
            degenerate,
            stream,
        )
        assert not cp.asnumpy(degenerate).any()
        return cp.asnumpy(out).reshape(-1, 4, 4)

    everything = run(np.arange(4))
    for i in range(4):
        np.testing.assert_array_equal(run(np.array([i]))[0], everything[i], err_msg=f"site {i}")


# --- native SIZE_4 aggregation ------------------------------------------------------------


@pytest.mark.realmesh
def test_native_aggregation_is_a_valid_partition(cp, patch_subject):
    """Every level must map fine rows onto [0, n_coarse) with no empty aggregate.

    An empty aggregate would give P a zero column, making the coarse Galerkin operator
    singular and the dense coarse inverse meaningless.
    """
    from cunibs.fem.solve import aggregation_levels

    solver = patch_subject.context.solver
    values_f32 = cp.ascontiguousarray(solver.values.astype(cp.float32))

    levels, _ = aggregation_levels(solver.row_ptr, solver.col_idx, values_f32)
    assert levels, "the patch should coarsen at least once"

    for i, level in enumerate(levels):
        agg = cp.asnumpy(level.aggregates)
        assert agg.min() >= 0, f"level {i} has a negative aggregate id"
        assert agg.max() < level.n_coarse, f"level {i} exceeds the declared coarse size"
        assert np.unique(agg).size == level.n_coarse, f"level {i} has an empty aggregate"


@pytest.mark.realmesh
def test_stiffness_assembly_is_reproducible(cp, patch_mesh):
    """Assembling the same mesh repeatedly must give byte-identical values.

    The aggregation below is only deterministic given deterministic values to aggregate, and
    the hierarchy it selects can turn on a tie. Summing each entry's element contributions in
    a fixed per-row order is what makes the matrix itself a function of the mesh.
    """
    from cunibs.fem.assembly import (
        assemble_stiffness,
        conductivity_per_tet,
        gradient_operator,
    )

    nodes_mm = cp.asarray(patch_mesh.nodes_mm, dtype=cp.float64)
    tet_nodes = cp.asarray(patch_mesh.tet_nodes)
    cond = conductivity_per_tet(cp.asarray(patch_mesh.tet_tags))
    g, vols = gradient_operator(nodes_mm, tet_nodes)

    first = assemble_stiffness(g, vols, cond, patch_mesh.n_nodes, tet_nodes)
    for attempt in range(1, 4):
        again = assemble_stiffness(g, vols, cond, patch_mesh.n_nodes, tet_nodes)
        np.testing.assert_array_equal(
            cp.asnumpy(again.indptr), cp.asnumpy(first.indptr), err_msg=f"indptr {attempt}"
        )
        np.testing.assert_array_equal(
            cp.asnumpy(again.indices), cp.asnumpy(first.indices), err_msg=f"indices {attempt}"
        )
        np.testing.assert_array_equal(
            cp.asnumpy(again.data), cp.asnumpy(first.data), err_msg=f"data {attempt}"
        )


def _csr_rows(cp, rows):
    """Build a device CSR from a list of ``(column, value)`` lists, one per row."""
    indptr = np.cumsum([0, *(len(r) for r in rows)], dtype=np.int32)
    flat = [entry for row in rows for entry in row]
    indices = np.array([c for c, _ in flat], dtype=np.int32)
    data = np.array([v for _, v in flat], dtype=np.float32)
    return cp.asarray(indptr), cp.asarray(indices), cp.asarray(data)


def test_l1_dinv_matches_the_dense_formula(cp):
    """1 / (sign(aᵢᵢ) · Σⱼ|aᵢⱼ|), with every branch of the kernel exercised once.

    The rows, in order: a positive diagonal, a negative one (which flips the sign of the whole
    row sum), a row summing to zero, an empty row, and a row with no stored diagonal at all.
    """
    from cunibs.solver import l1_dinv

    indptr, indices, data = _csr_rows(
        cp,
        [
            [(0, 2.0), (1, -1.0), (3, 0.5)],
            [(0, -1.0), (1, -4.0), (2, 1.0)],
            [(2, 0.0)],
            [],
            [(0, 3.0), (2, -1.0)],
        ],
    )
    dinv = cp.empty(5, dtype=cp.float32)
    l1_dinv(indptr, indices, data, dinv, cp.cuda.get_current_stream().ptr)

    expected = np.array([1.0 / 3.5, -1.0 / 6.0, 1.0, 1.0, 0.25])
    np.testing.assert_allclose(cp.asnumpy(dinv), expected, rtol=1e-6)


def test_l1_dinv_is_bit_reproducible(cp):
    """Fixed summation order, so repeated calls agree in the last bit.

    The rows are long enough for the order to matter: a sparse product over rows of this width
    splits them across a thread count the library picks at runtime.
    """
    from cunibs.solver import l1_dinv

    rng = np.random.default_rng(3)
    n, per_row = 64, 300
    dense = rng.standard_normal((n, per_row)).astype(np.float32)
    indptr, indices, data = _csr_rows(cp, [list(enumerate(row.tolist())) for row in dense])

    dinv = cp.empty(n, dtype=cp.float32)
    l1_dinv(indptr, indices, data, dinv, cp.cuda.get_current_stream().ptr)
    first = cp.asnumpy(dinv).copy()

    d = np.abs(dense).sum(axis=1, dtype=np.float64) * np.sign(dense.diagonal())
    np.testing.assert_allclose(first, 1.0 / d, rtol=1e-5)

    for attempt in range(1, 4):
        again = cp.empty(n, dtype=cp.float32)
        l1_dinv(indptr, indices, data, again, cp.cuda.get_current_stream().ptr)
        np.testing.assert_array_equal(cp.asnumpy(again), first, err_msg=f"call {attempt}")


def test_l1_dinv_rejects_a_wrong_length_output(cp):
    """A short ``dinv`` would otherwise be written past its end on the device."""
    from cunibs.solver import l1_dinv

    indptr, indices, data = _csr_rows(cp, [[(0, 1.0)], [(1, 1.0)]])
    with pytest.raises(ValueError, match="one entry per row"):
        l1_dinv(indptr, indices, data, cp.empty(1, dtype=cp.float32), 0)


@pytest.mark.realmesh
def test_native_aggregation_is_deterministic(cp, patch_subject):
    """One thread per row with fixed tie-breaks and no atomics: byte-identical across runs.

    The operators are checked alongside the aggregate maps because they are what the next
    level aggregates on: a prolongator or Galerkin value that moves in its last bits can flip
    a near-tie in ``select_size4``'s strongest-neighbour fold and change the hierarchy shape.
    """
    from cunibs.fem.solve import aggregation_levels

    solver = patch_subject.context.solver
    values_f32 = cp.ascontiguousarray(solver.values.astype(cp.float32))

    first, first_coarse = aggregation_levels(solver.row_ptr, solver.col_idx, values_f32)
    second, second_coarse = aggregation_levels(solver.row_ptr, solver.col_idx, values_f32)

    assert len(first) == len(second)
    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        assert a.n_coarse == b.n_coarse, f"level {i} aggregate count is not reproducible"
        for name, lhs, rhs in (
            ("aggregates", a.aggregates, b.aggregates),
            ("p.data", a.p.data, b.p.data),
            ("p.indices", a.p.indices, b.p.indices),
            ("a.data", a.a.data, b.a.data),
        ):
            np.testing.assert_array_equal(
                cp.asnumpy(lhs),
                cp.asnumpy(rhs),
                err_msg=f"level {i} {name} is not reproducible",
            )
    np.testing.assert_array_equal(
        cp.asnumpy(first_coarse.data),
        cp.asnumpy(second_coarse.data),
        err_msg="the coarsest operator is not reproducible",
    )


@pytest.mark.realmesh
def test_native_hierarchy_level_sizes(cp, patch_subject):
    """Pin the hierarchy shape on the committed patch fixture.

    Catches an accidental change to the matching rules: resetting the carried-over
    ``strongest``/``wsn`` state, for instance, drops the per-pass coarsening ratio and adds a
    level, which this notices immediately.

    The ratio is ~18 rather than a single pairwise pass's ~4.3 because a level composes two of
    them (see ``_aggregate``).
    """
    from cunibs.fem.solve import aggregation_levels

    solver = patch_subject.context.solver
    values_f32 = cp.ascontiguousarray(solver.values.astype(cp.float32))

    levels, _ = aggregation_levels(solver.row_ptr, solver.col_idx, values_f32)
    sizes = [int(solver.idx.shape[0])] + [level.n_coarse for level in levels]
    assert sizes == [8403, 465], f"hierarchy shape changed: {sizes}"

    for level, (fine, coarse) in enumerate(itertools.pairwise(sizes)):
        ratio = fine / coarse
        assert 14.0 <= ratio <= 22.0, f"level {level} coarsening ratio {ratio:.2f} out of range"


# --- PcgAmgSolver argument validation ------------------------------------------------------


def test_update_values_rejects_a_wrong_nonzero_count(cp, cube_subject):
    """A short ``values`` would otherwise be read past its end on the device."""
    pcg = cube_subject.context.solver.pcg
    short = cp.ascontiguousarray(cube_subject.context.solver.values[:-1])
    with pytest.raises(ValueError, match="one entry per nonzero"):
        pcg.update_values(short, cp.cuda.get_current_stream().ptr)


def test_solve_mixed_rejects_a_wrong_row_count(cp, cube_subject):
    solver = cube_subject.context.solver
    n = int(solver.row_ptr.shape[0]) - 1
    stream = cp.cuda.get_current_stream().ptr
    b = cp.zeros(n - 1, dtype=cp.float64)
    x = cp.empty(n - 1, dtype=cp.float64)
    with pytest.raises(ValueError, match="one entry per row"):
        solver.pcg.solve_mixed(solver.precond, b, x, 1e-6, 10, stream)


def test_solve_mixed_block_rejects_mismatched_operands(cp, cube_subject):
    """``k`` is taken from ``B``, so an ``X`` of a different width has to be rejected."""
    solver = cube_subject.context.solver
    n = int(solver.row_ptr.shape[0]) - 1
    stream = cp.cuda.get_current_stream().ptr
    B = cp.ascontiguousarray(cp.zeros((n, 4), dtype=cp.float64))
    X = cp.ascontiguousarray(cp.empty((n, 2), dtype=cp.float64))
    with pytest.raises(ValueError, match=r"\(n, k\)"):
        solver.pcg.solve_mixed_block(solver.precond, B, X, 1e-6, 10, stream)


def test_exported_block_widths_are_the_ones_the_solver_accepts(cp, cube_subject):
    """BLOCK_SIZES is the extension's own list; check it against dispatch_k's real behaviour.

    The C++ side spells the widths twice: once as ``kBlockWidths`` (which this exports) and
    once as ``dispatch_k`` template arguments in block_cg.cu and vcycle.cu. Adding a width to
    only one of them would otherwise surface as a runtime error on a user's mesh.
    """
    from cunibs.solver import BLOCK_SIZES, MAX_STAGE_BLOCK

    solver = cube_subject.context.solver
    n = int(solver.row_ptr.shape[0]) - 1
    stream = cp.cuda.get_current_stream().ptr
    accepted = []
    for k in range(1, MAX_STAGE_BLOCK + 1):
        B = cp.ascontiguousarray(cp.ones((n, k), dtype=cp.float64))
        X = cp.ascontiguousarray(cp.empty((n, k), dtype=cp.float64))
        try:
            solver.pcg.solve_mixed_block(solver.precond, B, X, 1e-6, 1, stream)
        except ValueError as exc:
            if "supports k in" in str(exc):
                continue
            raise
        accepted.append(k)
    assert tuple(accepted) == tuple(BLOCK_SIZES)


def test_pattern_handles_a_segment_too_wide_for_the_shared_block(cp):
    """A high-valence node forces the one-warp-per-block path, which no head mesh reaches.

    The segment CSR builders sort each segment in shared memory, reserving the widest segment's
    padded width per warp. Past what four warps can hold they fall to a warp per block; a head mesh
    stays two orders of magnitude under that, so only a fixture like this covers the branch.
    """
    from cunibs.fem.assembly import build_node2corner, stiffness_pattern

    # A fan of tetrahedra all sharing node 0, so its segment is four candidates per tet.
    n_fan = 900
    angle = np.linspace(0.0, 2.0 * np.pi, n_fan + 1)[:-1]
    ring = np.stack([np.cos(angle), np.sin(angle), np.zeros(n_fan)], axis=1)
    nodes = np.vstack([[0.0, 0.0, 0.0], ring, [0.0, 0.0, 1.0]])
    apex = len(nodes) - 1
    tets = np.array(
        [[0, apex, 1 + i, 1 + ((i + 1) % n_fan)] for i in range(n_fan)], dtype=np.int32
    )

    d_tets = cp.asarray(tets)
    ptr, idx = build_node2corner(d_tets, len(nodes))
    # Four warps sharing 48 KB get 3072 int32 slots each, and the buffer is padded to a power of
    # two, so this fixture's widest segment is over the line.
    widest = int(cp.diff(ptr).max()) * 4
    assert 1 << (widest - 1).bit_length() > 3072

    a = stiffness_pattern(d_tets, len(nodes), ptr, idx)
    expected = [np.unique(tets[np.any(tets == n, axis=1)]) for n in range(len(nodes))]
    np.testing.assert_array_equal(
        cp.asnumpy(a.indptr), np.concatenate([[0], np.cumsum([len(e) for e in expected])])
    )
    np.testing.assert_array_equal(cp.asnumpy(a.indices), np.concatenate(expected))


def test_tet_lowest_node_matches_a_numpy_gather(cp):
    """The Morton sort key: the smallest permuted node index over a tetrahedron's four nodes."""
    from cunibs.solver import tet_lowest_node

    rng = np.random.default_rng(3)
    inverse = rng.permutation(N_NODES).astype(np.int32)
    tets = rng.integers(0, N_NODES, size=(N_TET, 4), dtype=np.int32)

    lowest = cp.empty(N_TET, dtype=cp.int32)
    tet_lowest_node(
        cp.asarray(inverse), cp.asarray(tets), lowest, cp.cuda.get_current_stream().ptr
    )
    np.testing.assert_array_equal(cp.asnumpy(lowest), inverse[tets].min(axis=1))
