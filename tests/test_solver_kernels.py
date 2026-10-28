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

    # P1 basis gradients sum to zero over an element's corners, and reconstruct relies on it.
    g = rng.standard_normal((N_TET, 4, 3))
    g -= g.mean(axis=1, keepdims=True)
    return {
        "tet_nodes": tet_nodes,
        "ptr": ptr,
        "idx": idx,
        "g": g.astype(np.float32),
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
    """The per-element assembly the UQ draws use, which folds neg_vc in itself."""
    from cunibs.solver import rhs_assemble

    b = cp.empty(N_NODES, dtype=cp.float32)
    rhs_assemble(dev["dadt_elm"], dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], b, stream)
    np.testing.assert_allclose(cp.asnumpy(b), rhs_reference(mesh_arrays), rtol=1e-4)


def _rhs_operator(cp, dev, mesh_arrays):
    """The forward RHS operator over the fixture mesh, built the way build_context builds it."""
    from cunibs.fem.assembly import stiffness_pattern
    from cunibs.fem.solve import build_rhs_operator

    a = stiffness_pattern(dev["tet_nodes"], N_NODES, dev["ptr"], dev["idx"])
    op = build_rhs_operator(
        a,
        cp.asarray(mesh_arrays["g"].astype(np.float64)),
        cp.asarray(mesh_arrays["neg_vc"].astype(np.float64)),
        dev["tet_nodes"],
        dev["ptr"],
        dev["idx"],
    )
    return a.indptr, op


def test_rhs_operator_matches_the_element_assembly(cp, dev, mesh_arrays, stream):
    """B a must equal the per-element assembly fed the element average of the same nodal field."""
    from cunibs.solver import dadt_node_to_element, rhs_assemble, rhs_operator_apply

    indptr, op = _rhs_operator(cp, dev, mesh_arrays)
    rng = np.random.default_rng(4)
    nodal = cp.asarray(rng.standard_normal((N_NODES, 3)).astype(np.float32))

    b = cp.empty((N_NODES, 1), dtype=cp.float32)
    rhs_operator_apply(indptr, op, nodal.reshape(N_NODES, 1, 3), b, stream)

    elm = cp.empty((N_TET, 3), dtype=cp.float32)
    dadt_node_to_element(nodal, dev["tet_nodes"], elm, stream)
    by_element = cp.empty(N_NODES, dtype=cp.float32)
    rhs_assemble(elm, dev["g"], dev["neg_vc"], dev["ptr"], dev["idx"], by_element, stream)
    np.testing.assert_allclose(
        cp.asnumpy(b[:, 0]), cp.asnumpy(by_element), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_rhs_operator_block_matches_single(cp, dev, mesh_arrays, stream, k):
    """Each column of a k-wide apply must equal the k = 1 apply on that column, bit for bit."""
    from cunibs.solver import rhs_operator_apply

    indptr, op = _rhs_operator(cp, dev, mesh_arrays)
    rng = np.random.default_rng(5)
    block = cp.asarray(rng.standard_normal((N_NODES, k, 3)).astype(np.float32))
    b_block = cp.empty((N_NODES, k), dtype=cp.float32)
    rhs_operator_apply(indptr, op, block, b_block, stream)
    for c in range(k):
        single = cp.empty((N_NODES, 1), dtype=cp.float32)
        column = cp.ascontiguousarray(block[:, c : c + 1, :])
        rhs_operator_apply(indptr, op, column, single, stream)
        np.testing.assert_array_equal(
            cp.asnumpy(b_block[:, c]), cp.asnumpy(single[:, 0]), err_msg=f"column {c}"
        )


def test_rhs_assembly_is_bit_reproducible(cp, dev, stream):
    """Node-centric, atomic-free, fixed summation order — repeated calls must be identical."""
    from cunibs.solver import rhs_assemble

    runs = []
    for _ in range(3):
        b = cp.empty(N_NODES, dtype=cp.float32)
        rhs_assemble(
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
@pytest.mark.parametrize("shared", [False, True])
def test_reconstruct_e_nodal_block_matches_averaging_first(cp, dev, stream, k, shared):
    """Averaging the nodal dA/dt in the kernel gives the bits of averaging it onto the elements
    first, column by column, for an interleaved block and for one field every column shares."""
    from cunibs.solver import dadt_node_to_element, reconstruct_e, reconstruct_e_nodal_block

    rng = np.random.default_rng(3)
    v_block = cp.ascontiguousarray(cp.asarray(rng.standard_normal((N_NODES, k))))
    width = 1 if shared else k
    nodal = cp.asarray(rng.standard_normal((N_NODES, width, 3)).astype(np.float32))
    es = [cp.empty((N_TET, 3), dtype=cp.float32) for _ in range(k)]
    magns = [cp.empty(N_TET, dtype=cp.float32) for _ in range(k)]
    reconstruct_e_nodal_block(v_block, dev["tet_nodes"], dev["g"], nodal, es, magns, stream)

    for c in range(k):
        dadt_elm = cp.empty((N_TET, 3), dtype=cp.float32)
        column = cp.ascontiguousarray(nodal[:, 0 if shared else c, :])
        dadt_node_to_element(column, dev["tet_nodes"], dadt_elm, stream)
        e_ref = cp.empty((N_TET, 3), dtype=cp.float32)
        magn_ref = cp.empty(N_TET, dtype=cp.float32)
        reconstruct_e(
            cp.ascontiguousarray(v_block[:, c]),
            dev["tet_nodes"],
            dev["g"],
            dadt_elm,
            e_ref,
            magn_ref,
            stream,
        )
        np.testing.assert_array_equal(cp.asnumpy(es[c]), cp.asnumpy(e_ref), err_msg=f"col {c}")
        np.testing.assert_array_equal(
            cp.asnumpy(magns[c]), cp.asnumpy(magn_ref), err_msg=f"col {c}"
        )


def test_reconstruct_e_nodal_block_skips_an_output_without_moving_the_other(cp, dev, stream):
    from cunibs.solver import reconstruct_e_nodal_block

    k = 4
    rng = np.random.default_rng(4)
    v_block = cp.ascontiguousarray(cp.asarray(rng.standard_normal((N_NODES, k))))
    nodal = cp.asarray(rng.standard_normal((N_NODES, k, 3)).astype(np.float32))
    es = [cp.empty((N_TET, 3), dtype=cp.float32) for _ in range(k)]
    magns = [cp.empty(N_TET, dtype=cp.float32) for _ in range(k)]
    reconstruct_e_nodal_block(v_block, dev["tet_nodes"], dev["g"], nodal, es, magns, stream)

    magn_only = [cp.full(N_TET, cp.nan, dtype=cp.float32) for _ in range(k)]
    reconstruct_e_nodal_block(v_block, dev["tet_nodes"], dev["g"], nodal, [], magn_only, stream)
    e_only = [cp.full((N_TET, 3), cp.nan, dtype=cp.float32) for _ in range(k)]
    reconstruct_e_nodal_block(v_block, dev["tet_nodes"], dev["g"], nodal, e_only, [], stream)
    for c in range(k):
        np.testing.assert_array_equal(cp.asnumpy(magn_only[c]), cp.asnumpy(magns[c]))
        np.testing.assert_array_equal(cp.asnumpy(e_only[c]), cp.asnumpy(es[c]))


# --- ADM reciprocity weights -------------------------------------------------------------


def _p1_grad_reference(v, tet_nodes, g):
    """Σᵢ v_i g_i taken as Σ_{i≥1} (v_i − v_0) g_i, which P1 gradients summing to zero allow.

    The kernels use this form, and comparing against the direct sum would charge them the
    |v| |g| ε that the fp32 g's rounding leaves in it.
    """
    vt = v[tet_nodes]
    return np.einsum("ei,eik->ek", vt[:, 1:] - vt[:, :1], g[:, 1:].astype(np.float64))


def _node_weight(cp, dev, stream, values, roi, alpha, direction):
    from cunibs.solver import node_weight

    node_w = cp.empty((N_NODES, 3), dtype=cp.float64)
    node_weight(
        values,
        dev["tet_nodes"],
        dev["g"],
        dev["neg_vc"],
        dev["ptr"],
        dev["idx"],
        cp.asarray(roi, dtype=cp.int32),
        cp.asarray(alpha, dtype=cp.float64),
        list(direction),
        node_w,
        stream,
    )
    return cp.asnumpy(node_w)


ROI = np.array([2, 5, 6, 11, 30], dtype=np.int32)


def _node_weight_reference(mesh_arrays, values, roi, alpha, direction):
    """¼ Σ_{e∋n} w_e with w_e = (−neg_vc[e]) · Σᵢ values[tet_nodes[e,i]]·g[e,i,:] − α_e ê."""
    grad = _p1_grad_reference(values, mesh_arrays["tet_nodes"], mesh_arrays["g"])
    w_e = -mesh_arrays["neg_vc"].astype(np.float64)[:, None] * grad
    w_e[roi] -= np.asarray(alpha)[:, None] * np.asarray(direction)[None, :]
    expected = np.zeros((N_NODES, 3))
    np.add.at(expected, mesh_arrays["tet_nodes"].ravel(), np.repeat(w_e, 4, axis=0))
    return 0.25 * expected


def test_node_weight_matches_numpy(cp, dev, mesh_arrays, stream):
    alpha = np.linspace(0.1, 0.5, ROI.size)
    direction = (0.6, -0.8, 0.0)
    got = _node_weight(cp, dev, stream, dev["v"], ROI, alpha, direction)
    expected = _node_weight_reference(mesh_arrays, mesh_arrays["v"], ROI, alpha, direction)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_node_weight_without_an_roi_is_the_gathered_element_weight(
    cp, dev, mesh_arrays, stream
):
    got = _node_weight(cp, dev, stream, dev["v"], [], [], (1.0, 0.0, 0.0))
    expected = _node_weight_reference(mesh_arrays, mesh_arrays["v"], [], [], (1.0, 0.0, 0.0))
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_node_weight_of_a_constant_is_exactly_zero(cp, dev, stream):
    """A constant potential has no gradient, and the differenced form returns exactly that."""
    values = cp.full(N_NODES, 3.25, dtype=cp.float64)
    assert not np.any(_node_weight(cp, dev, stream, values, [], [], (0.0, 0.0, 1.0)))


def test_node_weight_counts_each_incident_roi_corner(cp, dev, mesh_arrays, stream):
    """With λ ≡ 0 and every element in the ROI at α = 1, each node gets −¼ ê per corner."""
    values = cp.zeros(N_NODES, dtype=cp.float64)
    got = _node_weight(
        cp, dev, stream, values, np.arange(N_TET), np.ones(N_TET), (0.0, 2.0, 0.0)
    )
    counts = np.bincount(mesh_arrays["tet_nodes"].ravel(), minlength=N_NODES)
    np.testing.assert_array_equal(got[:, 1], -0.5 * counts)
    assert not np.any(got[:, [0, 2]])


def test_node_weight_is_linear_in_values(cp, dev, stream):
    a = _node_weight(cp, dev, stream, dev["v"], [], [], (1.0, 0.0, 0.0))
    b = _node_weight(
        cp, dev, stream, cp.ascontiguousarray(dev["v"] * 3.0), [], [], (1.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(b, 3.0 * a, rtol=1e-12)


# --- streaming moments -------------------------------------------------------------------


def test_accumulate_moments_adds_to_the_running_sums(cp, stream):
    """sum_e += |E|, sumsq_e += |E|² — it accumulates, it does not overwrite."""
    from cunibs.solver import accumulate_moments_block

    rng = np.random.default_rng(5)
    draws = rng.random((6, N_TET)).astype(np.float32) + 0.1
    sum_e = cp.zeros(N_TET, dtype=cp.float64)
    sumsq_e = cp.zeros(N_TET, dtype=cp.float64)
    for magn in draws:
        accumulate_moments_block([cp.asarray(magn)], sum_e, sumsq_e, stream)

    ref = draws.astype(np.float64)
    np.testing.assert_allclose(cp.asnumpy(sum_e), ref.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(sumsq_e), (ref**2).sum(axis=0), rtol=1e-12)


def test_accumulate_moments_respects_a_nonzero_starting_value(cp, stream):
    from cunibs.solver import accumulate_moments_block

    magn = cp.full(N_TET, 2.0, dtype=cp.float32)
    sum_e = cp.full(N_TET, 10.0, dtype=cp.float64)
    sumsq_e = cp.full(N_TET, 100.0, dtype=cp.float64)
    accumulate_moments_block([magn], sum_e, sumsq_e, stream)
    np.testing.assert_allclose(cp.asnumpy(sum_e), 12.0)
    np.testing.assert_allclose(cp.asnumpy(sumsq_e), 104.0)


def test_accumulate_moments_recovers_mean_and_variance(cp, stream):
    """The moments must reproduce the ensemble mean and variance the UQ path derives."""
    from cunibs.solver import accumulate_moments_block

    rng = np.random.default_rng(6)
    draws = rng.random((64, N_TET)).astype(np.float32) + 0.5
    sum_e = cp.zeros(N_TET, dtype=cp.float64)
    sumsq_e = cp.zeros(N_TET, dtype=cp.float64)
    for magn in draws:
        accumulate_moments_block([cp.asarray(magn)], sum_e, sumsq_e, stream)

    n = draws.shape[0]
    mean = cp.asnumpy(sum_e) / n
    var = cp.asnumpy(sumsq_e) / n - mean**2
    ref = draws.astype(np.float64)
    np.testing.assert_allclose(mean, ref.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(var, ref.var(axis=0), rtol=1e-6, atol=1e-12)


# --- N-body and placement, called directly ------------------------------------------------


@pytest.mark.parametrize("planar", [False, True])
def test_dadt_block_matches_analytic_dipoles(cp, stream, planar):
    """The raw kernel against the closed form, for every placement of a block.

    Moments sharing one direction take the four-accumulator path, so both paths are checked.
    """
    from cunibs.fem.placement import MU0_OVER_4PI
    from cunibs.solver import dadt_block

    rng = np.random.default_rng(7)
    s = rng.uniform(-0.05, 0.05, size=(16, 3))
    if planar:
        m = rng.standard_normal((16, 1)) * np.array([0.3, -0.4, 0.866])
    else:
        m = rng.standard_normal((16, 3))
    r_mm = rng.uniform(100.0, 200.0, size=(40, 3))  # kept well away from the sources
    didt = 1e6
    angles = (0.0, 0.7, 2.1)
    tf = np.zeros((len(angles), 4, 4))
    for i, a in enumerate(angles):
        tf[i, :3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        tf[i, :3, 3] = [5.0 * i, -3.0, 2.0]
        tf[i, 3, 3] = 1.0

    out = cp.empty((40, 3 * len(angles)), dtype=cp.float32)
    dadt_block(
        cp.asarray(s),
        cp.asarray(m),
        cp.asarray(tf),
        cp.asarray(r_mm),
        out,
        didt,
        MU0_OVER_4PI,
        stream,
    )
    got = cp.asnumpy(out).reshape(40, len(angles), 3)
    for i in range(len(angles)):
        si = s @ tf[i, :3, :3].T + tf[i, :3, 3] * 1e-3
        mi = m @ tf[i, :3, :3].T
        diff = r_mm[:, None, :] * 1e-3 - si[None, :, :]
        dist3 = np.linalg.norm(diff, axis=2)[:, :, None] ** 3
        ref = didt * MU0_OVER_4PI * (np.cross(mi[None, :, :], diff) / dist3).sum(axis=1)
        assert np.linalg.norm(got[:, i] - ref) / np.linalg.norm(ref) <= 1e-5


def test_place_transforms_batches_independently(cp, cube_subject, stream):
    """The raw batched kernel: each placement's 4×4 must not depend on its neighbours."""
    from cunibs.solver import place_transforms

    ctx = cube_subject.context
    centers = np.array([[50.0, 50, 120], [20, 80, 120], [80, 20, 120], [50, 50, 130]])
    handles = centers + np.array([0.0, 50.0, 0.0])
    dists = np.array([4.0, 1.0, 8.0, 0.0])

    def run(sel):
        out = cp.empty((len(sel), 4, 4), dtype=cp.float64)
        degenerate = cp.empty(len(sel), dtype=cp.int32)
        place_transforms(
            cp.ascontiguousarray(cp.asarray(centers[sel])),
            cp.ascontiguousarray(cp.asarray(handles[sel])),
            cp.ascontiguousarray(cp.asarray(dists[sel])),
            ctx.skin_verts,
            ctx.skin_tri_normals,
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

    a = patch_subject.context.solver.matrix()
    levels, _ = aggregation_levels(a.indptr, a.indices, a.data.astype(cp.float32))
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

    a = patch_subject.context.solver.matrix()
    values_f32 = a.data.astype(cp.float32)

    first, first_coarse = aggregation_levels(a.indptr, a.indices, values_f32)
    second, second_coarse = aggregation_levels(a.indptr, a.indices, values_f32)

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
    a = solver.matrix()
    levels, _ = aggregation_levels(a.indptr, a.indices, a.data.astype(cp.float32))
    sizes = [int(solver.idx.shape[0])] + [level.n_coarse for level in levels]
    assert sizes == [8403, 465], f"hierarchy shape changed: {sizes}"

    for level, (fine, coarse) in enumerate(itertools.pairwise(sizes)):
        ratio = fine / coarse
        assert 14.0 <= ratio <= 22.0, f"level {level} coarsening ratio {ratio:.2f} out of range"


# --- PcgAmgSolver argument validation ------------------------------------------------------


def test_update_values_rejects_a_wrong_nonzero_count(cp, cube_subject):
    """A short ``values`` would otherwise be read past its end on the device."""
    pcg = cube_subject.context.solver.pcg
    short = cp.ascontiguousarray(cube_subject.context.solver.matrix().data[:-1])
    with pytest.raises(ValueError, match="one entry per nonzero"):
        pcg.update_values(short, cp.cuda.get_current_stream().ptr)


def test_solve_mixed_rejects_a_wrong_row_count(cp, cube_subject):
    solver = cube_subject.context.solver
    n = solver.pcg.n
    stream = cp.cuda.get_current_stream().ptr
    b = cp.zeros(n - 1, dtype=cp.float64)
    x = cp.empty(n - 1, dtype=cp.float64)
    with pytest.raises(ValueError, match="one entry per row"):
        solver.pcg.solve_mixed(solver.precond, b, x, 1e-6, 10, stream)


def test_solve_mixed_block_rejects_mismatched_operands(cp, cube_subject):
    """``k`` is taken from ``B``, so an ``X`` of a different width has to be rejected."""
    solver = cube_subject.context.solver
    n = solver.pcg.n
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
    n = solver.pcg.n
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


def _fan_mesh(n_fan, *, extra_tets=0):
    """A centre node joined to a ring of ``n_fan`` triangles, so its valence is ``n_fan``.

    This is how an analytic sphere is built -- ``metrics/_analytic.ball_mesh`` fills the core with
    exactly this fan -- which is why the shape has to work rather than merely not crash. The
    centre's segment is four candidates per incident tetrahedron.

    ``extra_tets`` appends disjoint unit tetrahedra whose nodes have ordinary valence, so one mesh
    carries segments from both width classes.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, n_fan + 1)[:-1]
    ring = np.stack([np.cos(angle), np.sin(angle), np.zeros(n_fan)], axis=1)
    nodes = np.vstack([[0.0, 0.0, 0.0], ring, [0.0, 0.0, 1.0]])
    apex = len(nodes) - 1
    tets = [[0, apex, 1 + i, 1 + ((i + 1) % n_fan)] for i in range(n_fan)]

    unit = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for e in range(extra_tets):
        base = len(nodes)
        nodes = np.vstack([nodes, unit + np.array([10.0 + 2.0 * e, 0.0, 0.0])])
        tets.append([base, base + 1, base + 2, base + 3])
    return nodes, np.array(tets, dtype=np.int32)


def _expected_pattern(n_nodes, tets):
    """The same CSR by a different route: every (corner, corner) pair of every tetrahedron.

    Written without a per-node scan of the mesh so the widest fixtures below stay quick -- the
    obvious ``tets[np.any(tets == n)]`` loop is quadratic and dominates their runtime.
    """
    owner = np.repeat(tets, 4, axis=1).reshape(-1)
    other = np.tile(tets, (1, 4)).reshape(-1)
    pairs = np.unique(owner.astype(np.int64) * n_nodes + other)
    rows, cols = np.divmod(pairs, n_nodes)
    counts = np.bincount(rows, minlength=n_nodes)
    return np.concatenate([[0], np.cumsum(counts)]), cols.astype(np.int32)


def _assert_pattern_matches_numpy(cp, nodes, tets):
    """Build the stiffness pattern on the device and return the mesh's widest segment."""
    from cunibs.fem.assembly import build_node2corner, stiffness_pattern

    d_tets = cp.asarray(tets)
    ptr, idx = build_node2corner(d_tets, len(nodes))
    a = stiffness_pattern(d_tets, len(nodes), ptr, idx)
    want_ptr, want_idx = _expected_pattern(len(nodes), tets)
    np.testing.assert_array_equal(cp.asnumpy(a.indptr), want_ptr)
    np.testing.assert_array_equal(cp.asnumpy(a.indices), want_idx)
    return int(cp.diff(ptr).max()) * 4


# kFastPad in pattern.cu: the widest padded segment four warps hold in 48 KB, and so the boundary
# between the shared-memory class and the global-memory one.
_FAST_PAD = 2048

# Fan sizes on both sides of that boundary. 400 sorts in shared memory at 1600 candidates; 20480 is
# the centre node of ball_mesh(5), the finest rung of the analytic sphere ladder, at 81920.
_FAN_SIZES = [400, 900, 20480]


def test_fan_sizes_straddle_the_segment_class_boundary():
    """The widths below only cover both classes as long as they sit either side of kFastPad."""
    assert 4 * min(_FAN_SIZES) <= _FAST_PAD < 4 * max(_FAN_SIZES)


@pytest.mark.parametrize("n_fan", _FAN_SIZES)
def test_pattern_builds_segments_of_any_width(cp, n_fan):
    """Node valence is not capped: a segment of any width builds the row a numpy scan gives."""
    nodes, tets = _fan_mesh(n_fan)
    assert _assert_pattern_matches_numpy(cp, nodes, tets) == 4 * n_fan


def test_pattern_wide_segment_does_not_widen_the_narrow_ones(cp):
    """A mesh carrying both classes: the wide segment must not disturb the ordinary rows.

    The two classes sort by different primitives, so a mesh whose widest node is far past the
    boundary is what shows the narrow rows still come out right beside it.
    """
    nodes, tets = _fan_mesh(6000, extra_tets=64)
    _assert_pattern_matches_numpy(cp, nodes, tets)


def test_patch_csr_builds_a_wide_segment(cp):
    """The patch gather reaches the same sort through a different gather, so it needs its own
    segment past the shared-memory class."""
    from cunibs.solver import count_patch_csr, fill_patch_csr

    # One slot whose first ring is wide enough to land in the global class, plus narrow ones.
    wide, narrow = 5000, 8
    r1_ptr = np.array([0, wide, wide + narrow, wide + 2 * narrow], dtype=np.int32)
    rng = np.random.default_rng(0)
    r1_idx = np.concatenate(
        [
            rng.permutation(wide).astype(np.int32),
            rng.integers(0, wide, narrow).astype(np.int32),
            rng.integers(0, wide, narrow).astype(np.int32),
        ]
    )
    # Each entry's neighbour slot; only consulted for rings under min_nodes, which is 0 here.
    neighbour = np.zeros(r1_idx.shape[0], dtype=np.int32)

    d_ptr, d_idx = cp.asarray(r1_ptr), cp.asarray(r1_idx)
    d_nb = cp.asarray(neighbour)
    stream = cp.cuda.get_current_stream().ptr
    out_ptr = cp.empty(len(r1_ptr), dtype=cp.int32)
    nnz = count_patch_csr(d_ptr, d_idx, d_nb, 0, out_ptr, stream)
    out_idx = cp.empty(nnz, dtype=cp.int32)
    fill_patch_csr(d_ptr, d_idx, d_nb, 0, out_ptr, out_idx, stream)

    expected = [np.unique(r1_idx[r1_ptr[s] : r1_ptr[s + 1]]) for s in range(len(r1_ptr) - 1)]
    np.testing.assert_array_equal(
        cp.asnumpy(out_ptr), np.concatenate([[0], np.cumsum([len(e) for e in expected])])
    )
    np.testing.assert_array_equal(cp.asnumpy(out_idx), np.concatenate(expected))


def _smoothed_skin_normals(nodes_mm, tris, *, rounds=2):
    """NumPy transcription of skin.cu.

    Area-weighted face normals are summed at each node and gathered back per face, ``rounds``
    times; the node vectors are then normalised and each triangle takes its three, normalised.
    """
    local, faces = np.unique(tris, return_inverse=True)
    faces = faces.reshape(-1, 3)
    coords = nodes_mm[local]

    side_a = coords[faces[:, 1]] - coords[faces[:, 0]]
    side_b = coords[faces[:, 2]] - coords[faces[:, 0]]
    normals = np.cross(side_a, side_b)

    node_vec = np.zeros((local.shape[0], 3))
    for _ in range(rounds):
        for i in range(3):
            node_vec[:, i] = np.bincount(
                faces.reshape(-1), np.repeat(normals[:, i], 3), local.shape[0]
            )
        normals = np.sum(node_vec[faces], axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
    node_vec /= np.linalg.norm(node_vec, axis=1)[:, None]

    tri = np.sum(node_vec[faces], axis=1)
    return tri / np.linalg.norm(tri, axis=1)[:, None]


@pytest.mark.realmesh
def test_skin_normals_match_a_numpy_transcription(patch_mesh, patch_skin_normals):
    """The whole smoothing chain against a transcription of it, on a real surface.

    Not bitwise: the kernel sums each node's faces in node2corner order and ``bincount`` sums them
    in triangle order, which lands a third of the components one ulp apart. The tolerance is a few
    ulp, tight enough that a wrong corner, weighting, or round count is nowhere near it -- one
    smoothing round instead of two moves this surface by more than 1e-3.
    """
    expected = _smoothed_skin_normals(patch_mesh.nodes_mm, patch_mesh.skin_tris)
    np.testing.assert_allclose(patch_skin_normals, expected, rtol=1e-13, atol=1e-15)

    one_round = _smoothed_skin_normals(patch_mesh.nodes_mm, patch_mesh.skin_tris, rounds=1)
    assert np.abs(one_round - expected).max() > 1e-3


def test_skin_normals_on_an_empty_surface(cp):
    """A mesh with no skin triangles returns an empty (0, 3) rather than launching anything."""
    from cunibs.fem.solve import skin_triangle_normals

    out = skin_triangle_normals(cp.zeros((4, 3), dtype=cp.float64), cp.empty((0, 3), cp.int32))
    assert out.shape == (0, 3)


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
