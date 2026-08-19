from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_conductivity_mapping_and_unknown_tag(cp):
    from cunibs.fem.assembly import conductivity_per_tet

    cond = conductivity_per_tet(cp.asarray([1, 2, 3], dtype=cp.int32))
    np.testing.assert_allclose(cp.asnumpy(cond), [0.126, 0.275, 1.654])
    with pytest.raises(ValueError):
        conductivity_per_tet(cp.asarray([2, 99], dtype=cp.int32))


def test_gradient_operator_reference_tet(cp):
    from cunibs.fem.assembly import gradient_operator

    nodes_mm = cp.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=cp.float64)
    tets = cp.asarray([[0, 1, 2, 3]], dtype=cp.int32)
    g, vols = gradient_operator(nodes_mm, tets)
    # Coordinates are millimetres, the operator is in metre units: a 1 mm edge gives a basis
    # gradient of 1000 /m and the unit tet a volume of (1/6) mm^3.
    assert float(vols[0]) == pytest.approx(1e-9 / 6)
    np.testing.assert_allclose(
        cp.asnumpy(g[0]), [[-1e3, -1e3, -1e3], [1e3, 0, 0], [0, 1e3, 0], [0, 0, 1e3]], atol=1e-9
    )
    # P1 basis gradients sum to zero (partition of unity).
    np.testing.assert_allclose(cp.asnumpy(g[0].sum(0)), [0, 0, 0], atol=1e-9)


@pytest.mark.parametrize(
    "nodes",
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ],
)
def test_build_context_rejects_zero_volume_tetrahedra(nodes):
    from cunibs.fem import build_context
    from cunibs.mesh import HeadMesh

    mesh = HeadMesh(
        np.asarray(nodes),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.empty((0, 3), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="1 tetrahedron has zero or non-finite volume"):
        build_context(mesh)


def test_stiffness_symmetric_zero_rowsum(cp, cube_mesh):
    from cunibs.fem.assembly import (
        assemble_stiffness,
        conductivity_per_tet,
        gradient_operator,
    )

    nodes = cp.asarray(cube_mesh.nodes_mm)
    tets = cp.asarray(cube_mesh.tet_nodes)
    g, vols = gradient_operator(nodes, tets)
    cond = conductivity_per_tet(cp.asarray(cube_mesh.tet_tags))
    a = assemble_stiffness(g, vols, cond, cube_mesh.n_nodes, tets).toarray()
    a = cp.asnumpy(a)
    np.testing.assert_allclose(a, a.T, atol=1e-12)
    np.testing.assert_allclose(a.sum(1), 0.0, atol=1e-10)


def test_node2corner_is_tet_transpose(cp, cube_mesh):
    from cunibs.fem.assembly import build_node2corner

    tets = cp.asarray(cube_mesh.tet_nodes)
    ptr, idx = build_node2corner(tets, cube_mesh.n_nodes)
    ptr, idx = cp.asnumpy(ptr), cp.asnumpy(idx)
    flat = cube_mesh.tet_nodes.ravel()
    for node in range(cube_mesh.n_nodes):
        corners = idx[ptr[node] : ptr[node + 1]]
        assert np.all(flat[corners] == node)
    assert len(idx) == cube_mesh.tet_nodes.size


def test_solve_placement_zero_didt_gives_zero_field(cp, cube_mesh):
    from cunibs.fem import build_context, solve_placement

    ctx = build_context(cube_mesh)
    coil_pos = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
    coil_mom = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    out = solve_placement(ctx, coil_pos, coil_mom, [50, 50, 100], [50, 100, 100], 4.0, 0.0)
    np.testing.assert_allclose(cp.asnumpy(out["magnE"]), 0.0, atol=1e-20)


def test_ground_node_is_lowest_z(cp):
    from cunibs.fem.solve import ground_node_of

    nodes = cp.asarray([[0, 0, 5.0], [0, 0, -2.0], [0, 0, 3.0], [0, 0, -2.0]])
    assert ground_node_of(nodes) == 1  # ties resolve to the lowest index


def test_ground_node_matches_argmin_on_cube(cp, cube_mesh):
    from cunibs.fem.solve import ground_node_of

    assert ground_node_of(cp.asarray(cube_mesh.nodes_mm)) == int(
        np.argmin(cube_mesh.nodes_mm[:, 2])
    )


@pytest.mark.parametrize("ground", [0, 3, 7])
def test_grounded_index_skips_ground(cp, ground):
    from cunibs.fem.solve import grounded_index

    idx = cp.asnumpy(grounded_index(8, ground))
    np.testing.assert_array_equal(idx, np.delete(np.arange(8), ground))
    assert idx.dtype == np.int32


def test_reduce_matrix_drops_row_and_col(cp, cube_mesh):
    from cunibs.fem.assembly import (
        assemble_stiffness,
        conductivity_per_tet,
        gradient_operator,
    )
    from cunibs.fem.solve import ground_node_of, grounded_index, reduce_matrix

    nodes = cp.asarray(cube_mesh.nodes_mm)
    tets = cp.asarray(cube_mesh.tet_nodes)
    g, vols = gradient_operator(nodes, tets)
    cond = conductivity_per_tet(cp.asarray(cube_mesh.tet_tags))
    a = assemble_stiffness(g, vols, cond, cube_mesh.n_nodes, tets)

    ground = ground_node_of(cp.asarray(cube_mesh.nodes_mm))
    a_red = reduce_matrix(a, grounded_index(cube_mesh.n_nodes, ground))

    dense = cp.asnumpy(a.toarray())
    expected = np.delete(np.delete(dense, ground, axis=0), ground, axis=1)
    np.testing.assert_allclose(cp.asnumpy(a_red.toarray()), expected, atol=1e-14)


def test_l1_dinv_matches_dense_reference(cp, cube_subject):
    """The JACOBI_L1 scaling is 1 / (sign(aᵢᵢ) · Σⱼ|aᵢⱼ|), diagonal included.

    Also pins the bound the Chebyshev interval rests on: with the diagonal in the row sum,
    D − A is weakly diagonally dominant with a non-negative diagonal, so ρ(D⁻¹A) ≤ 1.
    """
    import cupyx.scipy.sparse as csp

    from cunibs.fem.solve import _l1_dinv

    solver = cube_subject.context.solver
    n = int(solver.idx.shape[0])
    a32 = csp.csr_matrix(
        (solver.values.astype(cp.float32), solver.col_idx, solver.row_ptr), shape=(n, n)
    )
    dense = cp.asnumpy(a32.toarray()).astype(np.float64)
    d = np.abs(dense).sum(axis=1) * np.sign(np.diag(dense))
    np.testing.assert_allclose(cp.asnumpy(_l1_dinv(a32)), 1.0 / d, rtol=1e-6)

    dinv = cp.asnumpy(_l1_dinv(a32)).astype(np.float64)
    rho = np.max(np.abs(np.linalg.eigvals(dinv[:, None] * dense)))
    assert rho <= 1.0 + 1e-6, f"rho(D^-1 A) = {rho} breaks the smoother interval's upper bound"


def test_solve_grounded_matches_dense_solve(cp, cube_subject):
    """The mixed-precision PCG must reproduce a dense fp64 solve of the reduced system."""
    import cupyx.scipy.sparse as csp

    from cunibs.fem.solve import solve_grounded

    solver = cube_subject.context.solver
    n_red = int(solver.idx.shape[0])
    a_red = csp.csr_matrix(
        (solver.values, solver.col_idx, solver.row_ptr), shape=(n_red, n_red)
    )
    rng = np.random.default_rng(0)
    b = cp.asarray(rng.standard_normal(solver.n))

    v = solve_grounded(solver, b)
    assert float(v[int(np.argmin(cp.asnumpy(cube_subject.mesh.nodes_mm)[:, 2]))]) == 0.0

    x_ref = np.linalg.solve(
        cp.asnumpy(a_red.toarray()), cp.asnumpy(b[solver.idx]).astype(np.float64)
    )
    # The PCG stops at a 1e-6 relative residual, so the solution error is bounded by that
    # times the system's conditioning, not by machine precision.
    got = cp.asnumpy(v[solver.idx])
    assert np.linalg.norm(got - x_ref) / np.linalg.norm(x_ref) <= 1e-6


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_smoother_degree_changes_the_rate_not_the_answer(cp, cube_subject, degree):
    """Every Chebyshev degree must converge, and to the same place.

    The preconditioner is a pure function of A, so it cannot move the fixed point, only the rate.

    This is also the sharpest available check on the recurrence itself. A degree-d sweep is d
    out-of-place writes alternating between the level's two work buffers, so the degree's parity
    decides which buffer the result lands in and which one the next stage reads; every step from
    the second on also writes the buffer it reads x_prev from. Get any of that wrong and the
    cycle silently returns a corrupted correction, which shows up here as a wrong answer or a
    blown iteration count rather than as a crash.
    """
    import cupyx.scipy.sparse as csp

    from cunibs.fem.solve import (
        BLOCK_SIZES,
        SmootherParams,
        _solve_grounded_block_mat,
        build_native_vcycle,
        solve_grounded,
    )

    solver = cube_subject.context.solver
    n_red = int(solver.idx.shape[0])
    a_red = csp.csr_matrix(
        (solver.values, solver.col_idx, solver.row_ptr), shape=(n_red, n_red)
    )
    rng = np.random.default_rng(0)
    b = cp.asarray(rng.standard_normal(solver.n))
    x_ref = np.linalg.solve(
        cp.asnumpy(a_red.toarray()), cp.asnumpy(b[solver.idx]).astype(np.float64)
    )

    solver.precond = build_native_vcycle(
        solver.row_ptr,
        solver.col_idx,
        cp.ascontiguousarray(solver.values.astype(cp.float32)),
        smoother=SmootherParams(degree=degree),
    )
    got = cp.asnumpy(solve_grounded(solver, b)[solver.idx])

    assert np.linalg.norm(got - x_ref) / np.linalg.norm(x_ref) <= 1e-6
    # A broken sweep still converges eventually on a problem this small, so gate the rate too.
    assert solver.last_iterations < solver.max_iters // 4

    # Every compiled block width, not just the scalar path. The aliasing between a step's
    # output and its x_prev is per-column, so a K-wide bug can leave some columns exact and
    # others NaN at one width and one degree only, which the scalar path above cannot see.
    b_red = cp.ascontiguousarray(b[solver.idx], dtype=cp.float64)
    for k in BLOCK_SIZES:
        B = cp.ascontiguousarray(cp.tile(b_red[:, None], (1, k)))
        X, telemetry = _solve_grounded_block_mat(solver, B, k)
        assert bool(cp.isfinite(X).all()), f"block_k={k}, degree={degree}: non-finite solve"
        # Identical columns, so identical counts: a count that varied here would be reading the
        # wrong column's residual.
        assert len({t.iterations for t in telemetry}) == 1, f"block_k={k}, degree={degree}"
        for c in range(k):
            err = np.linalg.norm(cp.asnumpy(X[:, c]) - x_ref) / np.linalg.norm(x_ref)
            assert err <= 1e-6, f"block_k={k}, column {c}, degree={degree}: rel {err:.3e}"


def test_solve_placement_is_linear_in_didt(cp, cube_subject, synthetic_coil):
    from cunibs.fem import solve_placement

    ctx = cube_subject.context
    args = (
        synthetic_coil.positions_m,
        synthetic_coil.moments,
        [50, 50, 100],
        [50, 100, 100],
        4.0,
    )
    one = solve_placement(ctx, *args, 1e6)
    two = solve_placement(ctx, *args, 2e6)
    np.testing.assert_allclose(
        cp.asnumpy(two["E"]), 2.0 * cp.asnumpy(one["E"]), rtol=1e-5, atol=0
    )


def test_solve_placement_far_coil_gives_negligible_field(cp, cube_subject, synthetic_coil):
    """A dipole field falls off as r⁻³, so a coil 10 m away must contribute nothing."""
    from cunibs.fem import solve_placement

    ctx = cube_subject.context
    args = (synthetic_coil.positions_m, synthetic_coil.moments, [50, 50, 100], [50, 100, 100])
    near = float(cp.abs(solve_placement(ctx, *args, 4.0, 1e6)["E"]).max())
    far = float(cp.abs(solve_placement(ctx, *args, 1e4, 1e6)["E"]).max())
    assert far < 1e-6 * near


def test_build_context_dtypes_and_contiguity(cp, cube_subject):
    """The placement kernels take fp32 C-contiguous device arrays; the bindings enforce it."""
    from cunibs.fem.assembly import conductivity_per_tet

    ctx = cube_subject.context
    n_tet = int(ctx.tet_nodes.shape[0])
    assert ctx.g.dtype == cp.float32 and ctx.g.shape == (n_tet, 4, 3)
    assert ctx.vols.dtype == cp.float32 and ctx.vols.shape == (n_tet,)
    assert ctx.neg_vc.dtype == cp.float32
    for name in ("g", "vols", "neg_vc", "tet_nodes", "tet_tags"):
        assert getattr(ctx, name).flags.c_contiguous, name

    cond = conductivity_per_tet(ctx.tet_tags).astype(cp.float32)
    np.testing.assert_allclose(
        cp.asnumpy(ctx.neg_vc), -cp.asnumpy(ctx.vols) * cp.asnumpy(cond), rtol=1e-6
    )


def test_solver_bindings_reject_host_arrays(cp, cube_subject):
    """``nb::device::cuda`` on every binding parameter: a host array is a type error."""
    from cunibs.solver import dadt_node_to_element

    ctx = cube_subject.context
    n_tet = int(ctx.tet_nodes.shape[0])
    nodal = cp.zeros((int(ctx.n_nodes), 3), dtype=cp.float32)
    out = cp.empty((n_tet, 3), dtype=cp.float32)
    with pytest.raises(TypeError):
        dadt_node_to_element(
            cp.asnumpy(nodal), ctx.tet_nodes, out, cp.cuda.get_current_stream().ptr
        )


def test_solver_bindings_reject_dtype_and_layout(cp):
    """``.noconvert()`` on every array parameter: dtype and layout are strict, not converting.

    Without it nanobind repairs a mismatch through cupy's ``.astype(dtype, order)``, costing a
    device allocation and copy per call — invisible in a hot loop.
    """
    from cunibs.solver import dadt_node_to_element

    stream = cp.cuda.get_current_stream().ptr
    nodal = cp.arange(8 * 3, dtype=cp.float32).reshape(8, 3)
    tets = cp.zeros((5, 4), dtype=cp.int32)

    for variant in (nodal.astype(cp.float64), cp.repeat(nodal, 2, axis=0)[::2]):
        out = cp.zeros((5, 3), dtype=cp.float32)
        with pytest.raises(TypeError):
            dadt_node_to_element(variant, tets, out, stream)


def test_solver_bindings_reject_mismatched_output_buffer(cp):
    """A converted *output* buffer would swallow the kernel's writes.

    nanobind has no writeback: without ``.noconvert()`` it would hand the kernel a converted
    temporary, free it after the call, and leave the caller's array untouched with no error.
    """
    from cunibs.solver import dadt_node_to_element

    stream = cp.cuda.get_current_stream().ptr
    nodal = cp.arange(8 * 3, dtype=cp.float32).reshape(8, 3)
    tets = cp.zeros((5, 4), dtype=cp.int32)

    for bad_out in (
        cp.zeros((5, 3), dtype=cp.float64),
        cp.zeros((10, 3), dtype=cp.float32)[::2],
    ):
        with pytest.raises(TypeError):
            dadt_node_to_element(nodal, tets, bad_out, stream)


def test_solver_bindings_reject_dtype_inside_list_arg(cp):
    """``.noconvert()`` reaches through ``std::vector<nb::ndarray>`` to the element caster.

    nanobind's list caster forwards the flags unchanged, so one wrong-dtype entry in the
    per-placement list is rejected rather than silently converted.
    """
    from cunibs.solver import rhs_assemble_staged_block

    stream = cp.cuda.get_current_stream().ptr
    n_tet, n_nodes, k = 6, 5, 2
    g = cp.zeros((n_tet, 4, 3), dtype=cp.float32)
    neg_vc = cp.zeros(n_tet, dtype=cp.float32)
    ptr = cp.zeros(n_nodes + 1, dtype=cp.int32)
    idx = cp.zeros(1, dtype=cp.int32)
    b_block = cp.zeros((n_nodes, k), dtype=cp.float32)
    good = cp.zeros((n_tet, 3), dtype=cp.float32)

    dadt_elm = [good, good.astype(cp.float64)]
    with pytest.raises(TypeError):
        rhs_assemble_staged_block(dadt_elm, g, neg_vc, ptr, idx, b_block, stream)


@pytest.mark.realmesh
def test_build_node2corner_on_patch(cp, patch_mesh):
    """The corner transpose at 41k tets, where the stable sort actually has work to do."""
    from cunibs.fem.assembly import build_node2corner

    tets = cp.asarray(patch_mesh.tet_nodes)
    ptr, idx = build_node2corner(tets, patch_mesh.n_nodes)
    ptr, idx = cp.asnumpy(ptr), cp.asnumpy(idx)
    flat = patch_mesh.tet_nodes.ravel()

    assert idx.size == patch_mesh.tet_nodes.size
    assert ptr[0] == 0 and ptr[-1] == idx.size
    # Every corner slot appears exactly once, under the node it belongs to.
    np.testing.assert_array_equal(np.sort(idx), np.arange(idx.size))
    owner = np.repeat(np.arange(patch_mesh.n_nodes), np.diff(ptr))
    np.testing.assert_array_equal(flat[idx], owner)


def test_reconstruct_matches_numpy_reference(cp, cube_mesh):
    from cunibs.fem import build_context, solve_placement

    ctx = build_context(cube_mesh)
    coil_pos = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
    coil_mom = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    out = solve_placement(ctx, coil_pos, coil_mom, [50, 50, 100], [50, 100, 100], 4.0, 1e6)
    v = cp.asnumpy(out["v"])
    g = cp.asnumpy(ctx.g).astype(np.float64)
    tet_nodes = cp.asnumpy(ctx.tet_nodes)
    dadt = cp.asnumpy(out["dadt_elm"]).astype(np.float64)
    grad_v = np.einsum("ei,eik->ek", v[tet_nodes], g)
    e_ref = -grad_v - dadt
    np.testing.assert_allclose(
        cp.asnumpy(out["E"]), e_ref, rtol=1e-3, atol=1e-4 * np.abs(e_ref).max()
    )
