from __future__ import annotations

import numpy as np
import pytest

from gpu import requires_gpu

pytestmark = requires_gpu


@pytest.fixture
def cp():
    import cupy

    return cupy


def test_conductivity_mapping_and_unknown_tag(cp):
    from cunibs.fem.assembly import conductivity_per_tet

    cond = conductivity_per_tet(cp.asarray([1, 2, 3], dtype=cp.int32))
    np.testing.assert_allclose(cp.asnumpy(cond), [0.126, 0.275, 1.654])
    with pytest.raises(ValueError):
        conductivity_per_tet(cp.asarray([2, 99], dtype=cp.int32))


def test_gradient_operator_reference_tet(cp):
    from cunibs.fem.assembly import gradient_operator

    nodes = cp.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=cp.float64)
    tets = cp.asarray([[0, 1, 2, 3]], dtype=cp.int32)
    g, vols = gradient_operator(nodes, tets)
    assert float(vols[0]) == pytest.approx(1 / 6)
    # P1 basis gradients sum to zero (partition of unity).
    np.testing.assert_allclose(cp.asnumpy(g[0].sum(0)), [0, 0, 0], atol=1e-12)


def test_stiffness_symmetric_zero_rowsum(cp, cube_mesh):
    from cunibs.fem.assembly import (
        assemble_stiffness,
        conductivity_per_tet,
        gradient_operator,
    )

    nodes = cp.asarray(cube_mesh.nodes_mm) * 1e-3
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
