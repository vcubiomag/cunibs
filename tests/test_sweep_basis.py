"""Seeding block solves from a per-subject sweep basis."""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.simulation import Placement

pytestmark = pytest.mark.gpu

DIDT = 1e6


@pytest.mark.parametrize("m", [0, 1100])
def test_sweep_seed_handles_an_empty_or_wide_basis(cp, m):
    """A basis past one block's threads is tiled, and an empty one seeds zero."""
    from cunibs.fem.sweep import SweepBasis

    rng = np.random.default_rng(1)
    n = 3000
    q = cp.asarray((rng.standard_normal((n, m)) / np.sqrt(n)).astype(np.float32))
    b = cp.asarray(rng.standard_normal((n, 2)))
    got = cp.asnumpy(SweepBasis(q).seed(b))
    q64 = q.astype(cp.float64)
    np.testing.assert_allclose(got, cp.asnumpy(q64 @ (q64.T @ b)), rtol=1e-3, atol=1e-4)


def test_sweep_seed_is_the_projection_and_ignores_the_block(cp):
    """x0 = q (q^T b), and a column's seed is bitwise the same at every block width."""
    from cunibs.fem.sweep import SweepBasis

    rng = np.random.default_rng(0)
    n, m = 5000, 12
    q = cp.asarray(rng.standard_normal((n, m)).astype(np.float32))
    b = cp.asarray(rng.standard_normal((n, 8)))
    basis = SweepBasis(q)

    got = basis.seed(b)
    q64 = q.astype(cp.float64)
    np.testing.assert_allclose(
        cp.asnumpy(got), cp.asnumpy(q64 @ (q64.T @ b)), rtol=2e-4, atol=1e-3
    )
    for k in (1, 2, 4):
        for c0 in range(0, 8, k):
            part = basis.seed(cp.ascontiguousarray(b[:, c0 : c0 + k]))
            np.testing.assert_array_equal(cp.asnumpy(part), cp.asnumpy(got[:, c0 : c0 + k]))


def _magn(subject, coil, placements, width):
    from cunibs.simulation import _block_width

    with _block_width(width):
        return [
            (r.magnE, r.iterations)
            for r in subject.iter_simulate(coil, placements, DIDT, magnitude=True)
        ]


@pytest.mark.realmesh
def test_a_seeded_sweep_keeps_its_answers_and_its_invariance(
    cp, fresh_subject, patch_mesh, d70_coil
):
    """A basis moves where CG starts, never what it converges to or what the block width does.

    Seeded and unseeded fields of the same placement agree to the solver tolerance, a seeded
    placement is bitwise the same at every block width, and the basis is A-orthonormal, which is
    what makes q q^T b the Galerkin start rather than merely a projection.
    """

    centroids = patch_mesh.nodes_mm[patch_mesh.skin_tris].mean(axis=1)
    top = centroids[np.argsort(centroids[:, 2])[-400:]]
    picked = top[:: len(top) // 8][:8]
    placements = [
        Placement(c, c + np.array([40.0 * np.cos(a), 40.0 * np.sin(a), 0.0]), 4.0)
        for c, a in zip(picked, np.linspace(0.3, 2 * np.pi, 8, endpoint=False), strict=True)
    ]

    plain = fresh_subject(patch_mesh)
    unseeded = _magn(plain, d70_coil, placements, 8)

    seeded_subject = fresh_subject(patch_mesh)
    seeded_subject.prepare_sweep(d70_coil, size=16)
    ctx = seeded_subject.context
    q = ctx.sweep_basis.q.astype(cp.float64)
    a = ctx.solver.matrix()
    gram = cp.asnumpy(q.T @ (a @ q))
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-3)

    seeded = _magn(seeded_subject, d70_coil, placements, 8)
    for i, ((s, _), (u, _)) in enumerate(zip(seeded, unseeded, strict=True)):
        rel = np.linalg.norm(s - u) / np.linalg.norm(u)
        assert rel < 1e-5, f"placement {i}: seeded field is {rel:.2e} off the unseeded one"
    assert sum(i for _, i in seeded) <= sum(i for _, i in unseeded)

    for k in (1, 3):
        for i, ((g, _), (r, _)) in enumerate(
            zip(_magn(seeded_subject, d70_coil, placements, k), seeded, strict=True)
        ):
            np.testing.assert_array_equal(g, r, err_msg=f"width={k}, placement {i}")

    # A second prepare keeps the basis the earlier solves ran against, and refuses another.
    before = ctx.sweep_basis
    seeded_subject.prepare_sweep(d70_coil, size=16)
    assert ctx.sweep_basis is before
    with pytest.raises(ValueError, match="free"):
        seeded_subject.prepare_sweep(d70_coil, size=32)
    assert ctx.sweep_basis is before

    provenance = seeded_subject.simulate(d70_coil, placements[0], DIDT).provenance
    assert provenance["sweep_basis_size"] == 16
    assert provenance["sweep_basis_coil"] == d70_coil.name
    assert (
        plain.simulate(d70_coil, placements[0], DIDT).provenance.get("sweep_basis_size") is None
    )
