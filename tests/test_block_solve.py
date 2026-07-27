"""Block-CG parity: ``Subject.simulate``'s default path against the serial solve.

``simulate`` batches placements through ``solve_placements_block`` at ``block_k=MAX_BLOCK``
by default, so the lockstep k-RHS kernels, the RHS padding, and the warm start are what
almost every caller actually runs. The oracle throughout is the one-placement-at-a-time
``solve_placement`` path, held to the same 2e-5 relative-L2 gate the block probe uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.fem import solve_placement, solve_placements_block
from cunibs.fem.solve import MAX_BLOCK, BlockWarmStart
from cunibs.simulation import Placement

pytestmark = pytest.mark.gpu

PARITY = 2e-5
DIDT = 1e6


def rel_l2(got, ref) -> float:
    import cupy as cp

    got, ref = cp.asnumpy(got).astype(np.float64), cp.asnumpy(ref).astype(np.float64)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(got - ref) / denom) if denom else float(np.linalg.norm(got))


def cube_sites(k: int):
    """``k`` distinct placements over the cube's top face, each with its own handle angle."""
    sites = []
    for i in range(k):
        angle = 2 * np.pi * i / max(k, 1)
        cx = 50.0 + 12.0 * np.cos(angle)
        cy = 50.0 + 12.0 * np.sin(angle)
        center = np.array([cx, cy, 100.0])
        handle = center + np.array([50.0 * np.cos(angle), 50.0 * np.sin(angle), 0.0])
        sites.append((center, handle, 4.0))
    return sites


@pytest.fixture(scope="session")
def patch_sites(patch_mesh):
    """Eight well-separated scalp placements on the real patch."""
    centroids = patch_mesh.nodes_mm[patch_mesh.skin_tris].mean(axis=1)
    top = centroids[np.argsort(centroids[:, 2])[-400:]]
    picked = top[:: max(len(top) // 8, 1)][:8]
    return [
        (c, c + np.array([40.0 * np.cos(a), 40.0 * np.sin(a), 0.0]), 4.0)
        for c, a in zip(picked, np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ]


def serial(ctx, coil, sites):
    return [
        solve_placement(ctx, coil.positions_m, coil.moments, c, h, d, DIDT) for c, h, d in sites
    ]


def block(ctx, coil, sites, warm=None):
    return solve_placements_block(ctx, coil.positions_m, coil.moments, sites, DIDT, warm)


def test_block_k1_delegates_to_serial(cube_subject, synthetic_coil):
    """k == 1 forwards straight to ``solve_placement``, so the arrays are bitwise identical."""
    import cupy as cp

    sites = cube_sites(1)
    got = block(cube_subject.context, synthetic_coil, sites)[0]
    ref = serial(cube_subject.context, synthetic_coil, sites)[0]
    for key in ("E", "magnE", "v"):
        np.testing.assert_array_equal(cp.asnumpy(got[key]), cp.asnumpy(ref[key]), err_msg=key)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_block_matches_serial_cube(cube_subject, synthetic_coil, k):
    sites = cube_sites(k)
    ctx = cube_subject.context
    for got, ref in zip(block(ctx, synthetic_coil, sites), serial(ctx, synthetic_coil, sites)):
        assert rel_l2(got["magnE"], ref["magnE"]) <= PARITY
        assert rel_l2(got["E"], ref["E"]) <= PARITY
        assert rel_l2(got["v"], ref["v"]) <= PARITY


@pytest.mark.parametrize("k", [3, 5, 6, 7])
def test_block_padding_widths(cube_subject, synthetic_coil, k):
    """Widths off the compiled 2/4/8 ladder pad by replicating the last column.

    Every real column must still match serial — a padded column bleeding into a live one
    would show up here and nowhere else.
    """
    sites = cube_sites(k)
    ctx = cube_subject.context
    got = block(ctx, synthetic_coil, sites)
    ref = serial(ctx, synthetic_coil, sites)
    assert len(got) == k
    for i, (g, r) in enumerate(zip(got, ref)):
        assert rel_l2(g["magnE"], r["magnE"]) <= PARITY, f"column {i}"
        assert rel_l2(g["E"], r["E"]) <= PARITY, f"column {i}"


@pytest.mark.realmesh
def test_block_matches_serial_patch_k8(patch_subject, d70_coil, patch_sites):
    """The same parity on real multi-tissue geometry with a full 3000-dipole coil."""
    ctx = patch_subject.context
    got = block(ctx, d70_coil, patch_sites)
    ref = serial(ctx, d70_coil, patch_sites)
    for i, (g, r) in enumerate(zip(got, ref)):
        assert rel_l2(g["magnE"], r["magnE"]) <= PARITY, f"placement {i}"
        assert rel_l2(g["E"], r["E"]) <= PARITY, f"placement {i}"
        assert rel_l2(g["v"], r["v"]) <= PARITY, f"placement {i}"


def test_block_k_above_max_raises(cube_subject, synthetic_coil):
    with pytest.raises(ValueError, match=f"MAX_BLOCK={MAX_BLOCK}"):
        block(cube_subject.context, synthetic_coil, cube_sites(MAX_BLOCK + 1))


@pytest.mark.realmesh
def test_warm_start_does_not_change_solution(patch_subject, d70_coil, patch_sites):
    """The stopping test is ‖r‖/‖b‖, not ‖r‖/‖r₀‖, so a seeded x0 may only change iterations."""
    ctx = patch_subject.context
    first, second = patch_sites[:4], patch_sites[4:]

    warm = BlockWarmStart()
    block(ctx, d70_coil, first, warm)
    assert warm.centers is not None and warm.centers.shape == (4, 3)
    assert warm.x_red is not None and warm.x_red.shape[1] == 4

    warmed = block(ctx, d70_coil, second, warm)
    cold = block(ctx, d70_coil, second, None)
    for i, (w, c) in enumerate(zip(warmed, cold)):
        assert rel_l2(w["magnE"], c["magnE"]) <= 2e-6, f"placement {i}"
        assert rel_l2(w["v"], c["v"]) <= 2e-6, f"placement {i}"


@pytest.mark.realmesh
def test_warm_start_reduces_iterations(patch_subject, d70_coil, patch_sites):
    """Re-solving near-identical placements from a warm x0 must not cost more iterations."""
    ctx = patch_subject.context
    sites = patch_sites[:4]
    nudged = [(c + 0.05, h + 0.05, d) for c, h, d in sites]

    warm = BlockWarmStart()
    block(ctx, d70_coil, sites, warm)
    block(ctx, d70_coil, nudged, warm)
    warm_iters = ctx.solver.last_iterations

    block(ctx, d70_coil, nudged, None)
    cold_iters = ctx.solver.last_iterations

    assert 0 < warm_iters <= cold_iters


@pytest.mark.realmesh
def test_simulate_block_k_matches_serial(patch_subject, d70_coil, patch_sites):
    """Through the public API: the default block path agrees with block_k=1."""
    placements = [Placement(c, h, d) for c, h, d in patch_sites] * 2
    blocked = list(patch_subject.iter_simulate(d70_coil, placements, DIDT))
    serialized = list(patch_subject.iter_simulate(d70_coil, placements, DIDT, block_k=1))
    assert len(blocked) == len(serialized) == len(placements)
    for b, s in zip(blocked, serialized):
        assert b.peak_magnE() == pytest.approx(s.peak_magnE(), rel=1e-5)
        np.testing.assert_allclose(b.peak_location_mm(), s.peak_location_mm(), atol=1e-9)


def test_simulate_block_k_is_clamped(cube_subject, synthetic_coil):
    """block_k is clamped into [1, MAX_BLOCK] rather than rejected."""
    placements = [Placement(c, h, d) for c, h, d in cube_sites(8)]
    peaks = {
        k: [
            r.peak_magnE()
            for r in cube_subject.iter_simulate(synthetic_coil, placements, block_k=k)
        ]
        for k in (0, 1, MAX_BLOCK, 100, None)
    }
    assert peaks[0] == peaks[1]
    assert peaks[100] == peaks[MAX_BLOCK] == peaks[None]


def _force_fallback(ctx) -> None:
    """Make the mixed PCG miss unconditionally.

    A merely tiny tolerance is not enough — the mixed solver reaches ~5e-17 relative
    residual on this system. Only a target of exactly zero is unreachable.
    """
    ctx.solver.tolerance = 0.0
    ctx.solver.max_iters = 25


@pytest.mark.realmesh
def test_fp64_fallback_matches_mixed_solve(fresh_subject, patch_mesh, d70_coil, patch_sites):
    """An unreachable tolerance drives every block column through the lazy fp64 AMGX path."""
    subj = fresh_subject(patch_mesh)
    ctx = subj.context
    sites = patch_sites[:4]
    reference = block(ctx, d70_coil, sites)
    assert ctx.solver.amgx is None

    _force_fallback(ctx)
    fallback = block(ctx, d70_coil, sites)
    assert ctx.solver.amgx is not None

    for i, (f, r) in enumerate(zip(fallback, reference)):
        assert rel_l2(f["magnE"], r["magnE"]) <= PARITY, f"placement {i}"
        assert rel_l2(f["v"], r["v"]) <= PARITY, f"placement {i}"


@pytest.mark.realmesh
def test_fp64_fallback_matches_serial_solve(fresh_subject, patch_mesh, d70_coil, patch_sites):
    """The same fallback on the single-RHS path in ``solve_grounded``."""
    subj = fresh_subject(patch_mesh)
    ctx = subj.context
    site = patch_sites[0]
    reference = serial(ctx, d70_coil, [site])[0]

    _force_fallback(ctx)
    fallback = serial(ctx, d70_coil, [site])[0]
    assert ctx.solver.amgx is not None
    assert ctx.solver.last_relative_residual == 0.0
    # The fallback converges to AMGX's own 1e-6 relative tolerance, so the two independent
    # solutions agree to roughly that, not to machine precision.
    assert rel_l2(fallback["magnE"], reference["magnE"]) <= PARITY
