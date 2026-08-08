"""Block-CG parity: ``Subject.simulate``'s default path against the serial solve.

``simulate`` batches placements through ``solve_placements_block`` at ``block_k=MAX_BLOCK``
by default, so the lockstep k-RHS kernels and the RHS padding are what almost every caller
actually runs. The oracle throughout is the one-placement-at-a-time ``solve_placement`` path,
held to the same 2e-5 relative-L2 gate the block probe uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.fem import solve_placement, solve_placements_block
from cunibs.fem.solve import MAX_BLOCK
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
        for c, a in zip(picked, np.linspace(0, 2 * np.pi, 8, endpoint=False), strict=False)
    ]


def serial(ctx, coil, sites):
    return [
        solve_placement(ctx, coil.positions_m, coil.moments, c, h, d, DIDT) for c, h, d in sites
    ]


def block(ctx, coil, sites):
    return solve_placements_block(ctx, coil.positions_m, coil.moments, sites, DIDT)


def test_block_k1_matches_serial(cube_subject, synthetic_coil):
    """k == 1 runs the block kernels at width 1 and matches ``solve_placement`` bit for bit.

    Two separate implementations over the same operator, agreeing because both reduce in the
    same order.
    """
    import cupy as cp

    sites = cube_sites(1)
    got = block(cube_subject.context, synthetic_coil, sites)[0]
    ref = serial(cube_subject.context, synthetic_coil, sites)[0]
    for key in ("E", "magnE", "v"):
        np.testing.assert_array_equal(cp.asnumpy(got[key]), cp.asnumpy(ref[key]), err_msg=key)
    # The telemetry reaches the two results by different routes: the block path threads a
    # per-column list out of the solve, the serial one reads the solver afterwards.
    for key in ("iterations", "relative_residual"):
        assert got[key] == ref[key], key


@pytest.mark.realmesh
def test_single_and_block_solvers_share_one_order(cp, fresh_subject, patch_mesh):
    """``solve_mixed`` and ``solve_mixed_block`` at k=1 must agree bit for bit.

    Both are reachable for the same placement: the block path solves it, and the retry path
    re-solves a column that missed tolerance through the single-RHS one. If the two reduced in
    different orders, a placement's field would depend on whether it converged first time.
    """
    solver = fresh_subject(patch_mesh).context.solver
    n = int(solver.idx.shape[0])
    b = cp.sin(cp.arange(n, dtype=cp.float64) * 0.5) + 2.0
    stream = cp.cuda.get_current_stream().ptr

    x_single = cp.empty(n, dtype=cp.float64)
    solver.pcg.solve_mixed(
        solver.precond, b, x_single, solver.tolerance, solver.max_iters, stream
    )
    b_block = cp.ascontiguousarray(b.reshape(n, 1))
    x_block = cp.empty_like(b_block)
    solver.pcg.solve_mixed_block(
        solver.precond, b_block, x_block, solver.tolerance, solver.max_iters, stream
    )
    np.testing.assert_array_equal(cp.asnumpy(x_block[:, 0]), cp.asnumpy(x_single))


@pytest.mark.parametrize("k", [2, 4, 8])
def test_block_matches_serial_cube(cube_subject, synthetic_coil, k):
    sites = cube_sites(k)
    ctx = cube_subject.context
    for got, ref in zip(
        block(ctx, synthetic_coil, sites), serial(ctx, synthetic_coil, sites), strict=False
    ):
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
    for i, (g, r) in enumerate(zip(got, ref, strict=False)):
        assert rel_l2(g["magnE"], r["magnE"]) <= PARITY, f"column {i}"
        assert rel_l2(g["E"], r["E"]) <= PARITY, f"column {i}"


@pytest.mark.realmesh
def test_block_matches_serial_patch_k8(patch_subject, d70_coil, patch_sites):
    """The same parity on real multi-tissue geometry with a full 3000-dipole coil."""
    ctx = patch_subject.context
    got = block(ctx, d70_coil, patch_sites)
    ref = serial(ctx, d70_coil, patch_sites)
    for i, (g, r) in enumerate(zip(got, ref, strict=False)):
        assert rel_l2(g["magnE"], r["magnE"]) <= PARITY, f"placement {i}"
        assert rel_l2(g["E"], r["E"]) <= PARITY, f"placement {i}"
        assert rel_l2(g["v"], r["v"]) <= PARITY, f"placement {i}"


def test_block_k_above_max_raises(cube_subject, synthetic_coil):
    with pytest.raises(ValueError, match=f"MAX_BLOCK={MAX_BLOCK}"):
        block(cube_subject.context, synthetic_coil, cube_sites(MAX_BLOCK + 1))


def _magn(subject, coil, placements, block_k):
    return [
        r.magnE
        for r in subject.iter_simulate(coil, placements, DIDT, magnitude=True, block_k=block_k)
    ]


@pytest.mark.realmesh
@pytest.mark.parametrize("cut", [1, 2, 3, 5])
def test_batch_composition_does_not_change_results(patch_subject, d70_coil, patch_sites, cut):
    """Regrouping a sweep into different blocks must be bitwise invisible.

    The cuts are deliberately misaligned with the width. Splitting eight placements at four
    with ``block_k=4`` rebuilds the same two chunks and so tests nothing; cutting at three
    puts placements 3..6 in a block that no chunk of the whole sweep contains, and leaves
    placement 7 alone in a block of one. This is what lets a caller batch however they like,
    or resume an interrupted sweep, without their numbers moving.
    """
    placements = [Placement(c, h, d) for c, h, d in patch_sites]
    whole = _magn(patch_subject, d70_coil, placements, 4)
    split = _magn(patch_subject, d70_coil, placements[:cut], 4)
    split += _magn(patch_subject, d70_coil, placements[cut:], 4)
    for i, (w, s) in enumerate(zip(whole, split, strict=True)):
        np.testing.assert_array_equal(s, w, err_msg=f"cut={cut}, placement {i}")


@pytest.mark.realmesh
def test_sweep_position_invariance(patch_subject, d70_coil, patch_sites):
    """The same placement gives the same field at the head and at the tail of a sweep."""
    first, rest = patch_sites[0], patch_sites[1:]
    head = [Placement(*first)] + [Placement(*s) for s in rest]
    tail = [Placement(*s) for s in rest] + [Placement(*first)]
    got_head = _magn(patch_subject, d70_coil, head, 4)[0]
    got_tail = _magn(patch_subject, d70_coil, tail, 4)[-1]
    np.testing.assert_array_equal(got_tail, got_head)


@pytest.mark.realmesh
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_block_width_invariance(patch_subject, d70_coil, patch_sites, k):
    """block_k is a throughput and memory knob, never an accuracy one.

    Every compiled width shares one fp64 summation order and every column stops on its own
    residual, so the width cannot reach the answer at all -- not even in the last bits. A
    researcher who retunes block_k for a different GPU gets the same numbers.
    """
    placements = [Placement(c, h, d) for c, h, d in patch_sites]
    ref = _magn(patch_subject, d70_coil, placements, 1)
    got = _magn(patch_subject, d70_coil, placements, k)
    for i, (g, r) in enumerate(zip(got, ref, strict=True)):
        np.testing.assert_array_equal(g, r, err_msg=f"block_k={k}, placement {i}")


def _telemetry(outs):
    return [(o["iterations"], o["relative_residual"]) for o in outs]


@pytest.mark.realmesh
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_per_placement_telemetry_is_block_width_invariant(
    patch_subject, d70_coil, patch_sites, k
):
    """A placement's reported iterations and residual are its own, not its block's.

    Exact equality on the residual too: the per-column arithmetic is width-invariant, so there
    is nothing for a float to drift by. The padded widths are in the list because a padded
    column's residual bleeding into a live one would show up here first.
    """
    ctx = patch_subject.context
    ref = _telemetry(serial(ctx, d70_coil, patch_sites))
    got = [
        t
        for i in range(0, len(patch_sites), k)
        for t in _telemetry(block(ctx, d70_coil, patch_sites[i : i + k]))
    ]
    assert got == ref, f"block_k={k}"


@pytest.mark.realmesh
def test_repeated_subjects_agree_bitwise(fresh_subject, patch_mesh, d70_coil, patch_sites):
    """The same mesh built into three subjects must give the same field.

    This is the end-to-end cover for stiffness assembly: an ULP of drift in a few percent of
    the nonzeros reaches |E| at ~1e-9 relative. Nothing about a placement changes between these
    runs, so nothing about its field may.
    """
    placements = [Placement(*patch_sites[0])]
    fields = [_magn(fresh_subject(patch_mesh), d70_coil, placements, 8)[0] for _ in range(3)]
    for i, got in enumerate(fields[1:], 1):
        np.testing.assert_array_equal(got, fields[0], err_msg=f"subject {i}")


@pytest.mark.realmesh
def test_simulate_block_k_matches_serial(patch_subject, d70_coil, patch_sites):
    """Through the public API: the default block path agrees with block_k=1."""
    placements = [Placement(c, h, d) for c, h, d in patch_sites] * 2
    blocked = list(patch_subject.iter_simulate(d70_coil, placements, DIDT))
    serialized = list(patch_subject.iter_simulate(d70_coil, placements, DIDT, block_k=1))
    assert len(blocked) == len(serialized) == len(placements)
    for b, s in zip(blocked, serialized, strict=False):
        assert b.peak_magnE() == pytest.approx(s.peak_magnE(), rel=1e-5)
        np.testing.assert_allclose(b.peak_location_mm(), s.peak_location_mm(), atol=1e-9)


def _solve_two(solver, cp, left, right):
    """Solve a k=2 block; returns its columns, the block's count, and each column's own."""
    b = cp.ascontiguousarray(cp.stack([left, right], axis=1))
    x = cp.empty_like(b)
    iters, _, col_iters = solver.pcg.solve_mixed_block(
        solver.precond,
        b,
        x,
        solver.tolerance,
        solver.max_iters,
        cp.cuda.get_current_stream().ptr,
    )
    return x, int(iters), [int(i) for i in col_iters]


@pytest.mark.realmesh
def test_a_column_stops_on_its_own_residual(cp, fresh_subject, patch_mesh):
    """Neither a column's answer nor its cost may depend on how fast its block-mates converge.

    A column carried past its own tolerance by a slower block-mate lands further inside the
    tolerance ball. That is over-convergence rather than error, but it makes the field a
    function of who else was in the batch, which a caller can neither control nor reproduce.

    Every placement on the patch converges in the same 11 iterations, so composition alone
    cannot show this. Two synthetic right-hand sides with different spectra can: the sine
    needs 12 iterations and the two-point source 10, and the extra steps move the fast column
    by ~3e-7, two orders above anything the arithmetic contributes.
    """
    solver = fresh_subject(patch_mesh).context.solver
    n = int(solver.idx.shape[0])

    fast = cp.zeros(n, dtype=cp.float64)
    fast[n // 3] = 1.0
    fast[2 * n // 3] = -1.0
    slow = cp.sin(cp.arange(n, dtype=cp.float64) * (2 * np.pi / n))

    with_twin, twin_iters, twin_cols = _solve_two(solver, cp, fast, fast)
    with_slow, slow_iters, cols = _solve_two(solver, cp, fast, slow)
    assert slow_iters > twin_iters, "the companion must actually take longer"

    np.testing.assert_array_equal(cp.asnumpy(with_slow[:, 0]), cp.asnumpy(with_twin[:, 0]))
    assert twin_cols == [twin_iters, twin_iters]
    # The fast column is charged what it cost alone, and the block runs as long as its slowest.
    assert cols == [twin_iters, slow_iters]


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
def test_unreachable_tolerance_rebuilds_then_raises_block(
    fresh_subject, patch_mesh, d70_coil, patch_sites
):
    """An unreachable tolerance must rebuild the preconditioner once, then fail loudly.

    Reporting a converged solve that did not converge is the one outcome worse than raising;
    the fp64 fallback this replaced reported a residual of exactly 0.0 for a solve that had
    only reached 1e-6.
    """
    from cunibs.fem import SolverConvergenceError

    subj = fresh_subject(patch_mesh)
    ctx = subj.context
    sites = patch_sites[:4]
    block(ctx, d70_coil, sites)

    _force_fallback(ctx)
    before = ctx.solver.precond
    with pytest.raises(SolverConvergenceError) as excinfo:
        block(ctx, d70_coil, sites)

    assert ctx.solver.precond is not before, "the preconditioner was not rebuilt"
    assert excinfo.value.relative_residual > 0.0
    assert excinfo.value.tolerance == 0.0
    assert excinfo.value.column in range(len(sites)), "the failing column must be attributable"


@pytest.mark.realmesh
def test_unreachable_tolerance_rebuilds_then_raises_serial(
    fresh_subject, patch_mesh, d70_coil, patch_sites
):
    """The same retry-then-raise on the single-RHS path in ``solve_grounded``."""
    from cunibs.fem import SolverConvergenceError

    subj = fresh_subject(patch_mesh)
    ctx = subj.context
    site = patch_sites[0]
    serial(ctx, d70_coil, [site])

    _force_fallback(ctx)
    before = ctx.solver.precond
    with pytest.raises(SolverConvergenceError) as excinfo:
        serial(ctx, d70_coil, [site])
    assert ctx.solver.precond is not before, "the preconditioner was not rebuilt"
    assert excinfo.value.column is None, "there is no block to be a column of"


@pytest.mark.realmesh
def test_rebuild_preconditioner_restores_the_full_hierarchy(
    cp, fresh_subject, patch_mesh, d70_coil, patch_sites
):
    """Swapping the hierarchy changes only the rate; rebuilding restores the original exactly.

    That the preconditioner cannot move the fixed point is what lets the retry path replace an
    fp64 fallback solver: there is no more accurate solve to fall back to, only a faster one.

    The substitute is a hierarchy built from single pairwise passes rather than the composed
    pairs production uses, so on the patch it coarsens 8403 -> 1936 -> 465 where production goes
    straight to 465. Two coarsening levels against one, over the same matrix and down to the
    same dense coarse solve: a different rate reaching the same answer, which is the property
    under test. It also covers the multi-level recursion, which the patch's own hierarchy does
    not reach.
    """
    from cunibs.fem.solve import AggregationParams, build_native_vcycle

    subj = fresh_subject(patch_mesh)
    ctx = subj.context
    site = patch_sites[0]
    reference = serial(ctx, d70_coil, [site])[0]
    good_iters = ctx.solver.last_iterations
    good_levels = ctx.solver.precond.n_levels()
    assert good_levels >= 1

    other = build_native_vcycle(
        ctx.solver.row_ptr,
        ctx.solver.col_idx,
        cp.ascontiguousarray(ctx.solver.values.astype(cp.float32)),
        AggregationParams(rounds=1),
    )
    assert other.n_levels() > good_levels, "the substitute must be a different hierarchy"
    ctx.solver.precond = other

    swapped = serial(ctx, d70_coil, [site])[0]
    assert ctx.solver.precond is other, "the substitute should converge unaided"
    assert ctx.solver.last_relative_residual <= ctx.solver.tolerance
    assert rel_l2(swapped["magnE"], reference["magnE"]) <= PARITY

    ctx.solver.rebuild_preconditioner()
    assert ctx.solver.precond.n_levels() == good_levels
    rebuilt = serial(ctx, d70_coil, [site])[0]
    # Rebuilt from the same values by the same deterministic selector, so the hierarchy is
    # identical and the solve must repeat exactly.
    assert ctx.solver.last_iterations == good_iters
    assert rel_l2(rebuilt["magnE"], reference["magnE"]) <= PARITY
