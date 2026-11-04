"""Block-CG parity: ``Subject.simulate``'s default path against the serial solve.

``simulate`` batches placements through ``solve_placements_block`` at ``MAX_BLOCK`` wide unless
memory runs short, so the lockstep k-RHS kernels and the RHS padding are what almost every caller
runs. The oracle throughout is ``serial`` below, one placement at a time through the single-RHS
``solve_grounded``, to a relative L2 of 2e-5.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from cunibs.fem import solve_grounded, solve_placements_block
from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform
from cunibs.fem.solve import MAX_BLOCK, _assemble_rhs, _dadt_node_to_elm
from cunibs.simulation import Placement
from cunibs.solver import reconstruct_e

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
    """Each site alone through the single-RHS solve and the per-element reconstruction."""
    out = []
    for c, h, d in sites:
        tf = compute_coil_transform(ctx, c, h, d)
        dadt = coil_dadt_at_nodes(coil.positions_m, coil.moments, tf, DIDT, ctx.nodes_mm)
        b = _assemble_rhs(ctx, dadt.reshape(ctx.n_nodes, 1, 3)).ravel()
        v, telemetry = solve_grounded(ctx.solver, b)
        n_tet = int(ctx.tet_nodes.shape[0])
        e = cp.empty((n_tet, 3), dtype=cp.float32)
        magn = cp.empty(n_tet, dtype=cp.float32)
        reconstruct_e(
            v,
            ctx.tet_nodes,
            ctx.g,
            _dadt_node_to_elm(dadt, ctx.tet_nodes),
            e,
            magn,
            cp.cuda.get_current_stream().ptr,
        )
        out.append(
            {
                "transform": tf,
                "E": e,
                "magnE": magn,
                "v": v,
                "iterations": telemetry.iterations,
                "relative_residual": telemetry.relative_residual,
            }
        )
    return out


def block(ctx, coil, sites):
    return solve_placements_block(ctx, coil.positions_m, coil.moments, sites, DIDT)


def test_block_k1_matches_serial(cube_subject, synthetic_coil):
    """k == 1 runs the block kernels at width 1 and matches the serial path bit for bit.

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


def test_a_zero_column_solves_to_zero_beside_a_live_one(cube_subject):
    """A zero right-hand side has the exact solution zero, and its column changes no other."""
    import cupy as cp

    solver = cube_subject.context.solver
    n = int(solver.idx.shape[0])
    stream = cp.cuda.get_current_stream().ptr
    b = cp.sin(cp.arange(n, dtype=cp.float64) * 0.5) + 2.0

    def solve(B):
        X = cp.empty_like(B)
        _, rels, iters = solver.pcg.solve_mixed_block(
            solver.precond, B, X, solver.tolerance, solver.max_iters, stream
        )
        return X, rels, iters

    alone, _, _ = solve(cp.ascontiguousarray(b[:, None]))
    mixed, rels, iters = solve(cp.ascontiguousarray(cp.stack([b, cp.zeros(n)], axis=1)))
    np.testing.assert_array_equal(cp.asnumpy(mixed[:, 0]), cp.asnumpy(alone[:, 0]))
    assert not bool(mixed[:, 1].any())
    assert rels[1] == 0.0
    assert iters[1] == 0


@pytest.mark.parametrize("k", [2, 4, 8])
def test_block_matches_serial_cube(cube_subject, synthetic_coil, k):
    sites = cube_sites(k)
    ctx = cube_subject.context
    for got, ref in zip(
        block(ctx, synthetic_coil, sites), serial(ctx, synthetic_coil, sites), strict=True
    ):
        assert rel_l2(got["magnE"], ref["magnE"]) <= PARITY
        assert rel_l2(got["E"], ref["E"]) <= PARITY
        assert rel_l2(got["v"], ref["v"]) <= PARITY


@pytest.mark.parametrize("k", [3, 5, 6, 7])
def test_block_padding_widths(cube_subject, synthetic_coil, k):
    """Widths off the compiled 2/4/8 ladder pad by replicating the last column.

    Every real column must still match serial, or a padded column has bled into a live one.
    """
    sites = cube_sites(k)
    ctx = cube_subject.context
    got = block(ctx, synthetic_coil, sites)
    ref = serial(ctx, synthetic_coil, sites)
    assert len(got) == k
    for i, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert rel_l2(g["magnE"], r["magnE"]) <= PARITY, f"column {i}"
        assert rel_l2(g["E"], r["E"]) <= PARITY, f"column {i}"


@pytest.mark.realmesh
def test_block_matches_serial_patch_k8(patch_subject, d70_coil, patch_sites):
    """The same parity on real multi-tissue geometry with a full 3000-dipole coil."""
    ctx = patch_subject.context
    got = block(ctx, d70_coil, patch_sites)
    ref = serial(ctx, d70_coil, patch_sites)
    for i, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert rel_l2(g["magnE"], r["magnE"]) <= PARITY, f"placement {i}"
        assert rel_l2(g["E"], r["E"]) <= PARITY, f"placement {i}"
        assert rel_l2(g["v"], r["v"]) <= PARITY, f"placement {i}"


def test_block_k_above_max_raises(cube_subject, synthetic_coil):
    with pytest.raises(ValueError, match=f"MAX_BLOCK={MAX_BLOCK}"):
        block(cube_subject.context, synthetic_coil, cube_sites(MAX_BLOCK + 1))


def _magn(subject, coil, placements, width):
    from cunibs.simulation import _block_width

    with _block_width(width):
        return [r.magnE for r in subject.iter_simulate(coil, placements, DIDT, magnitude=True)]


@pytest.mark.realmesh
@pytest.mark.parametrize("cut", [1, 2, 3, 5])
def test_batch_composition_does_not_change_results(patch_subject, d70_coil, patch_sites, cut):
    """Regrouping a sweep into different blocks must be bitwise invisible.

    The cuts are deliberately misaligned with the width. Splitting eight placements at four
    with a width of 4 rebuilds the same two chunks and so tests nothing; cutting at three
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
    """The chunk width is a throughput and memory choice, never an accuracy one.

    Every compiled width shares one fp64 summation order and every column stops on its own
    residual, so the width cannot reach even the last bits. That is what lets a sweep narrow its
    chunks on a GPU short of memory without its numbers moving.
    """
    placements = [Placement(c, h, d) for c, h, d in patch_sites]
    ref = _magn(patch_subject, d70_coil, placements, 1)
    got = _magn(patch_subject, d70_coil, placements, k)
    for i, (g, r) in enumerate(zip(got, ref, strict=True)):
        np.testing.assert_array_equal(g, r, err_msg=f"width={k}, placement {i}")


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
    assert got == ref, f"width={k}"


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
def test_simulate_default_width_matches_serial(patch_subject, d70_coil, patch_sites):
    """Through the public API: the default chunking agrees with one placement at a time."""
    from cunibs.simulation import _block_width

    placements = [Placement(c, h, d) for c, h, d in patch_sites] * 2
    blocked = list(patch_subject.iter_simulate(d70_coil, placements, DIDT))
    with _block_width(1):
        serialized = list(patch_subject.iter_simulate(d70_coil, placements, DIDT))
    assert len(blocked) == len(serialized) == len(placements)
    for b, s in zip(blocked, serialized, strict=True):
        assert b.iterations == s.iterations
        assert b.relative_residual == s.relative_residual
        assert b.peak_magnE() == s.peak_magnE()
        assert b.summary["focality_m3"] == s.summary["focality_m3"]
        assert b.summary["distribution"] == s.summary["distribution"]
        np.testing.assert_array_equal(b.peak_location_mm(), s.peak_location_mm())
        np.testing.assert_array_equal(b.center_of_gravity_mm(), s.center_of_gravity_mm())


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


def test_a_chunk_that_runs_out_of_memory_is_retried_narrower(
    cube_subject, synthetic_coil, monkeypatch
):
    """Running out of device memory halves the chunk and changes no result."""
    from cunibs import simulation

    placements = [Placement(c, h, d) for c, h, d in cube_sites(8)]
    flags = {"magnitude": True, "vectors": True}
    reference = list(cube_subject.iter_simulate(synthetic_coil, placements, **flags))

    real = simulation.solve_placements_block
    widths = []

    def short_of_memory(ctx, pos, moment, sites, didt, **kwargs):
        widths.append(len(sites))
        if len(sites) > 2:
            raise MemoryError("out of memory")
        return real(ctx, pos, moment, sites, didt, **kwargs)

    monkeypatch.setattr(simulation, "solve_placements_block", short_of_memory)
    narrowed = list(cube_subject.iter_simulate(synthetic_coil, placements, **flags))
    assert widths[:3] == [MAX_BLOCK, MAX_BLOCK // 2, 2]
    assert set(widths[2:]) == {2}
    for got, want in zip(narrowed, reference, strict=True):
        np.testing.assert_array_equal(got.magnE, want.magnE)
        np.testing.assert_array_equal(got.E, want.E)


def test_running_out_of_memory_at_one_placement_still_raises(
    cube_subject, synthetic_coil, monkeypatch
):
    from cunibs import simulation

    def never(*_args, **_kwargs):
        raise MemoryError("out of memory")

    monkeypatch.setattr(simulation, "solve_placements_block", never)
    with pytest.raises(MemoryError):
        cube_subject.simulate(synthetic_coil, Placement(*cube_sites(1)[0]))


def test_a_native_allocation_failure_is_a_memory_error(
    cp, fresh_subject, cube_mesh, synthetic_coil
):
    """Out of memory is a MemoryError a caller can recover from, and leaves nothing behind."""
    from cunibs.solver import CudaOutOfMemory, PcgAmgSolver

    subject = fresh_subject(cube_mesh)
    a = subject.context.solver.matrix()
    cp.get_default_memory_pool().free_all_blocks()
    # Take what the device has left, a shrinking piece at a time, straight from the driver.
    held = []
    piece = cp.cuda.runtime.memGetInfo()[0]
    while piece >= 1:
        try:
            held.append(cp.cuda.runtime.malloc(piece))
        except cp.cuda.runtime.CUDARuntimeError:
            piece //= 2
    try:
        with pytest.raises(MemoryError) as raised:
            PcgAmgSolver(a.indptr, a.indices, a.data, cp.cuda.get_current_stream().ptr)
        assert isinstance(raised.value, CudaOutOfMemory)
        with pytest.raises(MemoryError):
            cp.empty(1 << 30, dtype=cp.uint8)
    finally:
        for ptr in held:
            cp.cuda.runtime.free(ptr)
    assert subject.simulate(synthetic_coil, Placement(*cube_sites(1)[0])).peak_magnE() > 0


def _force_fallback(ctx) -> None:
    """Make the mixed PCG miss unconditionally.

    A merely tiny tolerance is not enough: the mixed solver reaches ~5e-17 relative residual on
    this system. Only a target of exactly zero is unreachable.
    """
    ctx.solver.tolerance = 0.0
    ctx.solver.max_iters = 25


@pytest.mark.realmesh
def test_unreachable_tolerance_rebuilds_then_raises_block(
    fresh_subject, patch_mesh, d70_coil, patch_sites
):
    """An unreachable tolerance must rebuild the preconditioner once, then fail loudly.

    Reporting a converged solve that did not converge is the one outcome worse than raising.
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
def test_a_rebuild_recovers_from_a_preconditioner_that_gave_nan(
    cp, fresh_subject, patch_mesh, d70_coil, patch_sites
):
    """The rebuilt V-cycle is the whole remedy, so a NaN iterate must not reach the retry."""
    from cunibs.fem.solve import build_native_vcycle

    ctx = fresh_subject(patch_mesh).context
    sites = patch_sites[:2]
    a = ctx.solver.matrix()
    poison = cp.full(a.data.shape, cp.nan, dtype=cp.float32)

    ctx.solver.precond = build_native_vcycle(a.indptr, a.indices, poison)
    for r in serial(ctx, d70_coil, sites[:1]):
        assert r["relative_residual"] <= ctx.solver.tolerance

    ctx.solver.precond = build_native_vcycle(a.indptr, a.indices, poison)
    for r in block(ctx, d70_coil, sites):
        assert r["relative_residual"] <= ctx.solver.tolerance


@pytest.mark.realmesh
def test_rebuild_preconditioner_restores_the_full_hierarchy(
    cp, fresh_subject, patch_mesh, d70_coil, patch_sites
):
    """Swapping the hierarchy changes only the rate; rebuilding restores the original exactly.

    The preconditioner cannot move the fixed point, so a retry can only be faster, not more
    accurate.

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
    good_levels = ctx.solver.precond.n_levels()
    assert good_levels >= 1

    a = ctx.solver.matrix()
    other = build_native_vcycle(
        a.indptr, a.indices, a.data.astype(cp.float32), AggregationParams(rounds=1)
    )
    assert other.n_levels() > good_levels, "the substitute must be a different hierarchy"
    ctx.solver.precond = other

    swapped = serial(ctx, d70_coil, [site])[0]
    assert ctx.solver.precond is other, "the substitute should converge unaided"
    assert swapped["relative_residual"] <= ctx.solver.tolerance
    assert rel_l2(swapped["magnE"], reference["magnE"]) <= PARITY

    ctx.solver.rebuild_preconditioner()
    assert ctx.solver.precond.n_levels() == good_levels
    rebuilt = serial(ctx, d70_coil, [site])[0]
    # Rebuilt from the same values by the same deterministic selector, so the hierarchy is
    # identical and the solve must repeat exactly.
    assert rebuilt["iterations"] == reference["iterations"]
    for key in ("magnE", "E", "v"):
        np.testing.assert_array_equal(
            cp.asnumpy(rebuilt[key]), cp.asnumpy(reference[key]), err_msg=key
        )


@pytest.mark.realmesh
def test_a_block_solve_honours_its_own_tolerance(cp, fresh_subject, patch_mesh):
    """A solve must stop on its own tolerance whatever tolerance the solver last ran at.

    The iteration body is captured once and replayed, so anything it bakes in from one solve
    reaches the next, and ``build_reciprocity(tolerance=)`` shares the forward solver at any
    tolerance. Both orders matter: a loose capture replayed at a tight tolerance freezes columns
    the host still waits on and runs to max_iters, and a tight one replayed loose carries columns
    past their own crossing, which makes them depend on their block.
    """
    n = int(fresh_subject(patch_mesh).context.solver.idx.shape[0])
    b = cp.ascontiguousarray(
        cp.stack(
            [cp.sin(cp.arange(n, dtype=cp.float64) * 0.5) + 2.0, cp.cos(cp.arange(n) * 0.3)], 1
        )
    )
    stream = cp.cuda.get_current_stream().ptr

    def solve(solver, tol):
        x = cp.empty_like(b)
        iters, rels, _ = solver.pcg.solve_mixed_block(
            solver.precond, b, x, tol, solver.max_iters, stream
        )
        return x, int(iters), max(rels)

    for first, second in [(1e-3, 1e-9), (1e-9, 1e-3)]:
        shared = fresh_subject(patch_mesh).context.solver
        solve(shared, first)
        got, iters, rel = solve(shared, second)
        ref, ref_iters, _ = solve(fresh_subject(patch_mesh).context.solver, second)
        assert rel <= second, (
            f"after a solve at {first}, a solve at {second} ended at {rel:.2e}"
        )
        assert iters == ref_iters
        np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(ref))


@pytest.mark.realmesh
def test_retained_arrays_copied_behind_the_next_chunk_match_a_direct_copy(
    patch_subject, d70_coil, patch_sites
):
    """A multi-chunk sweep copies its arrays while the next chunk computes; nothing may differ."""
    from cunibs import Placement

    placements = [Placement(c, h, d) for c, h, d in patch_sites] * 2 + [
        Placement(*patch_sites[0])
    ]
    flags = {"magnitude": True, "vectors": True, "potential": True, "nodal": True}
    swept = list(patch_subject.iter_simulate(d70_coil, placements, 1e6, **flags))
    assert len(swept) == len(placements)
    for i, (placement, result) in enumerate(zip(placements, swept, strict=True)):
        alone = patch_subject.simulate(d70_coil, placement, 1e6, **flags)
        assert result.placement is placement
        for name in ("magnE", "E", "v", "E_slots"):
            np.testing.assert_array_equal(
                getattr(result, name), getattr(alone, name), err_msg=f"{name}, placement {i}"
            )


def _integer_laplacian(cp, m: int = 8):
    """A 7-point grid Laplacian with integer entries, whose residuals can be exactly zero."""
    from cunibs.fem.solve import build_native_vcycle
    from cunibs.solver import PcgAmgSolver

    grid = np.arange(m**3).reshape(m, m, m)
    ahead = [np.delete(grid, -1, axis=a).ravel() for a in range(3)]
    behind = [np.delete(grid, 0, axis=a).ravel() for a in range(3)]
    row = np.concatenate([grid.ravel(), *ahead, *behind])
    col = np.concatenate([grid.ravel(), *behind, *ahead])
    val = np.concatenate([np.full(m**3, 6.0), np.full(row.size - m**3, -1.0)])
    order = np.lexsort((col, row))
    row, col, val = row[order], col[order].astype(np.int32), val[order]
    ptr = np.searchsorted(row, np.arange(m**3 + 1)).astype(np.int32)
    a = (cp.asarray(ptr), cp.asarray(col), cp.asarray(val))
    stream = cp.cuda.get_current_stream().ptr
    return (
        PcgAmgSolver(*a, stream),
        build_native_vcycle(a[0], a[1], a[2].astype(cp.float32)),
        lambda x: np.add.reduceat(val[:, None] * x[col], ptr[:-1], axis=0),
    )


def test_block_solve_keeps_a_column_x0_solves_exactly(cp):
    """Its residual is exactly zero, and the first CG step of such a column would be 0/0."""
    pcg, vcycle, matmul = _integer_laplacian(cp)
    stream = cp.cuda.get_current_stream().ptr
    rng = np.random.default_rng(0)
    x0 = rng.integers(-5, 5, size=(pcg.n, 2)).astype(np.float64)
    b = matmul(x0)
    b[:, 1] = rng.standard_normal(pcg.n)
    x0[:, 1] = 0.0

    x = cp.empty((pcg.n, 2), dtype=cp.float64)
    _, rels, col_iters = pcg.solve_mixed_block(
        vcycle, cp.asarray(b), x, 1e-6, 200, stream, cp.asarray(x0)
    )
    assert col_iters[0] == 0
    assert rels[0] == 0.0
    np.testing.assert_array_equal(cp.asnumpy(x[:, 0]), x0[:, 0])

    alone = cp.empty((pcg.n, 1), dtype=cp.float64)
    pcg.solve_mixed_block(
        vcycle, cp.asarray(b[:, 1:]), alone, 1e-6, 200, stream, cp.asarray(x0[:, 1:])
    )
    np.testing.assert_array_equal(cp.asnumpy(x[:, 1]), cp.asnumpy(alone[:, 0]))


def test_a_column_settled_above_its_recurrence_stops_moving(cp):
    """A column can meet tolerance on its true residual while its recurrence sits just above it.

    The device freezes on the recurrence alone, so such a column has to be frozen by the host, or
    it goes on stepping for as long as a slower block-mate runs.
    """
    pcg, vcycle, matmul = _integer_laplacian(cp, 16)
    stream = cp.cuda.get_current_stream().ptr
    rng = np.random.default_rng(0)
    b = np.zeros((pcg.n, 2))
    b[pcg.n // 3, 0] = 1.0
    b[2 * pcg.n // 3, 0] = -1.0
    b[:, 1] = rng.standard_normal(pcg.n)
    x0 = np.zeros_like(b)
    x0[:, 1] = 1e4 * rng.standard_normal(pcg.n)

    def solve(tolerance, width):
        x = cp.empty((pcg.n, width), dtype=cp.float64)
        _, rels, col_iters = pcg.solve_mixed_block(
            vcycle,
            cp.asarray(b[:, :width]),
            x,
            tolerance,
            200,
            stream,
            cp.asarray(x0[:, :width]),
        )
        return cp.asnumpy(x), rels, col_iters

    _, first_below, _ = solve(1e-6, 1)
    for factor in (1.0, 1.0005, 1.001):
        tolerance = float(first_below[0]) * factor
        alone, _, _ = solve(tolerance, 1)
        block, rels, col_iters = solve(tolerance, 2)
        assert col_iters[1] > col_iters[0], "the companion must actually take longer"
        np.testing.assert_array_equal(block[:, 0], alone[:, 0])
        actual = np.linalg.norm(b[:, 0] - matmul(block[:, :1]).ravel()) / np.linalg.norm(
            b[:, 0]
        )
        assert actual == pytest.approx(rels[0], rel=1e-3)
