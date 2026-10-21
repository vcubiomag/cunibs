"""The solver's design choices, each priced against the shipped configuration on one placement.

Backs the ablation table and paragraph of the cuNIBS paper: the shipped iterations and cost, what
harmonic recovery adds to a placement, the hierarchy shapes (aggregation rounds by Chebyshev
degree), a diagonally scaled CG as the baseline without multigrid, a two-level hierarchy with a
dense coarse inverse, and the block width.

A placement is timed through ``Subject.simulate`` with |E| and the field vectors returned, the
same call for every row, so the rows compare with each other and not with other tables. Field
difference is the volume-weighted RDM in gray matter from the shipped field.

Recovery is priced by its apply cost alone: the same call with ``recovery="raw"``, after the
operator was built. Building the operator is a per-subject cost that single_placement times.

The two-level row stops coarsening after one level, so the dense inverse is of that level's
Galerkin operator. On a head mesh it needs gigabytes; the row is skipped, with the size it would
have needed, when the device does not have three times that free for the matrix, its
factorisation and the inverse.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields, _measure, _run

ROUNDS = (2, 3)
DEGREES = (1, 2, 3)
WIDTHS = (1, 2, 4, 8)


def _install(solver, **aggregation) -> None:
    import cupy as cp

    # Hierarchy shapes are not options of the public API, so each one is built and installed
    # directly on the solver; installing it also drops the captured CG graph.
    from cunibs.fem.solve import AggregationParams, SmootherParams, build_native_vcycle

    degree = aggregation.pop("degree", 2)
    solver.precond = build_native_vcycle(
        solver.row_ptr,
        solver.col_idx,
        solver.values.astype(cp.float32),
        AggregationParams(**aggregation),
        SmootherParams(degree=degree),
    )


def _field_from_potential(context, v, dadt_elm):
    import cupy as cp

    t = context.tet_nodes
    dv = v[t[:, 1:]] - v[t[:, :1]]
    return cp.asnumpy(-cp.einsum("ni,nic->nc", dv, context.g[:, 1:]) - dadt_elm)


def _ds_cg(subject, coil, placement, gm, w) -> dict:
    """The same reduced system solved by fp64 CG with a diagonal preconditioner."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csl

    from cunibs.fem import coil_dadt_at_nodes, solve_placement

    # The RHS has no public form, and the baseline has to solve the system the shipped solver did.
    from cunibs.fem.solve import _assemble_rhs, _dadt_node_to_elm

    context = subject.context
    solver = context.solver
    shipped = solve_placement(
        context,
        coil.positions_m,
        coil.moments,
        placement.center_mm,
        placement.handle_mm,
        placement.distance_mm,
        _data.DIDT,
    )
    dadt = coil_dadt_at_nodes(
        coil.positions_m, coil.moments, shipped["transform"], _data.DIDT, context.nodes_mm
    )
    b = _assemble_rhs(context, dadt.reshape(-1, 1, 3))[solver.idx, 0].astype(cp.float64)
    n = int(solver.row_ptr.shape[0]) - 1
    a = csp.csr_matrix((solver.values, solver.col_idx, solver.row_ptr), shape=(n, n))
    iterations = 0

    def count(_) -> None:
        nonlocal iterations
        iterations += 1

    ms, (x, _) = _measure.once(
        lambda: csl.cg(
            a,
            b,
            rtol=solver.tolerance,
            atol=0.0,
            maxiter=100_000,
            M=csp.diags(1.0 / a.diagonal()),
            callback=count,
        )
    )
    v = cp.zeros(solver.n, dtype=cp.float64)
    v[solver.idx] = x
    dadt_elm = _dadt_node_to_elm(dadt, context.tet_nodes)
    fields = [_field_from_potential(context, u, dadt_elm) for u in (v, shipped["v"])]
    return {
        "iterations": iterations,
        "relative_residual": float(cp.linalg.norm(a @ x - b) / cp.linalg.norm(b)),
        "ms": ms,
        "shipped_iterations": shipped["iterations"],
        "rdm": _fields.compare(fields[0][gm], fields[1][gm], w)["rdm"],
    }


def run(ctx: _run.Context) -> dict:
    import cupy as cp

    from cunibs import Coil, Subject
    from cunibs.fem import ensure_recovery, solve_placements_block
    from cunibs.fem.solve import AggregationParams, aggregation_levels
    from cunibs.metrics import region_tag

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    angles = _data.HANDLE_ANGLE_DEG + 360.0 * np.arange(max(WIDTHS)) / max(WIDTHS)
    placements = _data.placements(subject, path, labels=("C3",), angles_deg=angles)
    placement = placements[0]
    mesh = subject.mesh
    gm = mesh.tet_tags == region_tag("gray_matter")
    w = _fields.tet_volumes(mesh.nodes_mm, mesh.tet_nodes)[gm]
    context = subject.context
    solver = context.solver
    n = 5 if ctx.quick else 30

    def timed(recovery: str = "harmonic") -> dict:
        def call():
            return subject.simulate(
                coil, placement, _data.DIDT, magnitude=True, vectors=True, recovery=recovery
            )

        samples = _measure.timed(call, n)
        result = call()
        return {"ms": _measure.stats(samples), "iterations": result.iterations, "E": result.E}

    shipped = timed()
    raw = timed("raw")
    apply_ms = shipped["ms"]["median"] - raw["ms"]["median"]

    def row(result: dict) -> dict:
        return {
            "ms": result["ms"],
            "iterations": result["iterations"],
            "rdm": _fields.compare(result["E"][gm], shipped["E"][gm], w)["rdm"],
        }

    operator = ensure_recovery(context, "harmonic")
    sites = [(p.center_mm, p.handle_mm, p.distance_mm) for p in placements]
    widths = {}
    for k in WIDTHS:

        def block(k=k):
            return solve_placements_block(
                context,
                coil.positions_m,
                coil.moments,
                sites[:k],
                _data.DIDT,
                recovery=operator,
            )

        with _measure.PeakMemory() as peak:
            samples = _measure.timed(block, n)
        widths[str(k)] = {
            "ms_per_placement": float(np.median(samples)) / k,
            "peak_mib": peak.peak_mib,
            "bitwise_equal_to_width_1": bool(cp.array_equal(block()[0]["E"], block(1)[0]["E"])),
        }
    ds_cg = _ds_cg(subject, coil, placement, gm, w)

    shapes = {}
    for rounds in ROUNDS:
        for degree in DEGREES:
            _install(solver, rounds=rounds, degree=degree)
            shapes[f"rounds{rounds}_degree{degree}"] = row(timed()) | {
                "amg_levels": solver.precond.n_levels()
            }
    solver.rebuild_preconditioner()

    levels, coarse = aggregation_levels(
        solver.row_ptr,
        solver.col_idx,
        solver.values.astype(cp.float32),
        AggregationParams(max_levels=1),
    )
    two_level = {
        "coarse_rows": int(coarse.shape[0]),
        "dense_inverse_bytes": coarse.shape[0] ** 2 * 4,
    }
    del levels, coarse
    _measure.free_pools()
    if 3 * two_level["dense_inverse_bytes"] < cp.cuda.runtime.memGetInfo()[0]:
        _install(solver, max_levels=1)
        two_level |= row(timed())
        solver.rebuild_preconditioner()
    else:
        two_level["skipped"] = "the dense coarse inverse does not fit in free device memory"

    others = [v for k, v in shapes.items() if k != "rounds2_degree2"]

    def span(xs) -> list:
        xs = list(xs)
        return [min(xs), max(xs)]

    def relative(a: dict, b: dict) -> dict:
        return {
            "ms": a["ms"]["median"] / b["ms"]["median"] - 1,
            "iterations": a["iterations"] / b["iterations"] - 1,
        }

    values = {
        "subject": sid,
        "target": "C3",
        "shipped": {"ms": shipped["ms"], "iterations": shipped["iterations"]},
        "recovery_apply": {
            "raw_ms": raw["ms"]["median"],
            "apply_ms": apply_ms,
            "share_of_raw": apply_ms / raw["ms"]["median"],
        },
        "block_width": widths,
        "ds_cg": ds_cg,
        "hierarchy_shapes": shapes,
        "alternative_shapes": {
            "iterations": span(v["iterations"] for v in others),
            "ms": span(v["ms"]["median"] for v in others),
            "max_rdm": max(v["rdm"] for v in others),
        },
        "smoother_degree": {
            str(d): relative(shapes[f"rounds2_degree{d}"], shapes["rounds2_degree2"])
            for d in DEGREES
            if d != 2
        },
        "third_round": {
            str(d): relative(shapes[f"rounds3_degree{d}"], shapes[f"rounds2_degree{d}"])
            for d in DEGREES
        },
        "two_level": two_level,
    }
    subject.free()
    return values
