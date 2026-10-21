"""What a conductivity draw costs, against rebuilding everything per draw.

Backs fig_uq(a) and the conductivity rows of the cuNIBS cost table. An ensemble of each size is
run end to end through ``run_conductivity_uq`` with harmonic recovery, so the per-placement setup
is spread over the draws. The baseline does in this library what a draw costs without reuse:
assemble the stiffness matrix at the draw's conductivities, build its hierarchy and RHS operator,
then solve, reconstruct and recover the field.

SimNIBS has no conductivity-sampling mode, so its draw is a full rebuild and solve, as timed by
``simnibs_timing``. The hierarchy is built once at nominal conductivity; the number of draws that
needed one of their own is counted, next to the largest conductivity contrast within a draw.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from benchmarks import _data, _measure, _objective, _run

N_SWEEP = (50, 100, 200, 500, 1000)
NAIVE_DRAWS = 10
ENSEMBLE = 500


def _naive_draw(subject, coil, placement, operator, tags):
    """A function solving one draw with nothing reused but the geometry and recovery operator."""
    import cupy as cp

    # Internal: the assembly and solver setup a draw-specific operator needs.
    from cunibs.fem import solve_placements_block
    from cunibs.fem.assembly import TISSUE_CONDUCTIVITY, assemble_stiffness, gradient_operator
    from cunibs.fem.solve import build_rhs_operator, ground_node_of, prepare_grounded_solver

    ctx = subject.context
    g, vols = gradient_operator(ctx.nodes_mm, ctx.tet_nodes)
    ground = ground_node_of(ctx.nodes_mm)
    nominal = np.zeros(max(TISSUE_CONDUCTIVITY) + 1)
    for tag, s in TISSUE_CONDUCTIVITY.items():
        nominal[tag] = s
    site = (placement.center_mm, placement.handle_mm, placement.distance_mm)

    def draw(sigma):
        lut = nominal.copy()
        lut[list(tags)] = sigma
        cond = cp.asarray(lut)[ctx.tet_tags]
        a = assemble_stiffness(g, vols, cond, ctx.n_nodes, ctx.tet_nodes)
        rebuilt = dataclasses.replace(
            ctx,
            solver=prepare_grounded_solver(a, ctx.nodes_mm, ground),
            rhs_op=build_rhs_operator(
                a, g, -(vols * cond), ctx.tet_nodes, ctx.node2corner_ptr, ctx.node2corner_idx
            ),
            rhs_indptr=a.indptr,
        )
        solve_placements_block(
            rebuilt, coil.positions_m, coil.moments, [site], _data.DIDT, recovery=operator
        )

    return draw


def run(ctx: _run.Context) -> dict:
    import cupy as cp

    from cunibs import Coil, Subject
    from cunibs.adm import resolve_target
    from cunibs.fem import ensure_recovery
    from cunibs.uq import (
        ConductivityUQConfig,
        build_conductivity_uq_precompute,
        run_conductivity_uq,
        sample_conductivities,
    )

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    roi = resolve_target(subject, _objective.cortical_target(subject, placement))
    operator = ensure_recovery(subject.context, "harmonic")
    tags = tuple(int(t) for t in np.unique(subject.mesh.tet_tags))
    precompute_ms, pre = _measure.once(
        lambda: build_conductivity_uq_precompute(subject.context, tags)
    )

    def ensemble(config):
        run_conductivity_uq(
            subject.context, pre, coil, placement, config,
            didt=_data.DIDT, record_rois={"roi": roi}, recovery=operator,
        )  # fmt: skip

    ensemble(ConductivityUQConfig(n_samples=8, seed=1))
    rebuilds = []
    # The library does not report fallbacks; this hook on our own precompute is where a draw
    # that missed tolerance gets a hierarchy of its own.
    pre.preconditioner_for = lambda data, f=pre.preconditioner_for: (
        rebuilds.append(1) or f(data)
    )

    sweep, contrast = [], []
    for n in (10, 20) if ctx.quick else N_SWEEP:
        config = ConductivityUQConfig(n_samples=n)
        ms, _ = _measure.once(lambda c=config: ensemble(c))
        sigmas = cp.asnumpy(sample_conductivities(config, pre.perturbed_tags))
        sweep.append({"n": n, "wall_s": ms / 1e3, "ms_per_draw": ms / n})
        contrast.append(float((sigmas.max(axis=1) / sigmas.min(axis=1)).max()))
        print(f"{n:>5} draws: {ms / n:.2f} ms/draw", flush=True)
    per_draw_ms = min(s["ms_per_draw"] for s in sweep)
    nominal = cp.asnumpy(pre.nominal_sigma)

    draws = cp.asnumpy(
        sample_conductivities(ConductivityUQConfig(NAIVE_DRAWS + 1, seed=2), pre.perturbed_tags)
    )
    naive_draw = _naive_draw(subject, coil, placement, operator, pre.perturbed_tags)
    naive_draw(draws[0])
    naive = [
        _measure.once(lambda s=s: naive_draw(s))[0]
        for s in draws[1 : (2 if ctx.quick else NAIVE_DRAWS) + 1]
    ]

    values = {
        "subject": sid,
        "coil": _data.COIL,
        "precompute_ms": precompute_ms,
        "per_draw": sweep,
        "ms_per_draw": per_draw_ms,
        "naive_ms_per_draw": _measure.stats(naive),
        "speedup_vs_naive": float(np.median(naive)) / per_draw_ms,
        "hierarchy_rebuilds": {"n": len(rebuilds), "draws": sum(s["n"] for s in sweep)},
        "contrast": {"nominal": float(nominal.max() / nominal.min()), "max": max(contrast)},
    }

    timing = ctx.latest("simnibs_timing")
    if timing and timing.get("uq"):
        case = min(timing["uq"], key=lambda c: timing["uq"][c]["draw_s"])
        best = timing["uq"][case]
        # The 500-draw ensemble when it ran, else the largest that did.
        n_ens = next((s for s in sweep if s["n"] == ENSEMBLE), sweep[-1])
        values["simnibs"] = {k: best[k] for k in ("solver", "threads", "draw_s")} | {
            "case": case,
            "machine": timing["machine"],
        }
        values["speedup_vs_simnibs"] = {
            "draw": best["draw_s"] * 1e3 / per_draw_ms,
            "ensemble": {
                "n": n_ens["n"],
                "ratio": best["draw_s"] * n_ens["n"] / n_ens["wall_s"],
            },
        }

    # OnlineFEM factorises the conductivities in, so a draw costs it a new setup.
    onlinefem = ctx.latest("simnibs_onlinefem")
    if onlinefem and onlinefem.get("uq"):
        values["onlinefem"] = onlinefem["uq"] | {"machine": onlinefem["machine"]}
        values["speedup_vs_onlinefem"] = {"draw": onlinefem["uq"]["draw_s"] * 1e3 / per_draw_ms}
    return values
