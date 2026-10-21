"""Convergence of each recovery rule at a planar and a spherical conductivity interface.

Backs the recovery paper's convergence table and figure and the text around them. Both
manufactured problems of ``_exact`` are solved on five-rung ladders at node jitters of 0.25, 0.1
and 0, and every rule is applied to the same P1 solution.

``spr_global``, ``spr_tissue`` and ``harmonic`` are scored on the (node, tissue) slots against
the exact field from each slot's own side. ``spr_global`` has one value per node, so at an
interface it reports that value for both tissues. ``raw`` has no nodal value and is scored at
element barycentres. Slots on the outer surface, and elements touching it, are left out because
every rule treats them alike. Errors are relative RMS, and orders are log-log slopes against
the element size with their standard errors.

At jitter 0.25, SimNIBS's ``elm_data2node_data`` is scored on the same slots from the same raw
field.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _exact, _measure, _run

MODES = ("spr_global", "spr_tissue", "harmonic")
JITTERS = (0.25, 0.1, 0.0)
GEOMETRIES = ("planar", "sphere")


def _raw_errors(s: _exact.Solved) -> dict[str, float]:
    bary = s.nodes[s.tets].mean(axis=1)
    truth = np.zeros_like(bary)
    for tag in (_exact.INNER, _exact.OUTER):
        sel = s.tags == tag
        truth[sel] = s.problem.field(bary[sel], tag)
    surface = s.on_surface[s.tets].any(axis=1)
    interface = s.problem.on_interface(s.nodes)[s.tets].any(axis=1) & ~surface
    return {
        "interface": _exact.rel_rms(s.e_raw, truth, interface),
        "interior": _exact.rel_rms(s.e_raw, truth, ~surface & ~interface),
    }


def _rung(problem, simnibs) -> dict:
    import cupy as cp

    from cunibs.fem import ensure_recovery

    s = _exact.solve(problem)
    fields = {m: s.recover(m) for m in MODES}
    if simnibs is not None:
        response = simnibs.call(
            "node_recovery",
            {},
            nodes=s.nodes,
            tets=s.tets,
            tet_tags=s.tags,
            fields=s.e_raw[None],
        )
        if response is not None:
            fields["simnibs"] = response[1]["nodal"][0][s.slot_node]
    status = cp.asnumpy(ensure_recovery(s.ctx, "harmonic").status)
    raw = _raw_errors(s)
    row = {
        "label": problem.label,
        "h_mm": problem.h_mm,
        "n_nodes": len(s.nodes),
        "n_tets": len(s.tets),
        "n_slots": len(s.slot_node),
        "n_interface_slots": int(s.at_interface.sum()),
        "solve_residual": s.solve_residual,
        "interface": {m: _exact.rel_rms(f, s.truth, s.at_interface) for m, f in fields.items()}
        | {"raw": raw["interface"]},
        "interior": {m: _exact.rel_rms(f, s.truth, s.interior) for m, f in fields.items()}
        | {"raw": raw["interior"]},
        "harmonic_rung": {
            name: int((status == i).sum())
            for i, name in enumerate(("cubic", "quadratic", "linear", "undetermined"))
        }
        | {"quadratic_at_interface": int((status[s.at_interface] == 1).sum())},
    }
    del s, fields
    _measure.free_pools()
    return row


def _ladder(geometry: str, jitter: float, ctx: _run.Context) -> dict:
    simnibs = ctx.simnibs if jitter == _exact.JITTER else None
    rows = [_rung(p, simnibs) for p in _exact.ladder(geometry, ctx.quick, jitter)]
    h = [r["h_mm"] for r in rows]
    orders = {
        section: {m: _exact.order(h, [r[section][m] for r in rows]) for m in rows[0][section]}
        for section in ("interface", "interior")
    }
    return {"rungs": rows, "orders": orders}


def _summary(ladders: dict) -> dict:
    shipped, regular = ladders[f"{_exact.JITTER:g}"], ladders["0"]
    out: dict = {
        "spr_global_interface_error": {
            g: [shipped[g]["rungs"][i]["interface"]["spr_global"] for i in (0, -1)]
            for g in GEOMETRIES
        },
        "spr_tissue_over_harmonic_finest": {
            g: shipped[g]["rungs"][-1]["interface"]["spr_tissue"]
            / shipped[g]["rungs"][-1]["interface"]["harmonic"]
            for g in GEOMETRIES
        },
        "regular_planar_finest_interface": {
            m: regular["planar"]["rungs"][-1]["interface"][m]
            for m in ("harmonic", "spr_tissue")
        },
        "spr_tissue_interface_order_by_jitter": {
            g: {j: ladders[j][g]["orders"]["interface"]["spr_tissue"]["order"] for j in ladders}
            for g in GEOMETRIES
        },
        "max_order_change_jitter_0_vs_shipped": {
            g: max(
                abs(shipped[g]["orders"][s][m]["order"] - regular[g]["orders"][s][m]["order"])
                for s in ("interface", "interior")
                for m in (*MODES, "raw")
            )
            for g in GEOMETRIES
        },
        "harmonic_quadratic_fallback": {
            g: [
                {k: r[k] for k in ("label", "n_slots", "n_interface_slots")}
                | {k: r["harmonic_rung"][k] for k in ("quadratic", "quadratic_at_interface")}
                for r in shipped[g]["rungs"]
            ]
            for g in GEOMETRIES
        },
    }
    with_simnibs = [
        r for g in GEOMETRIES for r in shipped[g]["rungs"] if "simnibs" in r["interface"]
    ]
    if with_simnibs:
        out["simnibs_vs_spr_global_max_rel_diff"] = max(
            abs(r["interface"]["simnibs"] / r["interface"]["spr_global"] - 1)
            for r in with_simnibs
        )
    return out


def run(ctx: _run.Context) -> dict:
    ladders = {f"{j:g}": {g: _ladder(g, j, ctx) for g in GEOMETRIES} for j in JITTERS}
    shipped = ladders[f"{_exact.JITTER:g}"]
    return {
        "ladders": ladders,
        "summary": _summary(ladders),
        "figure": {
            g: [{"h_mm": r["h_mm"]} | r["interface"] for r in shipped[g]["rungs"]]
            for g in GEOMETRIES
        },
    }
