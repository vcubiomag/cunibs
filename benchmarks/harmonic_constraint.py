"""The harmonic constraint ablated: the same patch fits with and without it.

Backs the recovery paper's ablation table and the text around it. The shipped harmonic operator
fixes the slots, the node patches and the frame, centred on the node and scaled by the patch's
half-width in the infinity norm. This benchmark refits the potential on exactly those patches in
numpy, with four ladders of bases: the shipped one (16 harmonic cubics, then 9 harmonic
quadratics, then linear), the same with full polynomials (20 cubics, then 10 quadratics), and
both pinned at quadratic order. Every ladder uses the kernel's pivot floor, its bound of 50 on
the gradient amplification and its fp32 weight storage, so the basis is the only difference.
A full cubic needs 25 patch nodes against the harmonic cubic's 20, the same 1.25 margin.

The constrained refit must reproduce the shipped weights (``gate``) before the others mean
anything. ``spr_tissue``, which fits the field rather than the potential, is read from
``interface_mms`` to complete the decomposition of the gap between it and ``harmonic`` into the
change of target, the constraint and the degree. Interface orders are reported over the whole
ladder and over its three finest rungs, since the manuscript quotes the planar ones whole and
the spherical ones over the finest three.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _exact, _measure, _run

PIVOT_FLOOR = 1e-12
MAX_AMPLIFICATION = 50.0
THRESHOLDS = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
REASONS = ("taken", "patch_too_small", "not_positive_definite", "amplified", "degenerate")
_BATCH = 8192


def _linear(u, v, t):
    return [np.ones_like(u), u, v, t]


def _harmonic2(u, v, t):
    return _linear(u, v, t) + [u * u - v * v, v * v - t * t, u * v, u * t, v * t]


def _harmonic3(u, v, t):
    uu, vv, tt = u * u, v * v, t * t
    return _harmonic2(u, v, t) + [
        t * (2 * tt - 3 * uu - 3 * vv), u * (4 * tt - uu - vv), v * (4 * tt - uu - vv),
        t * (uu - vv), u * v * t, u * (uu - 3 * vv), v * (3 * uu - vv),
    ]  # fmt: skip


def _full2(u, v, t):
    return _linear(u, v, t) + [u * u, v * v, t * t, u * v, u * t, v * t]


def _full3(u, v, t):
    uu, vv, tt = u * u, v * v, t * t
    return _full2(u, v, t) + [
        u * uu, v * vv, t * tt, uu * v, uu * t, u * vv, vv * t, u * tt, v * tt, u * v * t,
    ]  # fmt: skip


# (basis, smallest patch that may attempt it), each ladder widest first.
LADDERS = {
    "constrained": ((_harmonic3, 20), (_harmonic2, 9), (_linear, 4)),
    "unconstrained": ((_full3, 25), (_full2, 10), (_linear, 4)),
    "constrained_quadratic": ((_harmonic2, 9), (_linear, 4)),
    "unconstrained_quadratic": ((_full2, 10), (_linear, 4)),
}


def _cholesky(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched Cholesky with the kernel's pivot test, which is relative to the patch size."""
    floor = PIVOT_FLOOR * a[:, 0, 0]
    low, ok = np.zeros_like(a), np.ones(len(a), bool)
    for j in range(a.shape[1]):
        d = a[:, j, j] - np.einsum("bk,bk->b", low[:, j, :j], low[:, j, :j])
        ok &= d > floor
        low[:, j, j] = np.sqrt(np.where(ok, d, 1.0))
        col = a[:, j + 1 :, j] - np.einsum("bik,bk->bi", low[:, j + 1 :, :j], low[:, j, :j])
        low[:, j + 1 :, j] = col / low[:, j, j, None]
    return low, ok


def _gradient_rows(low: np.ndarray) -> np.ndarray:
    """Rows 1 to 3 of the inverse normal matrix, the gradient in the scaled frame."""
    n, size, _ = low.shape
    z = np.zeros((n, 3, size))
    for c in range(3):
        y = np.zeros((n, size))
        for i in range(size):
            y[:, i] = (
                float(i == c + 1) - np.einsum("bk,bk->b", low[:, i, :i], y[:, :i])
            ) / low[:, i, i]
        for i in reversed(range(size)):
            rest = np.einsum("bk,bk->b", low[:, i + 1 :, i], z[:, c, i + 1 :])
            z[:, c, i] = (y[:, i] - rest) / low[:, i, i]
    return z


def _fit(nodes_m, ptr, idx, slot_node, ladder):
    """Weights per patch entry, the rung each slot took (``len(ladder)`` for none), why the top
    rung was missed, and the top rung's amplification where it could be formed."""
    length = np.diff(ptr)
    weights = np.zeros((len(idx), 3), np.float32)
    rung = np.full(len(slot_node), len(ladder))
    reason = np.where(length < ladder[0][1], 1, 0)
    amplification = np.full(len(slot_node), np.nan)
    for level, (basis, min_patch) in enumerate(ladder):
        todo = np.flatnonzero((rung == len(ladder)) & (length >= min_patch))
        for size in np.unique(length[todo]):
            group = todo[length[todo] == size]
            for chunk in np.array_split(group, -(-len(group) // _BATCH)):
                entries = ptr[chunk, None] + np.arange(size)
                offsets = nodes_m[idx[entries]] - nodes_m[slot_node[chunk], None]
                h = np.abs(offsets).reshape(len(chunk), -1).max(axis=1)
                live = h > 0
                scaled = offsets / np.where(live, h, 1)[:, None, None]
                q = np.stack(basis(*np.moveaxis(scaled, -1, 0)), -1)
                low, spd = _cholesky(np.einsum("bln,blm->bnm", q, q))
                acc = np.einsum("bcn,bln->blc", _gradient_rows(low), q)
                amp = np.abs(acc).sum(axis=1).max(axis=1)
                ok = live & spd & (amp <= MAX_AMPLIFICATION)
                if level == 0:
                    amplification[chunk] = np.where(live & spd, amp, np.nan)
                    reason[chunk] = np.select([~live, ~spd, ~ok], [4, 2, 3], 0)
                weights[entries[ok].ravel()] = (acc[ok] / h[ok, None, None]).reshape(-1, 3)
                rung[chunk[ok]] = level
    return weights, rung, reason, amplification


def _field(weights, ptr, idx, slot_node, v) -> np.ndarray:
    """E at every slot with dA/dt zero, differenced against the slot's own node as the kernel is."""
    owner = np.repeat(np.arange(len(slot_node)), np.diff(ptr))
    part = weights.astype(np.float64) * (v[idx] - v[slot_node[owner]])[:, None]
    return -np.stack([np.bincount(owner, part[:, c], len(slot_node)) for c in range(3)], -1)


def _rung(problem) -> dict:
    import cupy as cp

    from cunibs.fem import ensure_recovery

    s = _exact.solve(problem)
    op = ensure_recovery(s.ctx, "harmonic")
    ptr, idx = cp.asnumpy(op.ptr).astype(np.int64), cp.asnumpy(op.idx).astype(np.int64)
    # The shipped fit runs in metres, since its weights differentiate and E is in V/m.
    nodes_m = s.nodes * 1e-3
    iface = s.at_interface
    row: dict = {
        "label": problem.label,
        "h_mm": problem.h_mm,
        "n_interface_slots": int(iface.sum()),
    }
    for key in ("interface", "interior", "fell_back", "top_rung_missed", "amplification"):
        row[key] = {}
    fields, rungs = {}, {}
    for name, ladder in LADDERS.items():
        weights, rung, reason, amp = _fit(nodes_m, ptr, idx, s.slot_node, ladder)
        fields[name], rungs[name] = _field(weights, ptr, idx, s.slot_node, s.v), rung
        row["interface"][name] = _exact.rel_rms(fields[name], s.truth, iface)
        row["interior"][name] = _exact.rel_rms(fields[name], s.truth, s.interior)
        row["fell_back"][name] = float((rung[iface] != 0).mean())
        row["top_rung_missed"][name] = {
            r: int((reason[iface] == i).sum()) for i, r in enumerate(REASONS)
        }
        a = amp[iface][np.isfinite(amp[iface])]
        row["amplification"][name] = {
            "min": float(a.min()),
            "median": float(np.median(a)),
            "max": float(a.max()),
        }
        if name == "unconstrained":
            row["unconstrained_rejected_at"] = {
                f"{t:g}": float((a > t).mean()) for t in THRESHOLDS
            }
        if name == "constrained":
            shipped = cp.asnumpy(op.w).astype(np.float64)
            row["gate"] = {
                "max_abs_weight_diff_over_peak": float(
                    np.abs(weights - shipped).max() / np.abs(shipped).max()
                ),
                "n_rung_disagreements": int((rung != cp.asnumpy(op.status)).sum()),
            }
    matched = iface & (rungs["unconstrained"] == 0)
    err = {
        n: _exact.rel_rms(fields[n], s.truth, matched) for n in ("constrained", "unconstrained")
    }
    row["matched_degree"] = {
        "n_slots": int(matched.sum()),
        "n_interface_slots": int(iface.sum()),
        "error": err,
        "error_ratio": err["unconstrained"] / err["constrained"],
        "whole_interface_ratio": row["interface"]["unconstrained"]
        / row["interface"]["constrained"],
    }
    del s, op, fields
    _measure.free_pools()
    return row


def _spr_tissue(ctx: _run.Context, geometry: str, rows: list[dict]) -> dict | None:
    """``interface_mms``'s ``spr_tissue`` interface errors on these rungs, if it has them all."""
    other = ctx.result("interface_mms")
    if other is None:
        return None
    theirs = {r["label"]: r for r in other["ladders"][f"{_exact.JITTER:g}"][geometry]["rungs"]}
    if any(r["label"] not in theirs for r in rows):
        return None
    return {r["label"]: theirs[r["label"]]["interface"] for r in rows}


def _arm(h, errors, fell_back=None) -> dict:
    fit = _exact.order(h, errors)
    out = {"order": fit["order"], "order_se": fit["order_se"], "finest_error": errors[-1]}
    if "tail" in fit:
        out |= {"tail_order": fit["tail"]["order"], "tail_order_se": fit["tail"]["order_se"]}
    return out if fell_back is None else out | {"fell_back": fell_back}


def _geometry(ctx: _run.Context, geometry: str) -> dict:
    rows = [_rung(p) for p in _exact.ladder(geometry, ctx.quick)]
    h = [r["h_mm"] for r in rows]
    finest = rows[-1]
    orders = {
        section: {n: _exact.order(h, [r[section][n] for r in rows]) for n in LADDERS}
        for section in ("interface", "interior")
    }
    arms = {
        n: _arm(h, [r["interface"][n] for r in rows], finest["fell_back"][n]) for n in LADDERS
    }
    reference = _spr_tissue(ctx, geometry, rows)
    if reference is not None:
        arms["spr_tissue"] = _arm(h, [reference[r["label"]]["spr_tissue"] for r in rows])
    error = {n: a["finest_error"] for n, a in arms.items()}
    steps = (
        ("target", "spr_tissue", "unconstrained_quadratic"),
        ("constraint", "unconstrained_quadratic", "constrained_quadratic"),
        ("degree", "constrained_quadratic", "constrained"),
        ("degree_unconstrained", "unconstrained_quadratic", "unconstrained"),
        ("constraint_at_cubic", "unconstrained", "constrained"),
    )
    con, unc = finest["amplification"]["constrained"], finest["amplification"]["unconstrained"]
    out = {
        "rungs": rows,
        "orders": orders,
        "arms": arms,
        "error_ratio": {s: error[a] / error[b] for s, a, b in steps if a in error},
        "separation": {
            "max_constrained": con["max"],
            "min_unconstrained": unc["min"],
            "disjoint": unc["min"] > con["max"],
        },
        "gate": {
            "max_abs_weight_diff_over_peak": max(
                r["gate"]["max_abs_weight_diff_over_peak"] for r in rows
            ),
            "n_rung_disagreements": sum(r["gate"]["n_rung_disagreements"] for r in rows),
        },
        "max_interface_patch_too_small": max(
            r["top_rung_missed"][n]["patch_too_small"] for r in rows for n in LADDERS
        ),
    }
    if reference is not None:
        out["max_rel_diff_from_interface_mms_harmonic"] = max(
            abs(r["interface"]["constrained"] / reference[r["label"]]["harmonic"] - 1)
            for r in rows
        )
    return out


def run(ctx: _run.Context) -> dict:
    return {g: _geometry(ctx, g) for g in ("planar", "sphere")}
