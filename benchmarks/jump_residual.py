"""How far each recovery rule violates the interface conditions, on known solutions and on heads.

Backs the recovery paper's interface-condition results and its jump table. Across a jump in
conductivity the normal component of ``sigma E`` and the tangential component of ``E`` are
continuous. No rule imposes either, so the residual in them measures a rule against physics and
needs no exact solution. At each node on a face between tissues A and B, the area-weighted normal
``n`` of those faces is formed and

    r_n = (sigma_A E_A - sigma_B E_B) . n        s_n = (sigma_A E_A + sigma_B E_B) . n / 2
    r_t = |P (E_A - E_B)|                        s_t = |P (E_A + E_B) / 2|,  P = I - n n^T.

A residual is ``|r| / sqrt(mean s^2)`` over the scored node pairs, summarised by its median (and,
on the manufactured problems, by ``|r|``'s RMS over the same scale). Outer-surface nodes are left
out. ``raw`` is scored on each interface face from its two tetrahedra; ``spr_global`` has one
value per node, so its tangential residual is zero by construction.

The residual is first checked against the true interface error on the manufactured ladders of
``_exact``, where it should order the rules as the true error does. On the heads, each node pair
is also grouped by how many tissues meet at the node and by the coherence deficit
``1 - |sum A_f n_f| / sum A_f`` of its faces, which is zero on a flat interface; each group is
normalised by its own scale.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from benchmarks import _data, _exact, _measure, _run

NODAL = ("harmonic", "spr_tissue", "spr_global")
DEFICIT_EDGES = (1e-3, 1e-2, 1e-1)
DEFICIT_BINS = ("(0, 0.001]", "(0.001, 0.01]", "(0.01, 0.1]", "> 0.1")
_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
_TAGS = 64


def _geometry(nodes, tets, tags, on_surface, slot_node, slot_tag) -> dict:
    """Interface faces, and per (node, tissue pair) the normal, deficit and tissue count."""
    faces = np.sort(tets[:, _FACES].reshape(-1, 3), axis=1).astype(np.int64)
    key = (faces[:, 0] << 42) | (faces[:, 1] << 21) | faces[:, 2]
    order = np.argsort(key, kind="stable")
    shared = np.flatnonzero(key[order][1:] == key[order][:-1])
    a, b = order[shared] // 4, order[shared + 1] // 4
    mixed = tags[a] != tags[b]
    a, b, tri = a[mixed], b[mixed], faces[order[shared[mixed]]]
    lo, hi = np.where(tags[a] < tags[b], a, b), np.where(tags[a] < tags[b], b, a)
    p = nodes[tri]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = np.linalg.norm(cross, axis=1)
    normal = cross / area[:, None]
    step = nodes[tets[hi]].mean(1) - nodes[tets[lo]].mean(1)
    normal *= np.sign(np.einsum("ij,ij->i", normal, step))[:, None]

    corner = tri.ravel() * _TAGS**2 + np.repeat(tags[lo] * _TAGS + tags[hi], 3)
    pair, inverse = np.unique(corner, return_inverse=True)
    weights = np.repeat(area, 3)
    total = np.bincount(inverse, weights)
    vector = np.stack(
        [np.bincount(inverse, weights * np.repeat(normal[:, c], 3)) for c in range(3)], -1
    )
    length = np.linalg.norm(vector, axis=1)
    node = pair // _TAGS**2
    return {
        "face": {"lo": lo, "hi": hi, "normal": normal, "lo_tag": tags[lo], "hi_tag": tags[hi]},
        "node": node,
        "lo_tag": pair // _TAGS % _TAGS,
        "hi_tag": pair % _TAGS,
        "normal": vector / length[:, None],
        "keep": ~on_surface[node],
        "tissues": np.bincount(slot_node, minlength=len(nodes))[node],
        "deficit_bin": np.searchsorted(DEFICIT_EDGES, 1 - length / total),
        "slot_key": slot_node * _TAGS + slot_tag,
    }


def _residuals(e_lo, e_hi, lo_tag, hi_tag, n) -> tuple[np.ndarray, ...]:
    j_lo = _exact.sigma_of(lo_tag) * np.einsum("ij,ij->i", e_lo, n)
    j_hi = _exact.sigma_of(hi_tag) * np.einsum("ij,ij->i", e_hi, n)

    def tangential(v):
        return np.linalg.norm(v - np.einsum("ij,ij->i", v, n)[:, None] * n, axis=1)

    return (
        j_lo - j_hi,
        (j_lo + j_hi) / 2,
        tangential(e_lo - e_hi),
        tangential((e_lo + e_hi) / 2),
    )


def _summary(r: np.ndarray, s: np.ndarray) -> dict[str, float]:
    scale = np.sqrt(np.mean(s * s))
    return {
        "rel_median": float(np.median(np.abs(r)) / scale),
        "rel_rms": float(np.sqrt(np.mean(r * r)) / scale),
    }


def _score(g: dict, e: np.ndarray, mode: str, strata: bool = False) -> dict:
    """Both residuals for ``e``: per element for ``raw``, otherwise on the tissue slots."""
    if mode == "raw":
        f = g["face"]
        rn, sn, rt, st = _residuals(
            e[f["lo"]], e[f["hi"]], f["lo_tag"], f["hi_tag"], f["normal"]
        )
        return {"normal_current": _summary(rn, sn), "tangential_field": _summary(rt, st)}
    e_lo = e[np.searchsorted(g["slot_key"], g["node"] * _TAGS + g["lo_tag"])]
    e_hi = e[np.searchsorted(g["slot_key"], g["node"] * _TAGS + g["hi_tag"])]
    rn, sn, rt, st = _residuals(e_lo, e_hi, g["lo_tag"], g["hi_tag"], g["normal"])

    def block(sel):
        sel = sel & g["keep"]
        return {
            "normal_current": _summary(rn[sel], sn[sel]),
            "tangential_field": _summary(rt[sel], st[sel]),
        }

    out = block(np.ones_like(g["keep"]))
    if strata:
        out["by_tissue_count"] = {
            "two": block(g["tissues"] == 2),
            "three_or_more": block(g["tissues"] >= 3),
        }
        out["by_coherence_deficit"] = {
            b: block(g["deficit_bin"] == i) for i, b in enumerate(DEFICIT_BINS)
        }
        out["frac_three_or_more"] = float((g["tissues"][g["keep"]] >= 3).mean())
    return out


def _manufactured(geometry: str, quick: bool) -> dict:
    rows = []
    for problem in _exact.ladder(geometry, quick):
        s = _exact.solve(problem)
        g = _geometry(s.nodes, s.tets, s.tags, s.on_surface, s.slot_node, s.slot_tag)
        fields = {m: s.recover(m) for m in NODAL}
        rows.append(
            {
                "label": problem.label,
                "h_mm": problem.h_mm,
                "residual": {
                    m: _score(g, f, m)["normal_current"]["rel_rms"] for m, f in fields.items()
                }
                | {"exact": _score(g, s.truth, "exact")["normal_current"]["rel_rms"]}
                | {"raw": _score(g, s.e_raw, "raw")["normal_current"]["rel_rms"]},
                "error": {
                    m: _exact.rel_rms(f, s.truth, s.at_interface) for m, f in fields.items()
                },
            }
        )
        del s, fields
        _measure.free_pools()
    agree = [
        (r["residual"][a] < r["residual"][b]) == (r["error"][a] < r["error"][b])
        for r in rows
        for a, b in combinations(NODAL, 2)
    ]
    ratio = [r["residual"][m] / r["error"][m] for r in rows for m in NODAL]
    return {
        "rungs": rows,
        "orderings_agreeing": sum(agree),
        "orderings": len(agree),
        "residual_over_error": [min(ratio), max(ratio)],
        "exact_floor": [r["residual"]["exact"] for r in rows],
    }


def _subject(sid: str, path, coil) -> list[dict]:
    from cunibs import Subject

    subject = Subject.from_mesh(path)
    placements = _data.placements(subject, path)
    cases = [{"subject": sid, "target": t} for t in _data.TARGETS]
    mesh, g = subject.mesh, None
    for mode in (*NODAL, "raw"):
        nodal = mode != "raw"
        results = subject.iter_simulate(
            coil, placements, _data.DIDT, recovery=mode, vectors=not nodal, nodal=nodal
        )
        for case, r in zip(cases, results, strict=True):
            if g is None:
                on_surface = _surface(subject)
                tets = mesh.tet_nodes.astype(np.int64)
                slots = r.slot_node.astype(np.int64), r.slot_tag.astype(np.int64)
                g = _geometry(
                    mesh.nodes_mm, tets, mesh.tet_tags.astype(np.int64), on_surface, *slots
                )
            if not nodal:
                e = r.E.astype(np.float64)
            elif r.slot_tag is None:
                e = r.E_slots.astype(np.float64)[g["slot_key"] // _TAGS]
            else:
                e = r.E_slots.astype(np.float64)
            case[mode] = _score(g, e, mode, strata=nodal)
    subject.free()
    return cases


def _surface(subject) -> np.ndarray:
    import cupy as cp

    from cunibs.fem import outer_boundary_nodes

    ctx = subject.context
    return cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, ctx.n_nodes)).astype(bool)


def _subject_ci(values: np.ndarray, groups: np.ndarray, n: int = 10_000) -> list[float]:
    """95% bootstrap interval of the median, resampling subjects rather than cases."""
    rng = np.random.default_rng(0)
    by_group = [values[groups == g] for g in np.unique(groups)]
    medians = [
        np.median(
            np.concatenate([by_group[i] for i in rng.integers(0, len(by_group), len(by_group))])
        )
        for _ in range(n)
    ]
    return [float(v) for v in np.percentile(medians, [2.5, 97.5])]


def _cohort(cases: list[dict]) -> dict:
    groups = np.array([c["subject"] for c in cases])

    def column(mode, field, *path):
        out = []
        for c in cases:
            node = c[mode]
            for key in path:
                node = node[key]
            out.append(node[field]["rel_median"])
        return np.array(out)

    out: dict = {}
    for field in ("normal_current", "tangential_field"):
        out[field] = {}
        for mode in (*NODAL, "raw"):
            v = column(mode, field)
            out[field][mode] = {"median": float(np.median(v)), "ci95": _subject_ci(v, groups)}
    harmonic = column("harmonic", "normal_current")
    out["ratio_to_harmonic"] = {
        m: dict(
            zip(
                ("q1", "median", "q3"),
                np.percentile(column(m, "normal_current") / harmonic, [25, 50, 75]).tolist(),
                strict=True,
            )
        )
        for m in ("spr_tissue", "spr_global", "raw")
    }
    strata = {"by_tissue_count": ("two", "three_or_more"), "by_coherence_deficit": DEFICIT_BINS}
    out["strata"] = {
        group: {
            m: {
                b: {
                    f: float(np.median(column(m, f, group, b)))
                    for f in ("normal_current", "tangential_field")
                }
                for b in bins
            }
            for m in NODAL
        }
        for group, bins in strata.items()
    }
    deficit = out["strata"]["by_coherence_deficit"]
    out["harmonic_vs_spr_tissue_by_deficit"] = {
        b: deficit["harmonic"][b]["normal_current"] / deficit["spr_tissue"][b]["normal_current"]
        - 1
        for b in DEFICIT_BINS
    }
    out["frac_three_or_more"] = float(
        np.median([c["harmonic"]["frac_three_or_more"] for c in cases])
    )
    return out


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil

    mms = {g: _manufactured(g, ctx.quick) for g in ("planar", "sphere")}
    coil = Coil.load(_data.COIL)
    subjects = ctx.data.subjects[:2] if ctx.quick else ctx.data.subjects
    cases = [c for sid in subjects for c in _subject(sid, ctx.data.mesh(sid), coil)]
    cohort = _cohort(cases)
    floor = {r["h_mm"]: r["residual"]["exact"] for r in mms["sphere"]["rungs"]}
    if 1.25 in floor:
        cohort["sphere_floor_at_1.25mm_over_harmonic"] = (
            floor[1.25] / cohort["normal_current"]["harmonic"]["median"]
        )
    return {"coil": _data.COIL, "manufactured": mms, "cases": cases, "cohort": cohort}
