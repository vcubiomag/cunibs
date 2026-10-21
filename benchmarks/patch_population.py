"""How much of a head mesh sits at a tissue interface, and how often each rule stops fitting there.

Backs the recovery-paper results that over the cohort a median 62.6% of nodes belong to more than
one tissue, 93.3% of gray-matter tetrahedra have at least one such node, 1.1% of nodes belong to
three or more tissues, ``spr_tissue`` uses its one-sided average for 81.4% of slots and
``harmonic`` falls back to a lower rung for 1.5e-4 of slots. Each value is the median over subjects
of a per-subject fraction.

A slot is a (node, tissue) pair, and a node's tissue count is how many slots it has. ``spr_tissue``
replaces the fit with the tissue's volume-weighted average at every slot on the boundary of its
own tissue, which includes every mixed node and the outer surface; that is the rule, not a
rejection. ``harmonic`` has no boundary rule, so each of its fallbacks is a rejected fit.
Nothing here solves a placement: these are properties of the mesh and the recovery operators.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _fields, _measure, _run

GRAY_MATTER = 2


def _subject(path) -> dict:
    import cupy as cp

    from cunibs import Subject
    from cunibs.fem import ensure_recovery, tissue_boundary_nodes

    subject = Subject.from_mesh(path)
    ctx = subject.context
    tissue = ensure_recovery(ctx, "spr_tissue")
    harmonic = ensure_recovery(ctx, "harmonic")
    tags_per_node = np.bincount(cp.asnumpy(tissue.slot_node), minlength=ctx.n_nodes)
    boundary = cp.asnumpy(tissue_boundary_nodes(ctx.tet_nodes, ctx.n_nodes, tissue.slot_node))
    tet_nodes, tet_tags = subject.mesh.tet_nodes, subject.mesh.tet_tags
    gm = tet_tags == GRAY_MATTER
    row = {
        "n_nodes": int(ctx.n_nodes),
        "n_tets": int(tet_tags.size),
        "n_slots": int(tissue.n_slots),
        "nodes_mixed": float(np.mean(tags_per_node > 1)),
        "nodes_three_or_more": float(np.mean(tags_per_node > 2)),
        "nodes_tissue_boundary": float(np.mean(boundary > 0)),
        "tissues_per_node": {
            str(k): int(n) for k, n in enumerate(np.bincount(tags_per_node)) if n
        },
        "gm_tets_touching_mixed": float(
            np.mean((tags_per_node[tet_nodes[gm]] > 1).any(axis=1))
        ),
        "gm_tets_touching_three_or_more": float(
            np.mean((tags_per_node[tet_nodes[gm]] > 2).any(axis=1))
        ),
        "spr_tissue_average": tissue.n_fallback / tissue.n_slots,
        "harmonic_fallback": harmonic.n_fallback / harmonic.n_slots,
        "harmonic_undetermined": int(harmonic.n_undetermined),
    }
    subject.free()
    return row


def run(ctx: _run.Context) -> dict:
    subjects = ctx.data.subjects[:2] if ctx.quick else ctx.data.subjects
    per_subject = {}
    for sid in subjects:
        per_subject[sid] = _subject(ctx.data.mesh(sid))
        _measure.free_pools()
    keys = [k for k, v in next(iter(per_subject.values())).items() if isinstance(v, float)]
    return {
        "subjects": per_subject,
        "cohort": {
            k: _fields.quantiles([row[k] for row in per_subject.values()]) for k in keys
        },
    }
