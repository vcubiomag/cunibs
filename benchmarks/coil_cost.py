"""What the coil model adds to the cost of a placement, over every bundled coil.

Backs the coil paragraph of the cuNIBS cost section: the range of dipole counts, the range of
placement times, the share of a placement spent evaluating the source, and how that cost grows
per dipole.

The coil enters a placement only through dA/dt at the mesh nodes, a direct sum over its dipoles,
so that is the only cost expected to move with the coil. Each coil is placed at C3 on one mesh and
timed twice: a whole placement through ``Subject.simulate``, and dA/dt alone through
``fem.coil_dadt_at_nodes`` at the same coil frame. Both are fitted as lines in the dipole count.
"""

from __future__ import annotations

from benchmarks import _data, _fields, _measure, _run


def run(ctx: _run.Context) -> dict:
    import cupy as cp

    from cunibs import Coil, Subject, bundled_coils
    from cunibs.fem import coil_dadt_at_nodes

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    frame = cp.asarray(_data.frames(subject, [placement])[0])
    nodes = subject.context.nodes_mm
    n_tets = subject.mesh.n_tets
    names = sorted(bundled_coils())
    n = 10 if ctx.quick else 50

    rows = []
    for name in names[:4] if ctx.quick else names:
        coil = Coil.load(name)
        whole = _measure.timed(lambda c=coil: subject.simulate(c, placement, _data.DIDT), n)
        dadt = _measure.timed(
            lambda c=coil: coil_dadt_at_nodes(
                c.positions_m, c.moments, frame, _data.DIDT, nodes
            ),
            n,
        )
        rows.append(
            {
                "coil": name,
                "n_dipoles": coil.n_dipoles,
                "placement_ms": _measure.stats(whole),
                "dadt_ms": _measure.stats(dadt),
            }
        )
    subject.free()

    dipoles = [r["n_dipoles"] for r in rows]
    placement_ms = [r["placement_ms"]["median"] for r in rows]
    dadt_ms = [r["dadt_ms"]["median"] for r in rows]
    share = [d / p for d, p in zip(dadt_ms, placement_ms, strict=True)]
    dadt_fit = _fields.fit_line(dipoles, dadt_ms)
    return {
        "subject": sid,
        "n_tets": n_tets,
        "n_coils": len(rows),
        "coils": rows,
        "n_dipoles": {"min": min(dipoles), "max": max(dipoles)},
        "placement_ms": {"min": min(placement_ms), "max": max(placement_ms)},
        "dadt_share": {"min": min(share), "max": max(share)},
        "dadt_fit": dadt_fit | {"us_per_dipole": dadt_fit["slope"] * 1e3},
        "placement_fit": _fields.fit_line(dipoles, placement_ms),
    }
