"""The single-precision coil source term against float64 references, for every bundled coil.

Backs the source-field paragraph of the cuNIBS paper's verification results. cuNIBS sums each
coil's dipoles at the mesh nodes in float32. Each bundled coil is placed at C3 on the reference
subject, and its dA/dt is compared at the nodes with two references: the same dipole sum in
float64 on the host, at 20,000 sampled nodes, which checks the arithmetic; and SimNIBS's
``get_da_dt`` on the same coil file, nodes and frame at every node, which also checks the coil
model and the transform. Errors are the largest nodal difference over the reference's peak.

The coils differ in how many dipoles they have and in how close their nearest dipole comes to
the head, and the error is fitted against both to see which one sets it.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields, _run
from benchmarks._simnibs import mesh_arrays

SAMPLED_NODES = 20_000
FLOOR = 2e-5
QUICK_COILS = ("MagVenture_C-B70", "MagVenture_Cool-D-B80", "MagStim_D70")


def _dipole_sum(nodes_mm, positions_mm, moments, didt) -> np.ndarray:
    r = nodes_mm * 1e-3
    out = np.zeros_like(r)
    for s, m in zip(positions_mm * 1e-3, moments, strict=True):
        d = r - s
        out += np.cross(m, d) / (np.linalg.norm(d, axis=1) ** 3)[:, None]
    return out * 1e-7 * didt


def _nearest_mm(nodes, points_mm) -> float:
    """Distance from the nearest point to the nearest node, as |n|^2 + |p|^2 - 2 n.p per chunk."""
    import cupy as cp

    nn = (nodes * nodes).sum(axis=1)[:, None]
    best = np.inf
    for chunk in np.array_split(points_mm, -(-len(points_mm) // 64)):
        p = cp.asarray(chunk)
        d2 = nn + (p * p).sum(axis=1) - 2 * nodes @ p.T
        best = min(best, float(d2.min()))
    return float(np.sqrt(max(best, 0.0)))


def _coil(ctx, subject, name, frame, sample, mesh) -> dict:
    import cupy as cp

    from cunibs import Coil
    from cunibs.fem import coil_dadt_at_nodes

    coil = Coil.load(name)
    nodes = subject.context.nodes_mm
    got = cp.asnumpy(
        coil_dadt_at_nodes(coil.positions_m, coil.moments, frame, _data.DIDT, nodes)
    )
    positions = coil.positions_m * 1e3 @ frame[:3, :3].T + frame[:3, 3]
    exact = _dipole_sum(
        subject.mesh.nodes_mm[sample], positions, coil.moments @ frame[:3, :3].T, _data.DIDT
    )
    row = {
        "coil": name,
        "n_dipoles": coil.n_dipoles,
        "min_dipole_to_node_mm": _nearest_mm(nodes, positions),
        "vs_float64": float(
            np.linalg.norm(got[sample] - exact, axis=1).max()
            / np.linalg.norm(exact, axis=1).max()
        ),
    }
    response = ctx.simnibs.call(
        "dadt", {"coil": name, "didt": _data.DIDT}, frames=frame[None], **mesh
    )
    if response is not None:
        theirs = response[1]["dadt"][0]
        row["vs_simnibs"] = float(
            np.linalg.norm(got - theirs, axis=1).max() / np.linalg.norm(theirs, axis=1).max()
        )
    return row


def run(ctx: _run.Context) -> dict:
    from cunibs import Subject, bundled_coils

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    frame = _data.frames(subject, _data.placements(subject, path, labels=("C3",)))[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(subject.mesh.n_nodes, SAMPLED_NODES, replace=False)
    mesh = mesh_arrays(subject.mesh) if ctx.simnibs.available else {}
    names = QUICK_COILS if ctx.quick else tuple(bundled_coils())
    rows = [_coil(ctx, subject, name, frame, sample, mesh) for name in names]
    subject.free()

    errors = np.array([r["vs_float64"] for r in rows])
    distance = np.array([r["min_dipole_to_node_mm"] for r in rows])
    floor = errors <= FLOOR
    values = {
        "subject": sid,
        "target": "C3",
        "n_sampled_nodes": SAMPLED_NODES,
        "coils": rows,
        "worst_vs_float64": float(errors.max()),
        "floor": {
            "threshold": FLOOR,
            "n_coils": int(floor.sum()),
            "of": len(rows),
            "median_all_coils": float(np.median(errors)),
            "median_at_floor": float(np.median(errors[floor])) if floor.any() else None,
            "worst_with_dipoles_beyond_4mm": float(errors[distance > 4].max(initial=0.0)),
        },
        "above_floor": [
            {k: r[k] for k in ("coil", "vs_float64", "min_dipole_to_node_mm")}
            for r in rows
            if r["vs_float64"] > FLOOR
        ],
    }
    if len(rows) >= 3:
        values["fit_log_error"] = {
            "vs_min_dipole_to_node": _fields.fit_power(distance, errors),
            "vs_dipole_count": _fields.fit_power([r["n_dipoles"] for r in rows], errors),
        }
    if all("vs_simnibs" in r for r in rows):
        values["worst_vs_simnibs"] = max(r["vs_simnibs"] for r in rows)
    return values
