"""Both codes and every recovery rule against the closed-form field in a layered sphere.

Backs the sphere results of both manuscripts. The induced field of a coil outside a spherically
symmetric conductor is known exactly (Heller and van Hulsteyn 1992), so this is the one place a
true error can be quoted rather than a disagreement. The coil sits 10 mm above the pole of a
three-shell ball (brain, skull, scalp; ``_exact.ball_mesh``) refined over four levels.

Per level, cuNIBS is scored with each recovery rule at the default tolerance, and ``raw`` again
at a tolerance of 1e-10. SimNIBS solves the same mesh and frame with its direct solver, so only
its discretisation error is left. Errors are volume-weighted relative L2 over the whole ball and
over the inner shell. The exact field does not depend on the conductivity profile, so the
``raw`` fields on the layered ball and on the same mesh made homogeneous should differ only by
discretisation error.

Before any of that, the closed form is checked against SimNIBS's own implementation at 2000
random points, and its radial component, which is zero for the true field, is measured.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _exact, _fields, _measure, _run
from benchmarks._simnibs import mesh_arrays

SUBDIVISIONS = (2, 3, 4, 5)
MODES = ("raw", "spr_global", "spr_tissue", "harmonic")
DISTANCE_MM = 10.0
TIGHT_TOLERANCE = 1e-10


def _cross_check(ctx: _run.Context) -> dict:
    rng = np.random.default_rng(0)
    unit = rng.normal(size=(2000, 3))
    points = unit / np.linalg.norm(unit, axis=1, keepdims=True) * rng.uniform(1, 80, (2000, 1))
    dipoles = np.array([[0.0, 0.0, 110.0], [20.0, 0.0, 115.0], [-15.0, 10.0, 120.0]])
    moments = np.eye(3)[[1, 0, 2]] * 1e-3
    mine = _exact.induced_e(points, dipoles, moments, _data.DIDT)
    out = {"n_points": len(points), "radial_component_ratio": _exact.radial_ratio(points, mine)}
    response = ctx.simnibs.call(
        "sphere_field",
        {"didt": _data.DIDT},
        dipole_pos=dipoles * 1e-3,
        dipole_moment=moments,
        positions=points * 1e-3,
    )
    if response is not None:
        diff = np.abs(mine - response[1]["E"]).max()
        out["max_abs_diff"] = float(diff)
        out["max_diff_over_peak"] = float(diff / np.linalg.norm(mine, axis=1).max())
    return out


def _level(ctx: _run.Context, subdivisions: int, coil) -> dict:
    from cunibs import Placement, Subject

    subject = Subject(_exact.ball_mesh(subdivisions))
    mesh = subject.mesh
    nodes, tets, tags = mesh.nodes_mm, mesh.tet_nodes, mesh.tet_tags
    radius = _exact.BALL_RADII_MM[-1]
    placement = Placement([0, 0, radius], [0, 60, radius], DISTANCE_MM)
    frame = _data.frames(subject, [placement])[0]
    dipoles = coil.positions_m * 1e3 @ frame[:3, :3].T + frame[:3, 3]
    exact = _exact.induced_e(
        mesh.tet_barycenters_mm, dipoles, coil.moments @ frame[:3, :3].T, _data.DIDT
    )

    def simulate(recovery: str) -> tuple[np.ndarray, int]:
        r = subject.simulate(coil, placement, _data.DIDT, vectors=True, recovery=recovery)
        return r.E.astype(np.float64), r.iterations

    fields, iterations = {}, {}
    for mode in MODES:
        fields[mode], iterations[mode] = simulate(mode)
    solver = subject.context.solver
    # The tolerance is not a public setting.
    default, solver.tolerance = solver.tolerance, TIGHT_TOLERANCE
    fields["raw_tight"], iterations["raw_tight"] = simulate("raw")
    solver.tolerance = default

    response = ctx.simnibs.call(
        "forward",
        {"coil": _data.COIL, "didt": _data.DIDT, "solver": "pardiso"},
        frames=frame[None],
        **mesh_arrays(mesh),
    )
    if response is not None:
        fields["simnibs"] = response[1]["E"][0].astype(np.float64)
    subject.free()

    homogeneous = Subject(_exact.ball_mesh(subdivisions, tags=(2, 2, 2)))
    e_homogeneous = homogeneous.simulate(
        coil, placement, _data.DIDT, vectors=True, recovery="raw"
    ).E
    homogeneous.free()

    vols = _fields.tet_volumes(nodes, tets)
    inner = tags == _exact.BALL_TAGS[0]
    regions = {"whole_ball": np.ones_like(inner), "inner_shell": inner}
    row = {
        "subdivisions": subdivisions,
        "h_mm": _exact.mean_edge_mm(nodes, tets),
        "n_tets": len(tets),
        "clearance_radii": float(np.linalg.norm(dipoles, axis=1).min() / radius),
        "reference_radial_ratio": _exact.radial_ratio(mesh.tet_barycenters_mm, exact),
        "iterations": iterations,
        "rel_l2": {
            region: {
                k: _fields.compare(f[m], exact[m], vols[m])["rel_l2"] for k, f in fields.items()
            }
            for region, m in regions.items()
        },
        "layered_vs_homogeneous_rel_l2": _fields.compare(
            fields["raw"][inner], e_homogeneous[inner], vols[inner]
        )["rel_l2"],
    }
    if "simnibs" in fields:
        row["cunibs_vs_simnibs_rel_l2"] = _fields.compare(
            fields["raw"], fields["simnibs"], vols
        )["rel_l2"]
    _measure.free_pools()
    return row


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil

    coil = Coil.load(_data.COIL)
    rows = [_level(ctx, s, coil) for s in SUBDIVISIONS[: 3 if ctx.quick else None]]
    h = [r["h_mm"] for r in rows]
    return {
        "coil": _data.COIL,
        "radii_mm": list(_exact.BALL_RADII_MM),
        "cross_check": _cross_check(ctx),
        "levels": rows,
        "orders": {
            region: {k: _exact.order(h, [r["rel_l2"][region][k] for r in rows]) for k in fields}
            for region, fields in rows[0]["rel_l2"].items()
        },
    }
