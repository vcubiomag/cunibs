"""Whether the linear system is solved to what the solver reports, and what the tolerance costs.

Backs the solver paragraph of the cuNIBS paper's verification results, and the recovery paper's
statement that tightening the tolerance to 1e-12 changes the field by less than the
discretisation error.

The CG residual the solver reports is its recurrence, which drifts from the true residual in
finite precision, and the preconditioner runs in reduced precision. The true residual
``||b - K v|| / ||b||`` of the grounded system is recomputed here in float64 at the default
tolerance and compared with the reported one. The raw gray-matter field is then solved at a
sweep of tolerances and compared, volume weighted, with a solve to 1e-12 on the reference
subject at C3.
"""

from __future__ import annotations

from benchmarks import _data, _fields, _run

TOLERANCES = (1e-4, 1e-5, 1e-6, 1e-8, 1e-10)
REFERENCE = 1e-12
GRAY_MATTER = 2


def run(ctx: _run.Context) -> dict:
    import cupy as cp

    from cunibs import Coil, Subject
    from cunibs.fem import coil_dadt_at_nodes, solve_placement
    from cunibs.fem.solve import _assemble_rhs

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    coil = Coil.load(_data.COIL)
    (placement,) = _data.placements(subject, path, labels=("C3",))
    solver = subject.context.solver
    mesh = subject.mesh
    gm = mesh.tet_tags == GRAY_MATTER
    vols = _fields.tet_volumes(mesh.nodes_mm, mesh.tet_nodes)[gm]

    def solve(tolerance: float) -> dict:
        # The tolerance is not a public setting.
        solver.tolerance = tolerance
        return solve_placement(
            subject.context,
            coil.positions_m,
            coil.moments,
            placement.center_mm,
            placement.handle_mm,
            placement.distance_mm,
            _data.DIDT,
        )

    default = solver.tolerance
    result = solve(default)
    dadt = coil_dadt_at_nodes(
        coil.positions_m,
        coil.moments,
        result["transform"],
        _data.DIDT,
        subject.context.nodes_mm,
    )
    # The right-hand side the solve consumed is not returned, and only the solver assembles it.
    b = _assemble_rhs(subject.context, dadt.reshape(-1, 1, 3)).ravel()[solver.idx]
    b = b.astype(cp.float64)
    stiffness = solver.matrix()
    true = float(cp.linalg.norm(b - stiffness @ result["v"][solver.idx]) / cp.linalg.norm(b))
    recurrence = result["relative_residual"]
    residual = {
        "tolerance": default,
        "iterations": result["iterations"],
        "recurrence": recurrence,
        "true": true,
        "recurrence_over_true": recurrence / true,
    }
    del result, dadt, b, stiffness

    reference = solve(REFERENCE)
    e_ref = cp.asnumpy(reference["E"])[gm]
    sweep = []
    for tolerance in TOLERANCES[::2] if ctx.quick else TOLERANCES:
        r = solve(tolerance)
        agreement = _fields.compare(cp.asnumpy(r["E"])[gm], e_ref, vols)
        sweep.append(
            {"tolerance": tolerance, "iterations": r["iterations"]}
            | {k: agreement[k] for k in ("rel_l2", "max_abs_over_peak")}
        )
    solver.tolerance = default

    values = {
        "subject": sid,
        "target": "C3",
        "coil": _data.COIL,
        "default_residual": residual,
        "reference": {"tolerance": REFERENCE, "iterations": reference["iterations"]},
        "sweep": sweep,
    }
    sphere = ctx.result("sphere")
    if sphere is not None:
        values["sphere_raw_error_finest"] = sphere["levels"][-1]["rel_l2"]["whole_ball"]["raw"]
    subject.free()
    return values
