"""What one coil placement costs, split by what each piece of work is reused across.

Backs the forward-solve rows of the cuNIBS cost table. Work is split by what it depends on: the
mesh (import, solver setup, recovery operator), the coil, and the placement. That split is the
only one where a ratio against SimNIBS compares like with like. Dividing one code's "evaluate"
by the other's measures which side happened to do the work earlier.

"Cold" is the first subject in this process, so it includes the library's one-time CUDA module
loads. The same path is then repeated on a second subject, which is what every subject after the
first costs. A warm placement is timed at two tiers: ``Subject.simulate``, which returns host
arrays as a user gets them, and ``fem.solve_placement``, which leaves everything on the device.
"""

from __future__ import annotations

from benchmarks import _data, _measure, _run


def _cold(path) -> tuple[dict[str, float], object, object, object]:
    """Each step from a mesh file to the first field, in the order a user takes them."""
    from cunibs import Coil, Subject, load_mesh
    from cunibs.fem import ensure_recovery

    ms = {}
    ms["mesh_import"], mesh = _measure.once(lambda: load_mesh(path))
    subject = Subject(mesh)
    ms["setup"], _ = _measure.once(lambda: subject.context)
    ms["recovery_operator"], _ = _measure.once(
        lambda: ensure_recovery(subject.context, "harmonic")
    )
    ms["coil_load"], coil = _measure.once(lambda: Coil.load(_data.COIL))
    # Choosing the pose is the user's input, not work the library does, so it is not timed.
    (placement,) = _data.placements(subject, path, labels=("C3",))
    ms["first_placement"], _ = _measure.once(lambda: subject.simulate(coil, placement))
    ms["total"] = sum(ms.values())
    return ms, subject, coil, placement


def run(ctx: _run.Context) -> dict:
    from cunibs.fem import ensure_recovery, solve_placement

    path = ctx.data.mesh(ctx.data.reference_subject)
    # Read once untimed, so the import measures parsing rather than whether the operating system
    # still has the file cached.
    path.read_bytes()
    first, subject, _, _ = _cold(path)
    subject.free()
    with _measure.PeakMemory() as peak:
        later, subject, coil, placement = _cold(path)

    n = 20 if ctx.quick else 200
    user = _measure.timed(lambda: subject.simulate(coil, placement), n)
    operator = ensure_recovery(subject.context, "harmonic")
    device = _measure.timed(
        lambda: solve_placement(
            subject.context,
            coil.positions_m,
            coil.moments,
            placement.center_mm,
            placement.handle_mm,
            placement.distance_mm,
            _data.DIDT,
            recovery=operator,
        ),
        n,
    )
    values = {
        "subject": ctx.data.reference_subject,
        "n_tets": subject.mesh.n_tets,
        "coil": _data.COIL,
        "cold_ms": first,
        "next_subject_ms": later,
        "placement_ms": {"simulate": _measure.stats(user), "device": _measure.stats(device)},
        "peak_device_mib": peak.peak_mib,
    }

    simnibs = ctx.latest("simnibs_timing")
    if simnibs and simnibs.get("configs"):

        def config(case):
            best = simnibs["best"][case]
            return next(
                c
                for c in simnibs["configs"]
                if c["solver"] == best["solver"] and c["threads"] == best["threads"]
            )

        cold, sweep = config("cold"), config("sweep")
        s = {k: v * 1e3 for k, v in cold["setup_s"].items()}
        warm_ms = values["placement_ms"]["simulate"]["median"]
        values["simnibs"] = {
            "machine": simnibs["machine"],
            "cold": simnibs["best"]["cold"],
            "sweep": simnibs["best"]["sweep"],
        }
        values["speedup"] = {
            "one_placement_from_cold": cold["cold_s"] * 1e3 / first["total"],
            "mesh_import": s["mesh_read_s"] / first["mesh_import"],
            "setup": (s["conductivity_s"] + s["assemble_s"] + s["prepare_solver_s"])
            / first["setup"],
            "coil_load": s["coil_load_s"] / first["coil_load"],
            "warm_placement": sweep["steady_placement_s"] * 1e3 / warm_ms,
        }

    # OnlineFEM's full-volume output is the like-for-like counterpart: a placement here returns
    # |E| for every tetrahedron too.
    onlinefem = ctx.latest("simnibs_onlinefem")
    if onlinefem:
        cold, sweep = onlinefem["best"]["cold"]["volume"], onlinefem["best"]["sweep"]["volume"]
        values["onlinefem"] = {"machine": onlinefem["machine"], "cold": cold, "sweep": sweep}
        values["speedup_vs_onlinefem"] = {
            "one_placement_from_cold": cold["s"] * 1e3 / first["total"],
            "warm_placement": sweep["s"] * 1e3 / values["placement_ms"]["simulate"]["median"],
        }
    return values
