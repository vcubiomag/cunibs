"""Self-convergence of the four recovery rules at tissue interfaces, over the resolution ladder.

Backs the recovery-paper result that over an eight-rung ladder with 400 interface and 400 interior
probes the interface orders were 2.35 (raw), 1.96 (spr_global), 2.26 (spr_tissue) and 2.38
(harmonic), with interior orders in the same range. Also backs the cuNIBS-paper result that
deriving the pose on each rung from the same scalp target tilts the coil by 0.69 to 1.29 degrees.

Rungs have different elements, so each field is reduced to mesh-independent functionals: the
volume-weighted mean of gray-matter |E| in a 2.5 mm ball around a fixed point. The points are
gray-matter barycentres on the finest rung, half from tetrahedra touching a node shared by two
tissues (interface) and half from the rest (interior). A rule's error on a rung is the RMS relative
difference of its functionals from the finest rung's, and its order is the log-log slope of that
error against the rung's element scale. This is self-convergence: the finest rung is not truth.

The coil frame is computed once on the finest rung and passed to every solve. A frame derived on
each rung from the same placement would follow that rung's smoothed scalp normal, and the tilt
between the two is recorded per rung as ``pose_drift``. The finest rung is the finest that fits
on the device; finer rungs are listed under ``skipped``. The functionals are saved to
``interface_convergence.npz``.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _fields, _measure, _run

MODES = ("raw", "spr_global", "spr_tissue", "harmonic")
FAMILIES = ("interface", "interior")
PROBE_RADIUS_MM = 2.5
N_PROBES = 400
GRAY_MATTER = 2


def _mixed_nodes(ctx):
    import cupy as cp
    import cupyx

    tags = cp.repeat(ctx.tet_tags, 4)
    lo = cp.full(ctx.n_nodes, 1 << 30, dtype=tags.dtype)
    hi = cp.zeros(ctx.n_nodes, dtype=tags.dtype)
    cupyx.scatter_min(lo, ctx.tet_nodes.ravel(), tags)
    cupyx.scatter_max(hi, ctx.tet_nodes.ravel(), tags)
    return cp.asnumpy(lo != hi)


def _probes(subject, n: int) -> dict[str, np.ndarray]:
    mesh = subject.mesh
    gm = mesh.tet_tags == GRAY_MATTER
    touches = _mixed_nodes(subject.context)[mesh.tet_nodes].any(axis=1)
    rng = np.random.default_rng(0)
    return {
        family: mesh.tet_barycenters_mm[rng.choice(np.flatnonzero(gm & sel), n, replace=False)]
        for family, sel in zip(FAMILIES, (touches, ~touches), strict=True)
    }


def _measure_rung(subject, coil, placement, frame, probes) -> dict:
    import cupy as cp

    from cunibs.fem import solve_placements_block
    from cunibs.fem.recovery import build_recovery_operator
    from cunibs.metrics import roi_mean

    ctx = subject.context
    own = _data.frames(subject, [placement])[0]
    rois = [subject.roi(p, PROBE_RADIUS_MM) for f in FAMILIES for p in probes[f]]
    site = [(placement.center_mm, placement.handle_mm, placement.distance_mm)]
    functionals = {}
    for mode in MODES:
        # Built uncached, so one operator at a time is resident on the finest rungs.
        op = None if mode == "raw" else build_recovery_operator(ctx, mode)
        (out,) = solve_placements_block(
            ctx,
            coil.positions_m,
            coil.moments,
            site,
            _data.DIDT,
            recovery=op,
            transforms=frame[None],
        )
        functionals[mode] = cp.asnumpy(
            cp.stack([roi_mean(out["magnE"], r.elem_idx, r.weights) for r in rois])
        ).reshape(len(FAMILIES), -1)
        iterations = out["iterations"]
        del op, out
    tilt = np.degrees(np.arccos(np.clip(own[:3, 2] @ frame[:3, 2], -1.0, 1.0)))
    return {
        "n_tets": subject.mesh.n_tets,
        "iterations": iterations,
        "pose_drift": {
            "axis_tilt_deg": float(tilt),
            "origin_shift_mm": float(np.linalg.norm(own[:3, 3] - frame[:3, 3])),
        },
        "functionals": functionals,
    }


def run(ctx: _run.Context) -> dict:
    from cunibs import Coil, Subject

    rungs = ctx.data.ladder[:3] if ctx.quick else ctx.data.ladder
    n_probes = 50 if ctx.quick else N_PROBES
    cap_path = ctx.data.mesh(rungs[0].subject)
    coil = Coil.load(_data.COIL)
    pinned, rows, skipped = None, {}, []
    for rung in reversed(rungs):
        subject = None
        try:
            subject = Subject.from_mesh(rung.path)
            if pinned is None:
                (placement,) = _data.placements(subject, cap_path, labels=("C3",))
                frame = _data.frames(subject, [placement])[0]
                pinned = (placement, frame, _probes(subject, n_probes), rung)
            rows[rung] = _measure_rung(subject, coil, *pinned[:3])
        except MemoryError as exc:
            skipped.append({"rung": rung.path.stem, "error": str(exc)})
        finally:
            if subject is not None:
                subject.free()
            _measure.free_pools()

    finest = pinned[3]
    coarser = [r for r in sorted(rows, key=lambda r: -r.scale) if r is not finest]
    convergence = {}
    for i, family in enumerate(FAMILIES):
        for mode in MODES:
            ref = rows[finest]["functionals"][mode][i]
            errors = [
                float(np.sqrt(np.mean(((rows[r]["functionals"][mode][i] - ref) / ref) ** 2)))
                for r in coarser
            ]
            convergence.setdefault(family, {})[mode] = {
                "rms_relative_difference": errors,
                **_fields.fit_power([r.scale for r in coarser], errors),
            }

    np.savez_compressed(
        ctx.artifact("interface_convergence.npz"),
        **{f"{r.path.stem}_{m}": rows[r]["functionals"][m] for r in rows for m in MODES},
        **{f"probes_{f}": pinned[2][f] for f in FAMILIES},
    )
    tilts = [rows[r]["pose_drift"]["axis_tilt_deg"] for r in coarser]
    return {
        "subject": finest.subject,
        "target": "C3",
        "coil": _data.COIL,
        "probe_radius_mm": PROBE_RADIUS_MM,
        "n_probes": {f: int(pinned[2][f].shape[0]) for f in FAMILIES},
        "frame_rung": finest.path.stem,
        "frame": pinned[1].tolist(),
        "rungs": {
            r.path.stem: {"scale": r.scale}
            | {k: v for k, v in rows[r].items() if k != "functionals"}
            for r in [*coarser, finest]
        },
        "skipped": skipped,
        "scales": [r.scale for r in coarser],
        "convergence": convergence,
        "interface_orders": {m: convergence["interface"][m]["slope"] for m in MODES},
        "interior_orders": {m: convergence["interior"][m]["slope"] for m in MODES},
        "pose_tilt_deg": {"min": min(tilts), "max": max(tilts)} if tilts else None,
    }
