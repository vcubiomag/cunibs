"""How far the field moves when the same input is solved on a different GPU.

Backs the cuNIBS reproducibility result that across three GPU models the largest per-element
difference was 7.3e-8 of the peak. Each machine solves the four targets on the reference subject
and saves its gray-matter |E| beside its results as ``cross_hardware.npz``. The run then reads
every machine's file under the results root and compares each pair of machines: the largest
per-element difference over the four targets, divided by the peak of the first machine's field.

A pair is only compared when both solved the same mesh with the same coil frames, since any other
difference would read as one between GPUs. Frames count as the same within 1e-9 mm: each GPU
computes them itself, and their last bits differ between architectures. Each card's fp64 rate relative to fp32 is recorded:
consumer cards run fp64 at 1:64 and datacentre cards at 1:2, and a portability claim across that
axis needs machines on both sides of it.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from benchmarks import _data, _run

FILENAME = "cross_hardware.npz"
GRAY_MATTER = 2


def _save(ctx: _run.Context) -> None:
    import cupy as cp

    from cunibs import Coil, Subject, load_mesh

    path = ctx.data.mesh(ctx.data.reference_subject)
    mesh = load_mesh(path)
    subject = Subject(mesh)
    placements = _data.placements(subject, path)
    results = list(
        subject.iter_simulate(Coil.load(_data.COIL), placements, _data.DIDT, magnitude=True)
    )
    gm = results[0].tet_tags == GRAY_MATTER
    props = cp.cuda.runtime.getDeviceProperties(ctx.device)
    np.savez_compressed(
        ctx.artifact(FILENAME),
        magnE_gm=np.stack([r.magnE[gm] for r in results]),
        frames=np.stack([r.transform for r in results]),
        mesh_hash=np.array(mesh.content_hash),
        gpu=np.array(props["name"].decode()),
        compute_capability=np.array(f"{props['major']}.{props['minor']}"),
        fp64_ratio=np.array(
            cp.cuda.Device(ctx.device).attributes["SingleToDoublePrecisionPerfRatio"]
        ),
        cuda_runtime=np.array(cp.cuda.runtime.runtimeGetVersion()),
    )
    subject.free()


FRAME_TOLERANCE = 1e-9


def _compare(a: dict, b: dict) -> dict:
    if str(a["mesh_hash"]) != str(b["mesh_hash"]) or a["magnE_gm"].shape != b["magnE_gm"].shape:
        return {"comparable": False, "reason": "different meshes"}
    frame_diff = float(np.abs(a["frames"] - b["frames"]).max())
    if frame_diff > FRAME_TOLERANCE:
        return {
            "comparable": False,
            "reason": "different coil frames",
            "max_frame_diff": frame_diff,
        }
    x, y = a["magnE_gm"].astype(np.float64), b["magnE_gm"].astype(np.float64)
    per_target = np.abs(x - y).max(axis=1) / np.abs(x).max(axis=1)
    return {
        "comparable": True,
        "max_frame_diff": frame_diff,
        "bitwise_equal": bool(np.array_equal(a["magnE_gm"], b["magnE_gm"])),
        "n_differing": int(np.count_nonzero(x != y)),
        "max_abs_over_peak": float(per_target.max()),
        "max_abs_over_peak_per_target": dict(
            zip(_data.TARGETS, per_target.tolist(), strict=True)
        ),
    }


def run(ctx: _run.Context) -> dict:
    _save(ctx)
    fields = {}
    for path in sorted(ctx.out.glob(f"*/{FILENAME}")):
        with np.load(path) as f:
            fields[path.parent.name] = {k: f[k] for k in f.files}

    machines = {
        tag: {
            "gpu": str(f["gpu"]),
            "compute_capability": str(f["compute_capability"]),
            "fp64_ratio": int(f["fp64_ratio"]),
            "cuda_runtime": int(f["cuda_runtime"]),
            "mesh_hash": str(f["mesh_hash"]),
        }
        for tag, f in fields.items()
    }
    pairs = {
        f"{a}__vs__{b}": _compare(fields[a], fields[b]) for a, b in combinations(fields, 2)
    }
    compared = [p["max_abs_over_peak"] for p in pairs.values() if p["comparable"]]
    ratios = {m["fp64_ratio"] for m in machines.values()}
    return {
        "subject": ctx.data.reference_subject,
        "targets": list(_data.TARGETS),
        "coil": _data.COIL,
        "machines": machines,
        "gpus": sorted({m["gpu"] for m in machines.values()}),
        "pairs": pairs,
        "worst_max_abs_over_peak": max(compared, default=None),
        "fp64_ratios": sorted(ratios),
        "spans_consumer_and_datacentre": min(ratios) <= 2 and max(ratios) >= 32,
    }
