"""Cortical maps of one subject-target pair: both codes' fields, their difference, and the prior's spread.

Backs fig_maps of the cuNIBS paper. The pair is the reference subject at C3, and SimNIBS is asked
the same question field_agreement asks it for that subject, so the map shows the comparison the
cohort statistics summarise. Each gray-matter surface triangle takes the value of the gray-matter
tetrahedron it bounds, so every map is an element field, with no smoothing added for display.

Maps: cuNIBS's reported |E| (harmonic recovery), the unrecovered |E| of both codes, their
difference over the peak, and the mean and coefficient of variation of |E| over uq_cohort's prior.
Only the cortex within ``RADIUS_MM`` of the ROI centre is kept. The arrays are saved beside the
result as ``field_maps.npz`` for the figure script.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _run
from benchmarks._simnibs import mesh_arrays

FILENAME = "field_maps.npz"
GM_SURFACE = 1002
GRAY_MATTER = 2
RADIUS_MM = 40.0
N_DRAWS = 500


def _bounding_tet(tris: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """For each triangle, the index into ``tets`` of the tetrahedron that has it as a face."""
    # Three node indices packed into one int64 key, 21 bits each.
    if max(int(tris.max()), int(tets.max())) >= 1 << 21:
        raise ValueError("node indices exceed the 21-bit face key")
    faces = np.sort(
        np.concatenate(
            [tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]]
        ),
        axis=1,
    ).astype(np.int64)
    owner = np.tile(np.arange(len(tets)), 4)

    def key(f: np.ndarray) -> np.ndarray:
        return (f[:, 0] << 42) | (f[:, 1] << 21) | f[:, 2]

    k = key(faces)
    order = np.argsort(k, kind="stable")
    k, owner = k[order], owner[order]
    q = key(np.sort(tris, axis=1).astype(np.int64))
    at = np.searchsorted(k, q)
    found = (at < len(k)) & (k[np.minimum(at, len(k) - 1)] == q)
    if not found.all():
        raise ValueError(
            f"{int((~found).sum())} surface triangles bound no gray-matter element"
        )
    return owner[at]


def run(ctx: _run.Context) -> dict | None:
    import cupy as cp

    from cunibs import Coil, Subject
    from cunibs.uq import ConductivityUQConfig

    sid = ctx.data.reference_subject
    path = ctx.data.mesh(sid)
    subject = Subject.from_mesh(path)
    mesh = subject.mesh
    placements = _data.placements(subject, path)
    frames = _data.frames(subject, placements)
    response = ctx.simnibs.call(
        "forward",
        {
            "coil": _data.COIL,
            "didt": _data.DIDT,
            "solver": "hypre",
            "threads": 0,
            "node_round_trip": True,
        },
        frames=frames,
        **mesh_arrays(mesh),
    )
    if response is None:
        subject.free()
        return None
    simnibs = np.linalg.norm(response[1]["E"][0], axis=1)

    coil = Coil.load(_data.COIL)
    placement = placements[0]
    harmonic = subject.simulate(coil, placement, _data.DIDT, magnitude=True).magnE
    raw = subject.simulate(coil, placement, _data.DIDT, magnitude=True, recovery="raw").magnE
    n = 20 if ctx.quick else N_DRAWS
    uq = subject.simulate_conductivity_uq(
        coil, placement, ConductivityUQConfig(n_samples=n), _data.DIDT, moments=True
    )

    (center,) = _data.roi_centers(subject, [placement])
    nodes = mesh.nodes_mm
    gm_tets = np.flatnonzero(mesh.tet_tags == GRAY_MATTER)
    tris = mesh.surf_tris[mesh.surf_tags == GM_SURFACE]
    near = np.linalg.norm(nodes[tris].mean(axis=1) - center, axis=1) <= RADIUS_MM
    tris = tris[near]
    elem = gm_tets[_bounding_tet(tris, mesh.tet_nodes[gm_tets])]
    used, local = np.unique(tris, return_inverse=True)

    peak = float(raw[gm_tets].max())
    maps = {
        "cunibs_harmonic": harmonic[elem],
        "cunibs_raw": raw[elem],
        "simnibs_raw": simnibs[elem],
        "difference_over_peak": (raw[elem] - simnibs[elem]) / peak,
        "uq_mean": uq.mean_magnE[elem],
        "uq_cov": uq.cov_magnE[elem],
    }
    np.savez_compressed(
        ctx.artifact(FILENAME),
        nodes_mm=nodes[used],
        tris=local.reshape(-1, 3).astype(np.int32),
        roi_center_mm=np.asarray(center),
        coil_frame=frames[0],
        **{k: np.asarray(v, np.float32) for k, v in maps.items()},
    )
    subject.free()
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "subject": sid,
        "target": "C3",
        "coil": _data.COIL,
        "radius_mm": RADIUS_MM,
        "n_triangles": len(tris),
        "n_draws": n,
        "gm_peak_raw": peak,
        "max_abs_difference_over_peak": float(np.abs(maps["difference_over_peak"]).max()),
        "map_max": {k: float(np.max(v)) for k, v in maps.items()},
    }
