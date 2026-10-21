"""What each recovery rule does at a gray-matter/CSF interface, in one typical subject-target pair.

Backs the recovery paper's interface figure and the numbers around it: at gray-matter/CSF nodes
``spr_tissue`` reports the gray-matter normal field at a median 1.03 times the ``harmonic`` value,
and ``spr_global``, which holds one value per node, at 0.64 of the gray-matter value and 2.03 of
the CSF value (the figure's dashed slope); interface tetrahedra are 90.2% of gray matter; and the
5th to 95th percentile of each rule's |E| difference from ``harmonic`` in interface and interior
gray-matter tetrahedra.

The pair is the one from ``recovery_in_practice`` whose ``spr_global``-to-``harmonic`` peak
displacement is nearest the cohort median, so the case is typical rather than chosen to make the
point. At each node on a gray-matter/CSF face, off the outer surface, the normal is the
area-weighted mean of its gray-matter/CSF faces, oriented from gray matter to CSF, and each rule's
recovered field is read on both sides. A gray-matter tetrahedron is at the interface when one of
its nodes belongs to more than one tissue.

Writes the figure's tables as ``interface_normal.dat`` (a rank-thinned sample of the node pairs)
and ``interface_effect.dat`` beside this machine's results, in the columns fig_interface.tex reads.
"""

from __future__ import annotations

import numpy as np

from benchmarks import _data, _run

MODES = ("raw", "spr_global", "spr_tissue", "harmonic")
NODAL = MODES[1:]
GRAY_MATTER, CSF = 2, 3
FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
N_SCATTER = 700


def _case(payoff: dict) -> dict:
    median = payoff["displacement"]["spr_global__vs__harmonic"]["peak_shift_mm"]["median"]

    def gap(case):
        m = case["modes"]
        shift = np.linalg.norm(
            np.subtract(m["spr_global"]["peak_mm"], m["harmonic"]["peak_mm"])
        )
        return abs(shift - median)

    return min(payoff["cases"], key=gap)


def _interface_normals(nodes, tets, tags) -> tuple[np.ndarray, np.ndarray]:
    """Nodes on a gray-matter/CSF face and their area-weighted normals, gray matter to CSF."""
    cells = np.flatnonzero((tags == GRAY_MATTER) | (tags == CSF))
    faces = np.sort(tets[cells][:, FACES].reshape(-1, 3), axis=1)
    owner = np.repeat(cells, 4)
    n = np.int64(len(nodes))
    key = (faces[:, 0] * n + faces[:, 1]) * n + faces[:, 2]
    order = np.argsort(key, kind="stable")
    shared = np.flatnonzero(key[order][1:] == key[order][:-1])
    a, b = owner[order[shared]], owner[order[shared + 1]]
    keep = tags[a] != tags[b]
    tri = faces[order[shared[keep]]]
    gm = np.where(tags[a[keep]] == GRAY_MATTER, a[keep], b[keep])
    csf = np.where(tags[a[keep]] == GRAY_MATTER, b[keep], a[keep])
    p = nodes[tri]
    # Twice the face area times its unit normal, so summing them weights by area.
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    step = nodes[tets[csf]].mean(axis=1) - nodes[tets[gm]].mean(axis=1)
    normal *= np.sign(np.einsum("ij,ij->i", normal, step))[:, None]
    total = np.zeros_like(nodes)
    np.add.at(total, tri.ravel(), np.repeat(normal, 3, axis=0))
    on = np.unique(tri)
    return on, total[on] / np.linalg.norm(total[on], axis=1, keepdims=True)


def _write(path, header: list[str], columns: str, rows: list[str]) -> None:
    path.write_text("\n".join([*(f"# {h}" for h in header), columns, *rows]) + "\n")


def run(ctx: _run.Context) -> dict:
    import cupy as cp

    from cunibs import Coil, Subject
    from cunibs.fem import TISSUE_CONDUCTIVITY, outer_boundary_nodes

    payoff = ctx.result("recovery_in_practice")
    if payoff is None:
        raise RuntimeError("run recovery_in_practice on this machine first")
    case = _case(payoff)
    path = ctx.data.mesh(case["subject"])
    subject = Subject.from_mesh(path)
    (placement,) = _data.placements(subject, path, labels=(case["target"],))
    mesh = subject.mesh
    tets, tags = mesh.tet_nodes.astype(np.int64), mesh.tet_tags
    node, normal = _interface_normals(mesh.nodes_mm, tets, tags)
    surface = cp.asnumpy(outer_boundary_nodes(subject.context.tet_nodes, mesh.n_nodes)) > 0
    gm = tags == GRAY_MATTER

    coil = Coil.load(_data.COIL)
    magn, side = {}, {}
    for mode in MODES:
        r = subject.simulate(
            coil, placement, _data.DIDT, magnitude=True, nodal=mode != "raw", recovery=mode
        )
        magn[mode] = r.magnE[gm].astype(np.float64)
        if mode != "raw":
            side[mode] = [
                np.einsum("ij,ij->i", r.nodal_field(region)[node], normal)
                for region in ("gray_matter", "csf")
            ]
        if mode == "harmonic":
            mixed = np.bincount(r.slot_node, minlength=mesh.n_nodes) > 1
    subject.free()

    keep = ~surface[node] & np.all([np.isfinite(s) for m in NODAL for s in side[m]], axis=0)
    idx = np.flatnonzero(keep)
    ratios = {
        mode: {
            region: float(np.median(side[mode][i][idx] / side["harmonic"][i][idx]))
            for i, region in enumerate(("gray_matter", "csf"))
        }
        for mode in ("spr_global", "spr_tissue")
    }
    at_interface = mixed[tets[gm]].any(axis=1)
    effect, rows = {}, []
    for y, mode in enumerate(MODES[:3]):
        rel = 100 * (magn[mode] - magn["harmonic"]) / magn["harmonic"]
        row = [f"{y + 0.16:.2f}", f"{y - 0.16:.2f}"]
        for bucket, sel in (("interface", at_interface), ("interior", ~at_interface)):
            p5, q1, med, q3, p95 = np.percentile(rel[sel], [5, 25, 50, 75, 95])
            effect.setdefault(mode, {})[bucket] = {
                "p5": p5,
                "q1": q1,
                "median": med,
                "q3": q3,
                "p95": p95,
            }
            row += [f"{v:.6g}" for v in (med, q1, q3, p5, p95)]
        rows.append(" ".join(row))

    contrast = TISSUE_CONDUCTIVITY[CSF] / TISSUE_CONDUCTIVITY[GRAY_MATTER]
    where = f"case: {case['subject']} {case['target']}"
    _write(
        ctx.artifact("interface_effect.dat"),
        [
            f"{where}; gray-matter tetrahedra vs harmonic recovery",
            "rows: raw, spr_global, spr_tissue",
        ],
        "y_if y_in med_if q1_if q3_if p5_if p95_if med_in q1_in q3_in p5_in p95_in",
        rows,
    )
    # Thinned by rank of the harmonic value, so every decade of the log-log plot keeps the
    # density the population has there.
    sample = idx[np.argsort(np.abs(side["harmonic"][0][idx]))]
    sample = sample[np.linspace(0, sample.size - 1, min(N_SCATTER, sample.size)).astype(int)]
    _write(
        ctx.artifact("interface_normal.dat"),
        [f"{where}; gray-matter/CSF node pairs", f"|E.n| in V/m; sigma ratio {contrast:.4g}"],
        " ".join([f"gm_{m}" for m in NODAL] + [f"csf_{m}" for m in NODAL]),
        [" ".join(f"{abs(side[m][i][j]):.6g}" for i in (0, 1) for m in NODAL) for j in sample],
    )
    return {
        "subject": case["subject"],
        "target": case["target"],
        "coil": _data.COIL,
        "conductivity_ratio": contrast,
        "n_node_pairs": int(idx.size),
        "normal_field_ratio_to_harmonic": ratios,
        "slope": ratios["spr_global"]["gray_matter"],
        "spr_global_sign_flips": float(
            np.mean(np.sign(side["spr_global"][0][idx]) != np.sign(side["harmonic"][0][idx]))
        ),
        "gm_tets_at_interface": float(at_interface.mean()),
        "effect_pct": effect,
    }
