"""Child for test_determinism_cross_process.py: digests of a fixed run, from a fresh interpreter."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from cunibs import ConductivityUQConfig, Placement, Subject, load_mesh
from cunibs.adm import Target, build_reciprocity, evaluate
from cunibs.coil import Coil


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def digests(mesh_path: Path) -> dict[str, str]:
    """One forward sweep, one Monte Carlo and one reciprocity field on a new subject."""
    mesh = load_mesh(mesh_path)
    coil = Coil.load("MagStim_D70")
    centroids = mesh.nodes_mm[mesh.skin_tris].mean(axis=1)
    top = centroids[np.argsort(centroids[:, 2])[-90:]][::10]
    angles = np.linspace(0.0, 6.0, len(top))
    placements = [
        Placement(c, c + 40.0 * np.array([np.cos(a), np.sin(a), 0.0]))
        for c, a in zip(top, angles, strict=True)
    ]
    gray = mesh.tet_barycenters_mm[mesh.tet_tags == 2]
    target = Target(gray[np.argmax(gray[:, 2])] - np.array([0.0, 0.0, 4.0]))

    out = {}
    with Subject(mesh) as subject:
        sweep = subject.iter_simulate(
            coil, placements, magnitude=True, vectors=True, potential=True
        )
        for i, r in enumerate(sweep):
            out[f"sweep {i}"] = _digest(r.magnE, r.E, r.v)
        uq = subject.simulate_conductivity_uq(
            coil, placements[0], ConductivityUQConfig(n_samples=8, seed=0), moments=True
        )
        out["uq"] = _digest(uq.mean_magnE, uq.std_magnE, uq.peak_samples)
        recip = build_reciprocity(subject, coil, target, top[:2], spacing_mm=4.0)
        out["adm"] = _digest(evaluate(recip, coil, placements[:2]))
    return out


if __name__ == "__main__":
    json.dump(digests(Path(sys.argv[1])), sys.stdout)
