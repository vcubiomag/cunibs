"""Crop a spherical patch out of a full head mesh to make a committable test fixture.

The full SimNIBS head meshes are ~184 MB, far too large to track. A ball around the vertex
keeps what the synthetic cube cannot provide: seven tissue tags with real conductivity
contrasts, a curved non-convex scalp, and the real element-quality distribution.

    uv run python tools/make_test_patch.py --source .../sub-001.msh --radius 25 \
        --out tests/data/head_patch_r25.msh.gz

Deterministic: no RNG, so re-running reproduces the committed bytes exactly.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from msh_writer import pack_msh

from cunibs.mesh import SKIN_SURFACE_TAG, parse_msh_binary


def crop(
    nodes: np.ndarray,
    tet_nodes: np.ndarray,
    tet_tags: np.ndarray,
    skin_tris: np.ndarray,
    radius_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (nodes, tets, tet_tags, skin_tris, center) for a ball around the vertex."""
    skin_nodes = np.unique(skin_tris)
    top = nodes[skin_nodes][np.argmax(nodes[skin_nodes][:, 2])]
    center = top + np.array([0.0, 0.0, -0.45 * radius_mm])

    barycenters = nodes[tet_nodes].mean(axis=1)
    keep_tet = np.linalg.norm(barycenters - center, axis=1) <= radius_mm
    kept_tets = tet_nodes[keep_tet]

    keep_node = np.unique(kept_tets)
    remap = np.full(nodes.shape[0], -1, dtype=np.int32)
    remap[keep_node] = np.arange(keep_node.size, dtype=np.int32)

    # parse_msh_binary rejects a mesh unless every surface triangle survives node
    # reindexing, so only triangles whose three nodes sit inside the kept tets carry over.
    keep_tri = (remap[skin_tris] >= 0).all(axis=1)

    return (
        np.ascontiguousarray(nodes[keep_node]),
        np.ascontiguousarray(remap[kept_tets]),
        np.ascontiguousarray(tet_tags[keep_tet]),
        np.ascontiguousarray(remap[skin_tris[keep_tri]]),
        center,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="full binary Gmsh 2.2 .msh")
    ap.add_argument("--radius", type=float, default=25.0, help="crop radius in mm")
    ap.add_argument("--out", type=Path, required=True, help="output .msh.gz")
    args = ap.parse_args()

    nodes, tet_nodes, tet_tags, surf_tris, surf_tags = parse_msh_binary(args.source)
    skin_tris = surf_tris[surf_tags == SKIN_SURFACE_TAG]

    p_nodes, p_tets, p_tags, p_skin, center = crop(
        nodes, tet_nodes, tet_tags, skin_tris, args.radius
    )

    raw = pack_msh(p_nodes, p_tets + 1, p_tags, p_skin + 1, np.full(p_skin.shape[0], 1005))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))

    tags, counts = np.unique(p_tags, return_counts=True)
    manifest = {
        "source": args.source.name,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "radius_mm": args.radius,
        "center_mm": [float(c) for c in center],
        "n_nodes": int(p_nodes.shape[0]),
        "n_tets": int(p_tets.shape[0]),
        "n_skin_tris": int(p_skin.shape[0]),
        "tet_tags": {str(int(t)): int(c) for t, c in zip(tags, counts)},
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    args.out.with_suffix("").with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"{args.out}: {args.out.stat().st_size / 1e6:.2f} MB gz, {len(raw) / 1e6:.2f} MB raw")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
