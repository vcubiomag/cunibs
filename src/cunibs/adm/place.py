"""Batched coil placement: project many scalp targets and build their coil-to-head frames at once.

The per-placement ``fem.placement.compute_coil_transform`` re-launches many CuPy kernels over all
skin triangles each call, which dominates an ADM sweep. This wraps the ``place_transforms`` CUDA
kernel (one block per placement) to build all frames in a single launch, matching that transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy.typing as npt

from cunibs.solver import place_transforms

if TYPE_CHECKING:
    from cunibs.fem.solve import SolverContext


def compute_coil_transforms(
    ctx: "SolverContext",
    centers_mm: cp.ndarray,
    handles_mm: cp.ndarray,
    distances_mm: cp.ndarray,
) -> cp.ndarray:
    """Batched 4x4 coil-to-head affines ``(P,4,4)`` in mm (columns ``[x|y|z|c]``).

    Matches :func:`cunibs.fem.placement.compute_coil_transform` per placement (ties in the closest
    scalp triangle broken by lowest index).
    """
    centers = cp.ascontiguousarray(centers_mm, dtype=cp.float64)
    handles = cp.ascontiguousarray(handles_mm, dtype=cp.float64)
    dists = cp.ascontiguousarray(distances_mm, dtype=cp.float64)
    n_pl = centers.shape[0]
    out = cp.empty((n_pl, 16), dtype=cp.float64)
    place_transforms(
        centers,
        handles,
        dists,
        ctx.skin_a,
        ctx.skin_b,
        ctx.skin_c,
        cp.ascontiguousarray(ctx.skin_tri_normals, dtype=cp.float64),
        out,
        cp.cuda.get_current_stream().ptr,
    )
    return out.reshape(n_pl, 4, 4)


def place_coil_dipoles_batch(
    transforms: cp.ndarray, positions_m: npt.ArrayLike, moments: npt.ArrayLike
) -> tuple[cp.ndarray, cp.ndarray]:
    """Apply ``(P,4,4)`` affines to coil dipoles: return positions ``(P,N,3)`` (m) and moments."""
    rot = transforms[:, :3, :3]  # (P,3,3)
    trans = transforms[:, :3, 3]  # (P,3)
    pos = cp.asarray(positions_m) * 1e3  # (N,3) in mm
    mom = cp.asarray(moments)
    s = (cp.einsum("nj,pij->pni", pos, rot) + trans[:, None, :]) * 1e-3
    m = cp.einsum("nj,pij->pni", mom, rot)
    return s, m
