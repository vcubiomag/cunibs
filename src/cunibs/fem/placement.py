"""Place the coil and compute its magnetic-dipole dA/dt field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.solver import dadt_nbody, place_transforms

if TYPE_CHECKING:
    from cunibs.fem.solve import SolverContext

MU0_OVER_4PI = 1e-7  # μ0 / 4π in T·m/A, the magnetic-dipole vector-potential constant
DADT_COMPUTE_DTYPE = cp.float32


def compute_coil_transforms(
    ctx: SolverContext,
    centers_mm: npt.ArrayLike,
    pos_ydir_mm: npt.ArrayLike,
    distances_mm: npt.ArrayLike,
) -> cp.ndarray:
    """Compute batched 4x4 coil-to-head affines in millimetres."""
    centers = cp.ascontiguousarray(cp.asarray(centers_mm, dtype=cp.float64))
    handles = cp.ascontiguousarray(cp.asarray(pos_ydir_mm, dtype=cp.float64))
    dists = cp.ascontiguousarray(cp.asarray(distances_mm, dtype=cp.float64))
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


def compute_coil_transform(
    ctx: SolverContext,
    center_mm: npt.ArrayLike,
    pos_ydir_mm: npt.ArrayLike,
    distance_mm: float,
) -> npt.NDArray[np.float64]:
    """Compute the 4x4 coil-to-head affine in millimetres.

    The columns are ``[x | y | z | c]``. ``y`` follows the handle, ``z`` points inward,
    ``x = y × z``, and ``c`` is offset from the scalp by ``distance_mm``.
    """
    transform = compute_coil_transforms(
        ctx,
        np.asarray(center_mm, dtype=np.float64).reshape(1, 3),
        np.asarray(pos_ydir_mm, dtype=np.float64).reshape(1, 3),
        np.asarray([distance_mm], dtype=np.float64),
    )
    return cp.asnumpy(transform[0])


def coil_dadt_at_nodes(
    dip_pos_m: npt.ArrayLike,
    dip_moment: npt.ArrayLike,
    transform: npt.NDArray[np.float64],
    didt: float,
    target_nodes_mm: cp.ndarray,
) -> cp.ndarray:
    """Compute dA/dt at each target node.

    For dipole ``m`` at ``s`` and target ``r``:
        A(r) = (μ0/4π) Σ_j m_j × (r - s_j) / |r - s_j|³
        dA/dt = didt · A(r)

    Apply the full affine to positions and only its rotation to moments.
    """
    rot = cp.asarray(transform[:3, :3])
    trans = cp.asarray(transform[:3, 3])
    s = (cp.asarray(dip_pos_m) * 1e3 @ rot.T + trans) * 1e-3
    m = cp.asarray(dip_moment) @ rot.T

    # Center coordinates at the coil origin to reduce cancellation in |r - s|².
    # Use A(r) = μ0/4π (W × r - P), where W = Σ w_j m_j,
    # P = Σ w_j (m_j × s_j), and w_j = |r - s_j|⁻³.
    o = trans * 1e-3
    s = s - o
    r = target_nodes_mm * 1e-3 - o
    mp = cp.concatenate([m, cp.cross(m, s)], axis=1)

    # Keep centering in float64, then use float32 for the dominant N-body kernel.
    # The coil-scalp gap bounds |r - s| away from zero; measured relative L2 error is about 4e-7.
    s = cp.ascontiguousarray(s.astype(DADT_COMPUTE_DTYPE))
    r = cp.ascontiguousarray(r.astype(DADT_COMPUTE_DTYPE))
    mp = cp.ascontiguousarray(mp.astype(DADT_COMPUTE_DTYPE))
    sn = cp.ascontiguousarray((s * s).sum(1))

    out = cp.empty_like(r)
    dadt_nbody(s, mp, sn, r, out, float(didt), MU0_OVER_4PI, cp.cuda.get_current_stream().ptr)
    return out
