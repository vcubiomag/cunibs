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


def _require_defined_handle_axis(
    degenerate: cp.ndarray, transforms: cp.ndarray, handles: cp.ndarray, dists: cp.ndarray
) -> None:
    """Raise if any handle left the coil's rotation about the scalp normal undefined.

    The kernel falls back to a canonical in-plane axis there, so the frame comes back finite
    and orthonormal and nothing downstream could tell it apart from a placement the caller
    meant.
    """
    if not bool(degenerate.any()):
        return

    # Recover the offender's projection from its own frame: z is the inward normal and the
    # origin sits distance_mm out along the outward one.
    i = int(cp.argmax(degenerate))
    tf = cp.asnumpy(transforms[i])
    handle = cp.asnumpy(handles[i])
    normal = -tf[:3, 2]
    proj = tf[:3, 3] - float(dists[i]) * normal
    d = handle - proj
    length = float(np.linalg.norm(d))
    if length == 0.0:
        detail = "it coincides with that projection"
    else:
        sin_angle = float(np.linalg.norm(np.cross(d / length, normal)))
        angle_deg = np.degrees(np.arcsin(min(sin_angle, 1.0)))
        detail = f"it lies {angle_deg:.2e} degrees off the outward normal there"
    raise ValueError(
        f"Degenerate coil placement at index {i} of {degenerate.shape[0]}: handle_mm "
        f"{np.array2string(handle, precision=3)} mm against a scalp projection of center_mm at "
        f"{np.array2string(proj, precision=3)} mm, so {detail} and the coil's rotation about "
        "the normal is undefined. Move handle_mm off the outward normal through the projected "
        "centre."
    )


def compute_coil_transforms(
    ctx: SolverContext,
    centers_mm: npt.ArrayLike,
    pos_ydir_mm: npt.ArrayLike | None,
    distances_mm: npt.ArrayLike,
) -> cp.ndarray:
    """Compute batched 4x4 coil-to-head affines in millimetres.

    ``pos_ydir_mm`` may be ``None`` for callers that only want the scalp projection and
    normal; the in-plane axes are then arbitrary but orthonormal and deterministic. When it is
    given, a handle that leaves the in-plane axis undefined raises ``ValueError``.
    """
    centers = cp.ascontiguousarray(cp.asarray(centers_mm, dtype=cp.float64))
    handles = (
        None
        if pos_ydir_mm is None
        else cp.ascontiguousarray(cp.asarray(pos_ydir_mm, dtype=cp.float64))
    )
    dists = cp.ascontiguousarray(cp.asarray(distances_mm, dtype=cp.float64))
    n_pl = centers.shape[0]
    out = cp.empty((n_pl, 16), dtype=cp.float64)
    degenerate = None if handles is None else cp.empty(n_pl, dtype=cp.int32)
    place_transforms(
        centers,
        handles,
        dists,
        ctx.skin_a,
        ctx.skin_b,
        ctx.skin_c,
        cp.ascontiguousarray(ctx.skin_tri_normals, dtype=cp.float64),
        out,
        degenerate,
        cp.cuda.get_current_stream().ptr,
    )
    transforms = out.reshape(n_pl, 4, 4)
    if degenerate is not None:
        _require_defined_handle_axis(degenerate, transforms, handles, dists)
    return transforms


def compute_coil_transform(
    ctx: SolverContext,
    center_mm: npt.ArrayLike,
    pos_ydir_mm: npt.ArrayLike,
    distance_mm: float,
) -> npt.NDArray[np.float64]:
    """Compute the 4x4 coil-to-head affine in millimetres.

    The columns are ``[x | y | z | c]``. ``y`` follows the handle, ``z`` points inward,
    ``x = y × z``, and ``c`` is offset from the scalp by ``distance_mm``. A handle that leaves
    ``y`` undefined raises ``ValueError``.
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

    # Keep centering in float64, then use float32 for the dominant N-body kernel: the
    # coil-scalp gap bounds |r - s| away from zero, so the |r - s|⁻³ weights stay well
    # inside float32 range.
    s = cp.ascontiguousarray(s.astype(DADT_COMPUTE_DTYPE))
    r = cp.ascontiguousarray(r.astype(DADT_COMPUTE_DTYPE))
    mp = cp.ascontiguousarray(mp.astype(DADT_COMPUTE_DTYPE))
    sn = cp.ascontiguousarray((s * s).sum(1))

    out = cp.empty_like(r)
    dadt_nbody(s, mp, sn, r, out, float(didt), MU0_OVER_4PI, cp.cuda.get_current_stream().ptr)
    return out
