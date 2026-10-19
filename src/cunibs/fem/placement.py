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


# Target nodes in metres about the coil origin, narrowed to float32, in one pass. The intrinsics
# keep the multiply and the subtract separately rounded, as two array operations would round them.
_centred_m = cp.ElementwiseKernel(
    "float64 x_mm, raw float64 o",
    "float32 r",
    "r = (float)__dsub_rn(__dmul_rn(x_mm, 1e-3), o[i % 3])",
    "cunibs_centred_m",
)


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


def _all_finite(*arrays: npt.ArrayLike | None) -> bool:
    """Whether every value given is finite. Host inputs are checked before they are uploaded,
    so only device inputs cost a synchronisation."""
    given = [a for a in arrays if a is not None]
    if not all(bool(np.isfinite(a).all()) for a in given if not isinstance(a, cp.ndarray)):
        return False
    device = [cp.isfinite(a).all() for a in given if isinstance(a, cp.ndarray)]
    return not device or bool(cp.stack(device).all())


def compute_coil_transforms(
    ctx: SolverContext,
    centers_mm: npt.ArrayLike,
    handles_mm: npt.ArrayLike | None,
    distances_mm: npt.ArrayLike,
) -> cp.ndarray:
    """Compute batched 4x4 coil-to-head affines in millimetres, ``(P, 4, 4)``.

    ``centers_mm`` and ``handles_mm`` are ``(P, 3)`` points; ``distances_mm`` is ``(P,)`` or one
    value for all. ``handles_mm`` may be ``None`` for callers that only want the scalp projection
    and normal; the in-plane axes are then arbitrary but orthonormal and deterministic. When it is
    given, a handle that leaves the in-plane axis undefined raises ``ValueError``.
    """
    if not _all_finite(centers_mm, handles_mm, distances_mm):
        raise ValueError("centers_mm, handles_mm and distances_mm must be finite.")
    centers = cp.ascontiguousarray(cp.asarray(centers_mm, dtype=cp.float64))
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers_mm must be (P, 3), got shape {centers.shape}.")
    n_pl = centers.shape[0]
    handles = (
        None
        if handles_mm is None
        else cp.ascontiguousarray(cp.asarray(handles_mm, dtype=cp.float64))
    )
    if handles is not None and handles.shape != centers.shape:
        raise ValueError(
            f"handles_mm must match centers_mm's shape {centers.shape}, got {handles.shape}."
        )
    dists = cp.asarray(distances_mm, dtype=cp.float64)
    # The kernel reads one distance per placement, so a short array is an out-of-bounds read.
    if dists.ndim == 0:
        dists = cp.full(n_pl, float(dists), dtype=cp.float64)
    if dists.shape != (n_pl,):
        raise ValueError(f"distances_mm must be one value or ({n_pl},), got {dists.shape}.")
    dists = cp.ascontiguousarray(dists)
    transforms = cp.empty((n_pl, 4, 4), dtype=cp.float64)
    degenerate = None if handles is None else cp.empty(n_pl, dtype=cp.int32)
    place_transforms(
        centers,
        handles,
        dists,
        ctx.skin_verts,
        ctx.skin_tri_normals,
        transforms,
        degenerate,
        cp.cuda.get_current_stream().ptr,
    )
    if degenerate is not None:
        _require_defined_handle_axis(degenerate, transforms, handles, dists)
    return transforms


def compute_coil_transform(
    ctx: SolverContext,
    center_mm: npt.ArrayLike,
    handle_mm: npt.ArrayLike,
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
        np.asarray(handle_mm, dtype=np.float64).reshape(1, 3),
        np.asarray([distance_mm], dtype=np.float64),
    )
    return cp.asnumpy(transform[0])


def coil_dadt_at_nodes(
    dip_pos_m: npt.ArrayLike,
    dip_moment: npt.ArrayLike,
    transform: npt.NDArray[np.float64] | cp.ndarray,
    didt: float,
    target_nodes_mm: cp.ndarray,
    *,
    out: cp.ndarray | None = None,
    column: int = 0,
) -> cp.ndarray:
    """Compute dA/dt at each target node.

    For dipole ``m`` at ``s`` and target ``r``:
        A(r) = (μ0/4π) Σ_j m_j × (r - s_j) / |r - s_j|³
        dA/dt = didt · A(r)

    Apply the full affine to positions and only its rotation to moments.

    ``out`` is an optional C-contiguous ``(n_nodes, k, 3)`` float32 block, of which this fills
    placement ``column`` and returns that view. The block solve stages its placements this way so
    that a node's k values sit together for the RHS gather.
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
    mp = cp.concatenate([m, cp.cross(m, s)], axis=1)

    # Keep centering in float64, then use float32 for the dominant N-body kernel: the
    # coil-scalp gap bounds |r - s| away from zero, so the |r - s|⁻³ weights stay well
    # inside float32 range.
    s = s.astype(DADT_COMPUTE_DTYPE)
    r = _centred_m(target_nodes_mm, o)
    mp = mp.astype(DADT_COMPUTE_DTYPE)[None]
    sn = cp.ascontiguousarray((s * s).sum(1))

    stream = cp.cuda.get_current_stream().ptr
    if out is None:
        field = cp.empty_like(r)
        dadt_nbody(s, mp, sn, r, field, float(didt), MU0_OVER_4PI, stream)
        return field
    n = int(r.shape[0])
    dadt_nbody(s, mp, sn, r, out.reshape(n, -1), float(didt), MU0_OVER_4PI, stream, column)
    return out[:, column, :]
