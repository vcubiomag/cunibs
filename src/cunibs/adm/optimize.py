"""Coil-placement optimization over a scalp position set and in-plane rotation.

Each candidate position is projected onto the scalp once. The in-plane rotation is a rigid rotation
of the whole coil, so every target E-component ``E_d(θ)`` is band-limited in the angle. The optimum
rotation is found in closed form by sampling ``E_d(θ)`` at a few angles, trigonometrically
interpolating, and maximizing ``|E(θ)|²`` analytically, without any FEM solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from cunibs.adm.evaluate import _evaluate_native
from cunibs.adm.reciprocity import ReciprocityField, build_reciprocity
from cunibs.fem.placement import compute_coil_transforms
from cunibs.fem.solve import context_of
from cunibs.simulation import Placement

if TYPE_CHECKING:
    import numpy.typing as npt

    from cunibs.adm.target import ResolvedTarget, Target
    from cunibs.coil import Coil
    from cunibs.fem.solve import SolverContext
    from cunibs.simulation import Subject


# Distance from the centre to each reported handle point, along the handle direction.
_HANDLE_MM = 50.0


@dataclass
class OptimizeResult:
    """Best placement plus the full per-candidate objective map, all as host arrays.

    ``best_placement`` is the optimum ready for a forward solve: ``Subject.simulate(coil,
    result.best_placement)``. ``best_angle_rad`` is the handle's angle in the tangent plane,
    measured from ``u = n x e / |n x e|`` towards ``v = n x u``, where ``n`` is the outward scalp
    normal and ``e`` the world axis least aligned with it. That reference changes from centre to
    centre, so compare handles, not angles, across candidates.
    """

    best_placement: Placement
    best_center_mm: npt.NDArray[np.float64]  # scalp contact point of the optimum
    best_handle_mm: npt.NDArray[np.float64]  # a point 50 mm out along the optimum handle
    best_angle_rad: float
    best_E: npt.NDArray[np.float64]  # target E-vector components (per adjoint direction)
    best_objective: float  # |mean E| over the ROI (magnitude target) or signed mean E·ê
    centers_mm: npt.NDArray[np.float64]  # (C,3) scalp contact point per candidate
    center_objective: npt.NDArray[np.float64]  # (C,) optimum objective per candidate
    center_handle_mm: npt.NDArray[np.float64]  # (C,3) optimum handle point per candidate


def _inplane_basis(normal: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Two orthonormal in-plane vectors per row of ``normal`` (C,3), stable for any normal."""
    ref = cp.eye(3, dtype=cp.float64)[cp.argmin(cp.abs(normal), axis=1)]  # least-aligned axis
    u = cp.cross(normal, ref)
    u /= cp.linalg.norm(u, axis=1, keepdims=True)
    v = cp.cross(normal, u)
    return u, v


def _objective(e: cp.ndarray, magnitude: bool) -> cp.ndarray:
    """Scalar objective from an E-vector array ``(...,D)``: |E| for magnitude, else signed E·ê."""
    return cp.linalg.norm(e, axis=-1) if magnitude else e[..., 0]


def _frames(
    proj: cp.ndarray,
    normal: cp.ndarray,
    u: cp.ndarray,
    v: cp.ndarray,
    angles: cp.ndarray,
    distance_mm: float,
) -> cp.ndarray:
    """Coil-to-head affines ``(C·A, 4, 4)`` for each (position, angle) from a fixed projection.

    ``angles`` (A,) is shared across positions; ``y = cosθ·u + sinθ·v``, ``z = -normal``.
    """
    c = proj.shape[0]
    a = angles.shape[0]
    cos = cp.cos(angles)[None, :, None]
    sin = cp.sin(angles)[None, :, None]
    y = cos * u[:, None, :] + sin * v[:, None, :]  # (C,A,3)
    z = cp.broadcast_to((-normal)[:, None, :], (c, a, 3))
    x = cp.cross(y, z)
    t = cp.broadcast_to((proj + distance_mm * normal)[:, None, :], (c, a, 3))
    m = cp.zeros((c, a, 4, 4), dtype=cp.float64)
    m[..., :3, 0] = x
    m[..., :3, 1] = y
    m[..., :3, 2] = z
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m.reshape(c * a, 4, 4)


def _sample_objective(
    recip: ReciprocityField,
    coil: Coil,
    proj: cp.ndarray,
    normal: cp.ndarray,
    u: cp.ndarray,
    v: cp.ndarray,
    angles: cp.ndarray,
    distance_mm: float,
    didt: float,
) -> cp.ndarray:
    """Target E-vectors ``(C, A, D)`` for every position at each shared angle in ``angles`` (A,)."""
    c = proj.shape[0]
    a = angles.shape[0]
    frames = _frames(proj, normal, u, v, angles, distance_mm)
    return _evaluate_native(recip, coil, frames, didt).reshape(c, a, -1)


def _parabolic_refine(
    obj_grid: cp.ndarray, a_best: cp.ndarray, thetas: cp.ndarray
) -> cp.ndarray:
    """Sub-grid vertex of a parabola through the max and its two neighbours on a uniform θ grid."""
    n = thetas.shape[0]
    rows = cp.arange(obj_grid.shape[0])
    dtheta = float(thetas[1] - thetas[0])
    fa = obj_grid[rows, (a_best - 1) % n]
    fb = obj_grid[rows, a_best]
    fc = obj_grid[rows, (a_best + 1) % n]
    denom = fa - 2.0 * fb + fc
    delta = cp.where(denom < 0, 0.5 * (fa - fc) / cp.where(denom < 0, denom, 1.0), 0.0)
    return thetas[a_best] + cp.clip(delta, -0.5, 0.5) * dtheta


def _project(
    ctx: SolverContext, centers: cp.ndarray, distance_mm: float
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Scalp contact point, outward normal, and in-plane rotation basis for each center."""
    dist = cp.full(centers.shape[0], distance_mm, dtype=cp.float64)
    # No handle: only the normal and translation are read, and the rotation basis comes from
    # _inplane_basis below.
    base = compute_coil_transforms(ctx, centers, None, dist)
    normal = -base[:, :3, 2]
    proj = base[:, :3, 3] - distance_mm * normal
    u, v = _inplane_basis(normal)
    return proj, normal, u, v


def optimize(
    subject: Subject | SolverContext,
    coil: Coil,
    target: Target | ResolvedTarget,
    centers_mm: npt.ArrayLike,
    *,
    n_samples: int = 9,
    resolution: int = 512,
    distance_mm: float = 4.0,
    didt: float = 1e6,
    spacing_mm: float = 3.0,
    margin_mm: float = 8.0,
) -> OptimizeResult:
    """Build the reciprocity field for ``target`` and search ``centers_mm`` × rotations for the best
    coil placement, with no per-placement FEM solve.

    ``didt`` is the coil current's rate of change in A/s. It scales ``best_E`` linearly and
    leaves the chosen placement unchanged. To search several candidate sets against one target,
    build the field once with :func:`~cunibs.adm.build_reciprocity` and call
    :func:`optimize_placement` on it.
    """
    _check_angle_sampling(n_samples, resolution)
    recip = build_reciprocity(
        context_of(subject),
        coil,
        target,
        centers_mm,
        distance_mm=distance_mm,
        spacing_mm=spacing_mm,
        margin_mm=margin_mm,
    )
    return optimize_placement(
        recip, coil, n_samples=n_samples, resolution=resolution, didt=didt
    )


def _check_angle_sampling(n_samples: int, resolution: int) -> None:
    if n_samples < 1 or n_samples % 2 == 0:
        raise ValueError("n_samples must be odd for symmetric Fourier interpolation.")
    if resolution < n_samples:
        raise ValueError(
            f"resolution must be at least n_samples ({n_samples}), got {resolution}."
        )


def optimize_placement(
    recip: ReciprocityField,
    coil: Coil,
    centers_mm: npt.ArrayLike | None = None,
    *,
    n_samples: int = 9,
    resolution: int = 512,
    distance_mm: float | None = None,
    didt: float = 1e6,
) -> OptimizeResult:
    """Find the coil position and in-plane angle maximizing the target objective.

    ``centers_mm`` is a set of candidate scalp targets ``(C,3)`` (each is projected onto the scalp),
    and ``distance_mm`` the coil's offset from it. Both default to what ``recip`` was built for.
    The rotation is optimized in closed form: ``E_d(θ)`` is sampled at ``n_samples`` (odd) angles,
    Fourier-interpolated to ``resolution`` points, and ``|E(θ)|²`` is maximized on that fine grid.

    ``didt`` is the coil current's rate of change in A/s. It scales ``best_E`` linearly and
    leaves the chosen placement unchanged.

    The angle sweep is flattened over (position, angle), so a grid-bounds error from it names
    a sample index that decodes as ``position = index // n_samples``.
    """
    _check_angle_sampling(n_samples, resolution)
    if centers_mm is None:
        centers_mm = recip.centers_mm
    if distance_mm is None:
        distance_mm = recip.distance_mm
    if centers_mm is None or distance_mm is None:
        raise ValueError(
            "this reciprocity field does not record the centres and distance it was built for; "
            "pass centers_mm and distance_mm."
        )

    ctx = recip.ctx
    centers = cp.ascontiguousarray(cp.asarray(centers_mm, dtype=cp.float64).reshape(-1, 3))
    proj, normal, u, v = _project(ctx, centers, distance_mm)

    # Sample E_d(θ) at n_samples angles, then trigonometrically interpolate to a fine θ grid.
    order = (n_samples - 1) // 2
    theta_s = cp.linspace(0.0, 2.0 * cp.pi, n_samples, endpoint=False)
    e_s = _sample_objective(
        recip, coil, proj, normal, u, v, theta_s, distance_mm, didt
    )  # (C,K,D)

    theta_f = cp.linspace(0.0, 2.0 * cp.pi, resolution, endpoint=False)
    d = theta_f[:, None] - theta_s[None, :]  # (F,K)
    kernel = cp.ones_like(d)
    for h in range(1, order + 1):
        kernel += 2.0 * cp.cos(h * d)
    kernel /= n_samples  # interpolating Dirichlet kernel (exact for band limit ``order``)

    e_f = cp.tensordot(e_s, kernel, axes=([1], [1]))  # (C,D,F)
    obj_f = _objective(cp.moveaxis(e_f, 1, 2), recip.magnitude)  # (C,F)
    theta = _parabolic_refine(obj_f, cp.argmax(obj_f, axis=1), theta_f)

    # Exact final evaluation at each position's own optimum angle (one field eval per position).
    e = _eval_per_position(recip, coil, proj, normal, u, v, theta, distance_mm, didt)  # (C,D)
    obj = cp.asnumpy(_objective(e, recip.magnitude))  # (C,)
    handle = cp.asnumpy(
        proj + _HANDLE_MM * (cp.cos(theta)[:, None] * u + cp.sin(theta)[:, None] * v)
    )
    proj_h = cp.asnumpy(proj)

    gi = int(np.argmax(obj))
    return OptimizeResult(
        best_placement=Placement(proj_h[gi], handle[gi], distance_mm),
        best_center_mm=proj_h[gi],
        best_handle_mm=handle[gi],
        best_angle_rad=float(theta[gi]),
        best_E=cp.asnumpy(e[gi]),
        best_objective=float(obj[gi]),
        centers_mm=proj_h,
        center_objective=obj,
        center_handle_mm=handle,
    )


def _eval_per_position(
    recip: ReciprocityField,
    coil: Coil,
    proj: cp.ndarray,
    normal: cp.ndarray,
    u: cp.ndarray,
    v: cp.ndarray,
    theta: cp.ndarray,
    distance_mm: float,
    didt: float,
) -> cp.ndarray:
    """Target E-vectors ``(C, D)`` with one angle ``theta[c]`` per position."""
    y = cp.cos(theta)[:, None] * u + cp.sin(theta)[:, None] * v
    z = -normal
    x = cp.cross(y, z)
    t = proj + distance_mm * normal
    m = cp.zeros((proj.shape[0], 4, 4), dtype=cp.float64)
    m[:, :3, 0] = x
    m[:, :3, 1] = y
    m[:, :3, 2] = z
    m[:, :3, 3] = t
    m[:, 3, 3] = 1.0
    return _evaluate_native(recip, coil, m, didt)
