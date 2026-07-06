"""Coil-placement optimization over a scalp position set and in-plane rotation.

Each candidate position is projected onto the scalp once. The in-plane rotation is a rigid rotation
of the whole coil, so every target E-component ``E_d(θ)`` is band-limited in the angle. The optimum
rotation is found in closed form by sampling ``E_d(θ)`` at a few angles, trigonometrically
interpolating, and maximizing ``|E(θ)|²`` analytically, with no dense sweep and no FEM solves.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.adm.evaluate import _interp_reduce
from cunibs.adm.place import compute_coil_transforms, place_coil_dipoles_batch
from cunibs.adm.reciprocity import ReciprocityField, build_reciprocity
from cunibs.adm.target import Target
from cunibs.coil import Coil
from cunibs.fem.solve import SolverContext


@dataclass
class OptimizeResult:
    """Best placement plus the full per-position objective map."""

    best_center_mm: npt.NDArray[np.float64]  # scalp contact point of the optimum
    best_handle_mm: npt.NDArray[np.float64]  # a point along the optimum handle direction
    best_angle_rad: float
    best_E: npt.NDArray[np.float64]  # target E-vector components (per adjoint direction)
    best_objective: float  # |E| (magnitude target) or signed E·ê (directional)
    centers_mm: cp.ndarray  # (C,3) scalp contact point per candidate
    center_objective: cp.ndarray  # (C,) optimum objective per candidate
    center_handle_mm: cp.ndarray  # (C,3) optimum handle point per candidate


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
    s, m = place_coil_dipoles_batch(frames, coil.positions_m, coil.moments)
    return _interp_reduce(recip, s, m, didt).reshape(c, a, -1)


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
    base = compute_coil_transforms(ctx, centers, centers + cp.asarray([1.0, 0.0, 0.0]), dist)
    normal = -base[:, :3, 2]
    proj = base[:, :3, 3] - distance_mm * normal
    u, v = _inplane_basis(normal)
    return proj, normal, u, v


def optimize(
    ctx: SolverContext,
    coil: Coil,
    target: Target,
    centers_mm: npt.ArrayLike,
    *,
    n_samples: int = 9,
    distance_mm: float = 4.0,
    didt: float = 1e6,
    spacing_mm: float = 2.0,
) -> OptimizeResult:
    """Build the reciprocity field for ``target`` and search ``centers_mm`` × rotations for the best
    coil placement, with no per-placement FEM solve."""
    recip = build_reciprocity(
        ctx, coil, target, centers_mm, distance_mm=distance_mm, spacing_mm=spacing_mm
    )
    return optimize_placement(
        recip, coil, centers_mm, n_samples=n_samples, distance_mm=distance_mm, didt=didt
    )


def optimize_placement(
    recip: ReciprocityField,
    coil: Coil,
    centers_mm: npt.ArrayLike,
    *,
    n_samples: int = 9,
    resolution: int = 512,
    distance_mm: float = 4.0,
    didt: float = 1e6,
) -> OptimizeResult:
    """Find the coil position and in-plane angle maximizing the target objective.

    ``centers_mm`` is a set of candidate scalp targets ``(C,3)`` (each is projected onto the scalp).
    The rotation is optimized in closed form: ``E_d(θ)`` is sampled at ``n_samples`` (odd) angles,
    Fourier-interpolated to ``resolution`` points, and ``|E(θ)|²`` is maximized on that fine grid.
    """
    if n_samples % 2 == 0:
        raise ValueError("n_samples must be odd for symmetric Fourier interpolation.")

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
    obj = _objective(e, recip.magnitude)  # (C,)
    handle = proj + cp.cos(theta)[:, None] * u + cp.sin(theta)[:, None] * v

    gi = int(cp.argmax(obj))
    return OptimizeResult(
        best_center_mm=cp.asnumpy(proj[gi]),
        best_handle_mm=cp.asnumpy(handle[gi]),
        best_angle_rad=float(theta[gi]),
        best_E=cp.asnumpy(e[gi]),
        best_objective=float(obj[gi]),
        centers_mm=proj,
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
    s, mom = place_coil_dipoles_batch(m, coil.positions_m, coil.moments)
    return _interp_reduce(recip, s, mom, didt)
