"""Place the coil and compute its magnetic-dipole dA/dt field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import numpy.typing as npt

from cunibs.solver import dadt_nbody

if TYPE_CHECKING:
    from cunibs.fem.solve import SolverContext

MU0_OVER_4PI = 1e-7  # μ0 / 4π in T·m/A, the magnetic-dipole vector-potential constant
DADT_COMPUTE_DTYPE = cp.float32


def _safe_div(num: cp.ndarray, den: cp.ndarray) -> cp.ndarray:
    """Elementwise ``num/den`` with 0 where ``den == 0`` (cupy has no ``divide(where=)``)."""
    ok = den != 0
    return cp.where(ok, num / cp.where(ok, den, 1.0), 0.0)


def _closest_point_on_triangles(
    p: cp.ndarray, a: cp.ndarray, b: cp.ndarray, c: cp.ndarray
) -> cp.ndarray:
    """Closest point to ``p`` on each triangle (a,b,c) (Ericson, Real-Time Collision)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = p - b
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    pc = p - c
    d5 = (ab * pc).sum(1)
    d6 = (ac * pc).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    inv = _safe_div(1.0, denom)
    v = vb * inv
    w = vc * inv
    out = a + v[:, None] * ab + w[:, None] * ac
    # Ericson's region tests replace the interior result at edges and vertices.
    reg_a = (d1 <= 0) & (d2 <= 0)
    reg_b = (d3 >= 0) & (d4 <= d3)
    reg_c = (d6 >= 0) & (d5 <= d6)
    e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t_ab = _safe_div(d1, d1 - d3)
    t_ac = _safe_div(d2, d2 - d6)
    t_bc = _safe_div(d4 - d3, (d4 - d3) + (d5 - d6))
    out = cp.where(e_bc[:, None], b + t_bc[:, None] * (c - b), out)
    out = cp.where(e_ac[:, None], a + t_ac[:, None] * ac, out)
    out = cp.where(e_ab[:, None], a + t_ab[:, None] * ab, out)
    out = cp.where(reg_c[:, None], c, out)
    out = cp.where(reg_b[:, None], b, out)
    out = cp.where(reg_a[:, None], a, out)
    return out


def _project_point_on_skin(
    point_mm: cp.ndarray, a: cp.ndarray, b: cp.ndarray, c: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return the closest skin point and its triangle index."""
    proj = _closest_point_on_triangles(point_mm[None, :], a, b, c)
    tri = cp.argmin(cp.linalg.norm(proj - point_mm, axis=1))
    return proj[tri], tri


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
    center, tri = _project_point_on_skin(
        cp.asarray(center_mm, cp.float64), ctx.skin_a, ctx.skin_b, ctx.skin_c
    )

    y = cp.asarray(pos_ydir_mm, cp.float64) - center
    y /= cp.linalg.norm(y)
    normal = ctx.skin_tri_normals[tri]
    z = -normal
    y -= z * y.dot(z)
    y /= cp.linalg.norm(y)
    x = cp.cross(y, z)
    c = center + distance_mm * normal

    m = cp.zeros((4, 4), cp.float64)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = c
    m[3, 3] = 1.0
    return cp.asnumpy(m)


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
