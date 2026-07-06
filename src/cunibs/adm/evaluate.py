"""Fast per-placement target E-field evaluation from a sampled reciprocity field."""

from __future__ import annotations

from collections.abc import Sequence

import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

from cunibs.adm.grid import Grid, build_grid
from cunibs.adm.place import compute_coil_transforms, place_coil_dipoles_batch
from cunibs.adm.reciprocity import AdjointField, ReciprocityField
from cunibs.coil import Coil
from cunibs.fem.placement import MU0_OVER_4PI
from cunibs.fem.solve import SolverContext
from cunibs.simulation import Placement
from cunibs.solver import dadt_nbody


def _placed_dipoles(
    ctx: SolverContext, coil: Coil, placements: Sequence[Placement]
) -> tuple[cp.ndarray, cp.ndarray]:
    """Stack placed dipole positions ``(P,N,3)`` (m) and moments ``(P,N,3)`` for all placements."""
    centers = cp.asarray([pl.center_mm for pl in placements], dtype=cp.float64)
    handles = cp.asarray([pl.handle_mm for pl in placements], dtype=cp.float64)
    dists = cp.asarray([pl.distance_mm for pl in placements], dtype=cp.float64)
    transforms = compute_coil_transforms(ctx, centers, handles, dists)
    return place_coil_dipoles_batch(transforms, coil.positions_m, coil.moments)


def grid_for_placements(
    ctx: SolverContext,
    coil: Coil,
    placements: Sequence[Placement],
    spacing_mm: float = 2.0,
    margin_mm: float = 8.0,
) -> Grid:
    """Build a Q-sampling grid covering every coil dipole across ``placements``."""
    s, _ = _placed_dipoles(ctx, coil, list(placements))
    return build_grid(s.reshape(-1, 3), spacing_mm=spacing_mm, margin_mm=margin_mm)


def _reduce(dipole_q: cp.ndarray, moments: cp.ndarray, didt: float) -> cp.ndarray:
    """``E_{p,d} = didt · Σ_j m_{p,j} · Q_d(s_{p,j})`` from per-dipole Q samples ``(D,P,N,3)``."""
    return didt * cp.einsum("dpnk,pnk->pd", dipole_q, moments)


# Cap the number of (placement × dipole) samples interpolated at once so large sweeps do not
# allocate a multi-gigabyte (D, P·N, 3) buffer; placements are processed in chunks under this bound.
_INTERP_CHUNK_SAMPLES = 1 << 23


def _interp_reduce(
    recip: ReciprocityField, s: cp.ndarray, m: cp.ndarray, didt: float
) -> cp.ndarray:
    """Target E-vector ``(P,D)`` from placed dipole positions ``s`` and moments ``m`` (both P,N,3)."""
    p, n, _ = s.shape
    n_dir = int(recip.q.shape[0])
    out = cp.empty((p, n_dir), dtype=cp.float64)
    chunk = max(1, _INTERP_CHUNK_SAMPLES // n)
    for lo in range(0, p, chunk):
        hi = min(lo + chunk, p)
        pc = hi - lo
        coords = recip.grid.world_to_index(s[lo:hi].reshape(-1, 3))  # (3, pc*N)
        dipole_q = cp.empty((n_dir, pc * n, 3), dtype=cp.float64)
        for d in range(n_dir):
            for k in range(3):
                dipole_q[d, :, k] = map_coordinates(
                    recip.q[d, ..., k], coords, order=1, mode="nearest"
                )
        out[lo:hi] = _reduce(dipole_q.reshape(n_dir, pc, n, 3), m[lo:hi], didt)
    return out


def evaluate(
    recip: ReciprocityField,
    coil: Coil,
    placements: Placement | Sequence[Placement],
    didt: float = 1e6,
) -> cp.ndarray:
    """Target E-vector (component per adjoint direction) for each placement via grid interpolation.

    Returns ``(P, D)``: for magnitude targets ``D=3`` (orthonormal basis, ``|E| = ‖row‖``); for a
    directional target ``D=1``. No FEM solve, only trilinear sampling of the cached Q-field.
    """
    single = isinstance(placements, Placement)
    pls = [placements] if single else list(placements)
    s, m = _placed_dipoles(recip.ctx, coil, pls)  # (P,N,3)
    e = _interp_reduce(recip, s, m, didt)
    return e[0] if single else e


def evaluate_exact(
    adjoint: AdjointField,
    coil: Coil,
    placements: Placement | Sequence[Placement],
    didt: float = 1e6,
    center_m: cp.ndarray | None = None,
) -> cp.ndarray:
    """Same functional as :func:`evaluate` but with ``Q`` evaluated exactly at each dipole (no grid).

    Isolates the grid-interpolation error: this is the auxiliary-dipole method *without* the
    auxiliary grid, so it must agree with the exact reciprocity functional to float32 precision.
    ``Q(s_j) = dadt_nbody(sources=mesh nodes, moments=W_n, targets=s_j)``.
    """
    single = isinstance(placements, Placement)
    pls = [placements] if single else list(placements)
    ctx = adjoint.ctx
    s, m = _placed_dipoles(ctx, coil, pls)  # (P,N,3)
    p, n, _ = s.shape

    center = cp.zeros(3) if center_m is None else cp.asarray(center_m, dtype=cp.float64)
    src = cp.ascontiguousarray(
        ((cp.asarray(ctx.nodes_mm) * 1e-3) - center[None, :]).astype(cp.float32)
    )
    sn = cp.ascontiguousarray((src * src).sum(1))
    tgt = cp.ascontiguousarray((s.reshape(-1, 3) - center[None, :]).astype(cp.float32))

    n_dir = int(adjoint.node_weights.shape[0])
    dipole_q = cp.empty((n_dir, p * n, 3), dtype=cp.float64)
    out = cp.empty((p * n, 3), dtype=cp.float32)
    stream = cp.cuda.get_current_stream().ptr
    for d in range(n_dir):
        w = adjoint.node_weights[d].astype(cp.float32)
        mp = cp.ascontiguousarray(cp.concatenate([w, cp.cross(w, src)], axis=1))
        dadt_nbody(src, mp, sn, tgt, out, 1.0, MU0_OVER_4PI, stream)
        dipole_q[d] = out.astype(cp.float64)
    e = _reduce(dipole_q.reshape(n_dir, p, n, 3), m, didt)
    return e[0] if single else e
