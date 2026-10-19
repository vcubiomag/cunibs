"""Fast per-placement target E-field evaluation from a sampled reciprocity field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from cunibs.adm.grid import Grid, build_grid
from cunibs.adm.place import place_coil_dipoles_batch
from cunibs.fem.placement import MU0_OVER_4PI, compute_coil_transforms
from cunibs.simulation import Placement
from cunibs.solver import adm_evaluate, adm_mark, dadt_nbody

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from cunibs.adm.reciprocity import AdjointField, ReciprocityField
    from cunibs.coil import Coil
    from cunibs.fem.solve import SolverContext


def _placement_frames(ctx: SolverContext, placements: Sequence[Placement]) -> cp.ndarray:
    """Coil-to-head affines ``(P,4,4)`` for ``placements``, translation in mm."""
    centers = cp.asarray([pl.center_mm for pl in placements], dtype=cp.float64)
    handles = cp.asarray([pl.handle_mm for pl in placements], dtype=cp.float64)
    dists = cp.asarray([pl.distance_mm for pl in placements], dtype=cp.float64)
    return compute_coil_transforms(ctx, centers, handles, dists)


def _placed_dipoles(
    ctx: SolverContext, coil: Coil, placements: Sequence[Placement]
) -> tuple[cp.ndarray, cp.ndarray]:
    """Stack placed dipole positions ``(P,N,3)`` (m) and moments ``(P,N,3)`` for all placements."""
    transforms = _placement_frames(ctx, placements)
    return place_coil_dipoles_batch(transforms, coil.positions_m, coil.moments)


def grid_for_placements(
    ctx: SolverContext,
    coil: Coil,
    placements: Sequence[Placement],
    spacing_mm: float = 3.0,
    margin_mm: float = 8.0,
) -> Grid:
    """Build a Q-sampling grid covering every coil dipole across ``placements``."""
    s, _ = _placed_dipoles(ctx, coil, list(placements))
    return build_grid(s.reshape(-1, 3), spacing_mm=spacing_mm, margin_mm=margin_mm)


def _reduce(dipole_q: cp.ndarray, moments: cp.ndarray, didt: float) -> cp.ndarray:
    """``E_{p,d} = didt · Σ_j m_{p,j} · Q_d(s_{p,j})`` from per-dipole Q samples ``(D,P,N,3)``."""
    return didt * cp.einsum("dpnk,pnk->pd", dipole_q, moments)


# Slack on the fractional index bounds. At a 3 mm spacing this is 3 nm (noise). The kernel applies
# the same slack.
_INDEX_TOL = 1e-6


def _require_in_grid(
    grid: Grid, coords: cp.ndarray, offset: int, n_dip: int, n_total: int
) -> None:
    """Raise if any sample falls outside ``grid``.

    Interpolation runs with ``mode="nearest"``, so an out-of-grid dipole would be clamped to the
    grid edge and yield a wrong field rather than an error.
    """
    # >= / <= rather than a negated < / >: a non-finite coordinate compares false either way, so
    # it fails closed instead of reaching the interpolation.
    ok = (coords >= -_INDEX_TOL) & (coords <= grid.max_index[:, None] + _INDEX_TOL)
    if bool(ok.all()):
        return

    bad = ~ok
    axis = int(cp.argmax(bad.any(axis=1)))
    flat = int(cp.argmax(bad[axis]))
    sample = offset + flat // n_dip
    origin_mm = float(grid.origin_m[axis]) * 1e3
    upper_mm = float(grid.upper_m[axis]) * 1e3
    pos_mm = origin_mm + float(coords[axis, flat]) * float(grid.spacing_m[axis]) * 1e3
    outside_mm = origin_mm - pos_mm if pos_mm < origin_mm else pos_mm - upper_mm
    raise ValueError(
        f"A coil dipole falls outside the reciprocity grid: sample {sample} of {n_total} on "
        f"axis {axis} ({'xyz'[axis]}), dipole at {pos_mm:.3f} mm against a grid spanning "
        f"[{origin_mm:.3f}, {upper_mm:.3f}] mm ({outside_mm:.3f} mm outside). Rebuild the "
        "reciprocity field so the grid covers it: widen centers_mm or margin_mm, or "
        "evaluate at the distance_mm the grid was built with."
    )


def _raise_off_grid(recip: ReciprocityField, coil: Coil, frames: cp.ndarray, cand: int) -> None:
    """Raise the off-grid error for candidate ``cand``, with its own dipoles' coordinates."""
    s, _ = place_coil_dipoles_batch(frames[cand : cand + 1], coil.positions_m, coil.moments)
    n_dip = int(s.shape[1])
    coords = recip.grid.world_to_index(s.reshape(-1, 3))
    _require_in_grid(recip.grid, coords, cand, n_dip, int(frames.shape[0]))
    raise AssertionError(
        "the evaluator reported a dipole off the grid that the host check passed"
    )


def _evaluate_native(
    recip: ReciprocityField, coil: Coil, frames: cp.ndarray, didt: float
) -> cp.ndarray:
    """Target E-vectors ``(P, D)`` for ``frames``, sampling any cell a frame needs that is not yet.

    One warp per candidate transforms the coil, interpolates every direction at each dipole and
    reduces the dot products, all on chip. A dipole in a cell the coverage did not sample is
    reported rather than read; its frame's points are then sampled and the call re-run, so a frame
    outside the coverage costs its own sampling once and is never answered from zeros.
    """
    grid = recip.grid
    frames = cp.ascontiguousarray(frames, dtype=cp.float64).reshape(-1, 4, 4)
    pos = cp.ascontiguousarray(cp.asarray(coil.positions_m, dtype=cp.float32))
    mom = cp.ascontiguousarray(cp.asarray(coil.moments, dtype=cp.float32))
    origin = [float(x) for x in cp.asnumpy(grid.origin_m)]
    spacing = [float(x) for x in cp.asnumpy(grid.spacing_m)]
    shape = list(grid.shape)
    stream = cp.cuda.get_current_stream().ptr
    n_dir = recip.n_directions
    out = cp.empty((int(frames.shape[0]), n_dir), dtype=cp.float64)
    fault = cp.zeros(2, dtype=cp.int32)
    q = recip.q.reshape(-1)
    for attempt in range(2):
        fault[:] = 0
        adm_evaluate(
            frames,
            pos,
            mom,
            q,
            recip.cell_ok,
            origin,
            spacing,
            shape,
            n_dir,
            float(didt),
            out,
            fault,
            stream,
        )
        kind, cand = (int(x) for x in cp.asnumpy(fault))
        if kind == 0:
            return out
        if kind == 1:
            _raise_off_grid(recip, coil, frames, cand)
        if attempt == 0:
            need = cp.zeros(grid.n_points, dtype=cp.uint8)
            adm_mark(frames, pos, origin, spacing, shape, need, stream)
            recip.sample_points(cp.flatnonzero(need & (recip.sampled ^ 1)))
    raise AssertionError(
        "cells were still unsampled after sampling every point the frames need"
    )


def evaluate(
    recip: ReciprocityField,
    coil: Coil,
    placements: Placement | Sequence[Placement],
    didt: float = 1e6,
) -> npt.NDArray[np.float64]:
    """Target E-vector (component per adjoint direction) for each placement via grid interpolation.

    Returns ``(P, D)``, or ``(D,)`` for a single placement: for magnitude targets ``D=3``
    (orthonormal basis, ``|E| = ‖row‖``); for a directional target ``D=1``. No FEM solve, only
    trilinear sampling of the cached Q-field.

    ``didt`` is the coil current's rate of change in A/s; the field is linear in it.
    """
    single = isinstance(placements, Placement)
    pls = [placements] if single else list(placements)
    e = cp.asnumpy(_evaluate_native(recip, coil, _placement_frames(recip.ctx, pls), didt))
    return e[0] if single else e


def evaluate_frames(
    recip: ReciprocityField,
    coil: Coil,
    transforms: npt.ArrayLike,
    didt: float = 1e6,
) -> npt.NDArray[np.float64]:
    """Target E-vector ``(P,D)`` for coil-to-head affines given directly, one per placement.

    ``transforms`` is ``(P,4,4)`` with the rotation in ``[:3,:3]`` and a translation in **mm** in
    ``[:3,3]``. Unlike :func:`evaluate`, the frame is used verbatim rather than re-derived from a
    scalp projection, so a caller comparing against another code can adopt that code's frames
    exactly instead of reproducing its projection rule.
    """
    return cp.asnumpy(
        _evaluate_native(recip, coil, cp.asarray(transforms, dtype=cp.float64), didt)
    )


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

    ``didt`` is the coil current's rate of change in A/s; the field is linear in it.
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
