"""Batched coil placement helpers for ADM evaluation."""

from __future__ import annotations

import cupy as cp
import numpy.typing as npt


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
