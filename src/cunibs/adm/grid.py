"""Regular sampling grid for the reciprocity Q-field above the scalp."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp


@dataclass
class Grid:
    """Axis-aligned regular grid in metres.

    ``origin_m`` is the coordinate of index ``(0,0,0)``, ``spacing_m`` the per-axis step, and
    ``shape`` the number of samples per axis. Coordinates are stored absolute (uncentred); callers
    that run the N-body in float32 should subtract :attr:`center_m` from both sources and samples.
    """

    origin_m: cp.ndarray  # (3,) float64
    spacing_m: cp.ndarray  # (3,) float64
    shape: tuple[int, int, int]

    @property
    def n_points(self) -> int:
        return int(self.shape[0] * self.shape[1] * self.shape[2])

    @property
    def center_m(self) -> cp.ndarray:
        return self.origin_m + 0.5 * self.spacing_m * (cp.asarray(self.shape) - 1)

    def points_m(self) -> cp.ndarray:
        """All grid points as ``(n_points, 3)`` in C order (matches ``reshape(shape+(3,))``)."""
        ax = [
            self.origin_m[i] + self.spacing_m[i] * cp.arange(self.shape[i], dtype=cp.float64)
            for i in range(3)
        ]
        gx, gy, gz = cp.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
        return cp.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def world_to_index(self, points_m: cp.ndarray) -> cp.ndarray:
        """Fractional index coordinates ``(3, N)`` for ``map_coordinates`` (order matches axes)."""
        idx = (points_m - self.origin_m[None, :]) / self.spacing_m[None, :]
        return cp.ascontiguousarray(idx.T)


def build_grid(dipole_positions_m: cp.ndarray, spacing_mm: float, margin_mm: float) -> Grid:
    """Build a grid covering ``dipole_positions_m`` (metres) dilated by ``margin_mm``."""
    spacing_m = spacing_mm * 1e-3
    margin_m = margin_mm * 1e-3
    lo = dipole_positions_m.min(0) - margin_m
    hi = dipole_positions_m.max(0) + margin_m
    span = hi - lo
    nx, ny, nz = (int(cp.ceil(span[i] / spacing_m)) + 1 for i in range(3))
    return Grid(
        origin_m=cp.ascontiguousarray(lo),
        spacing_m=cp.ascontiguousarray(cp.full(3, spacing_m, dtype=cp.float64)),
        shape=(nx, ny, nz),
    )
