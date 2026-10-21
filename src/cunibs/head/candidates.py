"""Candidate coil positions on the scalp, for an ADM or brute-force placement search.

Candidates are sampled from the scalp surface itself rather than rastered in a tangent plane and
projected, so the radius test *is* the disc and every candidate already lies on the surface. Where
the requested spacing is finer than the mesh's own facets, those facets are subdivided; midpoints
of a planar triangle stay in it, so refinement needs no projection either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from cunibs.fem.solve import context_of
from cunibs.head.geometry import _project_to_scalp

if TYPE_CHECKING:
    import numpy.typing as npt

    from cunibs.fem.solve import SolverContext
    from cunibs.simulation import Subject


@dataclass(frozen=True)
class ScalpGrid:
    """Candidate positions on the scalp, and what the mesh actually supported.

    Pass it straight to :func:`cunibs.adm.optimize`, which converts it to its ``centers_mm``
    array. ``achieved_spacing_mm`` is the median nearest-neighbour distance of what came back: a
    request finer than the mesh's own facets is met by subdividing them, a coarser one exactly.
    """

    centers_mm: npt.NDArray[np.float64]
    center_mm: npt.NDArray[np.float64]
    outward_normal: npt.NDArray[np.float64]
    radius_mm: float
    requested_spacing_mm: float
    achieved_spacing_mm: float
    n_source_triangles: int

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[np.float64]:
        return np.array(self.centers_mm, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        return int(self.centers_mm.shape[0])


def _tangent_basis(centroids: cp.ndarray, normal: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """Two in-plane axes from an SVD of the sampled patch.

    Fitted to the patch rather than taken from the normal at the landmark: where the scalp is
    nearly flat, in-plane axes derived from a single normal swing with tiny changes in it, while
    the patch fit is stable. Only the plane's orientation is used, so its handedness is free.
    """
    if centroids.shape[0] >= 3:
        centred = centroids - centroids.mean(axis=0)
        _, s, vh = cp.linalg.svd(centred, full_matrices=False)
        s = cp.asnumpy(s)
        if s[1] > 1e-12 * s[0]:
            return vh[0], vh[1]
    # A patch that is a point or a line does not fix a plane, so the axes come from the normal.
    ref = cp.eye(3, dtype=cp.float64)[int(cp.argmin(cp.abs(normal)))]
    u = cp.cross(normal, ref)
    u /= cp.linalg.norm(u)
    return u, cp.cross(normal, u)


def _subdivide(verts: cp.ndarray, levels: int) -> cp.ndarray:
    """Split each triangle into four on its edge midpoints, ``levels`` times."""
    for _ in range(levels):
        a, b, c = verts[:, 0], verts[:, 1], verts[:, 2]
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        verts = cp.concatenate(
            [
                cp.stack([a, ab, ca], axis=1),
                cp.stack([ab, b, bc], axis=1),
                cp.stack([ca, bc, c], axis=1),
                cp.stack([ab, bc, ca], axis=1),
            ]
        )
    return verts


def _thin_to_spacing(
    centroids: cp.ndarray, origin: cp.ndarray, normal: cp.ndarray, spacing_mm: float
) -> cp.ndarray:
    """Keep one candidate per ``spacing_mm`` cell of the fitted tangent plane.

    The winner is the candidate nearest its cell centre, ties going to the lower index. That is a
    total order over the candidates, so the choice does not depend on their layout in memory.
    """
    u, v = _tangent_basis(centroids, normal)
    rel = centroids - origin
    cu, cv = rel @ u, rel @ v
    iu = cp.floor(cu / spacing_mm)
    iv = cp.floor(cv / spacing_mm)
    d2 = (cu - (iu + 0.5) * spacing_mm) ** 2 + (cv - (iv + 0.5) * spacing_mm) ** 2

    cell = (iu - iu.min()) * (iv.max() - iv.min() + 1) + (iv - iv.min())
    order = cp.lexsort(cp.stack([cp.arange(cell.shape[0], dtype=cp.float64), d2, cell]))
    ordered = cell[order]
    first = cp.concatenate([cp.array([True]), ordered[1:] != ordered[:-1]])
    return cp.sort(order[first])


_NN_BLOCK = 1024


def _median_nn(points: npt.NDArray[np.float64]) -> float:
    """Median nearest-neighbour distance, the resolution the returned set actually has."""
    n = points.shape[0]
    if n < 2:
        return float("nan")
    # A block of rows at a time, so a dense grid never holds every pairwise distance at once.
    nearest = np.empty(n)
    for start in range(0, n, _NN_BLOCK):
        rows = points[start : start + _NN_BLOCK]
        d2 = ((rows[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        d2[np.arange(rows.shape[0]), np.arange(start, start + rows.shape[0])] = np.inf
        nearest[start : start + rows.shape[0]] = d2.min(axis=1)
    return float(np.median(np.sqrt(nearest)))


def scalp_grid(
    subject: Subject | SolverContext,
    center_mm: npt.ArrayLike,
    radius_mm: float = 20.0,
    spacing_mm: float = 2.0,
) -> ScalpGrid:
    """Candidate scalp positions within ``radius_mm`` of ``center_mm``, about ``spacing_mm`` apart.

    ``center_mm`` is projected onto the scalp first, so the disc is centred on the surface rather
    than on wherever the caller's point happened to be.
    """
    if radius_mm <= 0.0 or spacing_mm <= 0.0:
        raise ValueError("radius_mm and spacing_mm must be positive.")
    ctx = context_of(subject)

    origin, inward = _project_to_scalp(ctx, center_mm)
    if origin.shape[0] != 1:
        raise ValueError("scalp_grid takes a single centre; pass one 3-vector.")
    p0, normal = origin[0], -inward[0]

    verts = ctx.skin_verts
    near = cp.linalg.norm(verts.mean(axis=1) - p0, axis=1) <= radius_mm
    n_source = int(near.sum())
    if n_source == 0:
        raise ValueError(
            f"no scalp triangles within {radius_mm} mm of the projected centre; "
            "the radius is smaller than one facet."
        )
    verts = cp.ascontiguousarray(verts[near])

    # Refine only where the mesh cannot already resolve the requested spacing: below facet size,
    # extra candidates differ only in a position the geometry does not distinguish.
    edge = float(cp.linalg.norm(verts[:, 1] - verts[:, 0], axis=1).mean())
    verts = _subdivide(verts, max(0, int(np.ceil(np.log2(max(edge / spacing_mm, 1.0))))))

    centroids = verts.mean(axis=1)
    centroids = centroids[cp.linalg.norm(centroids - p0, axis=1) <= radius_mm]
    chosen = _thin_to_spacing(centroids, p0, normal, spacing_mm)

    centers = cp.asnumpy(cp.ascontiguousarray(centroids[chosen]))
    return ScalpGrid(
        centers_mm=centers,
        center_mm=cp.asnumpy(p0),
        outward_normal=cp.asnumpy(normal),
        radius_mm=float(radius_mm),
        requested_spacing_mm=float(spacing_mm),
        achieved_spacing_mm=_median_nn(centers),
        n_source_triangles=n_source,
    )
