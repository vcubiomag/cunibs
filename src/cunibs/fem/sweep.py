"""A per-subject basis that seeds every forward solve of a long sweep.

The potential a placement induces is a smooth, global field, so the solutions of placements
spread over the scalp span most of any other placement's solution. With an A-orthonormal basis Q
of such solutions, ``x0 = Q Q^T b`` is the Galerkin solution over span(Q): the best start that
span offers in the energy norm CG minimises.

The basis is fixed by the subject and the coil it was built with, never by the placements being
solved, so a placement's result does not depend on the sweep it is part of. It does depend on
whether a basis is present: a seeded solve lands at a different point inside the same tolerance.
Building it costs ``size`` solves, which a sweep repays after a few hundred placements.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from cunibs.fem.assembly import csr_matmul
from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transforms
from cunibs.solver import sweep_seed, sweep_seed_work_size

if TYPE_CHECKING:
    from cunibs.coil import Coil
    from cunibs.fem.solve import SolverContext

# Basis placements' offset from the scalp, as a default Placement takes.
_BASIS_DISTANCE_MM = 4.0
# The basis points lie on the cap of a sphere about the head's centroid, down to this height
# (in units of the radius): far enough below the equator to cover the temporal targets.
_CAP_BOTTOM = -0.3
# Radius of that sphere: outside any head, so every point projects onto the scalp beneath it.
_CAP_RADIUS_MM = 150.0
# How far from its point a basis placement's handle point sits; any distance gives the same axis.
_HANDLE_MM = 30.0
# Relative eigenvalue below which a new direction counts as already spanned.
_SPAN_RTOL = 1e-10


@dataclass
class SweepBasis:
    """An A-orthonormal basis ``q`` (n_red, m), fp32, of forward solutions on one subject.

    ``coil_name``, ``coil_digest`` and ``requested_size`` identify what it was built from;
    ``size`` can be smaller than ``requested_size`` when some placements added no new direction.
    """

    q: cp.ndarray
    coil_name: str = ""
    coil_digest: str = ""
    requested_size: int = 0

    @property
    def size(self) -> int:
        return int(self.q.shape[1])

    def seed(self, b: cp.ndarray) -> cp.ndarray:
        """``q q^T b`` for a reduced (n_red, k) right-hand-side block, k in {1, 2, 4, 8}."""
        n, k = b.shape
        work = cp.empty(int(sweep_seed_work_size(n, self.size, k)), dtype=cp.float64)
        x0 = cp.empty((n, k), dtype=cp.float64)
        sweep_seed(self.q, cp.ascontiguousarray(b), work, x0, cp.cuda.get_current_stream().ptr)
        return x0


def _cap_points(ctx: SolverContext, count: int) -> np.ndarray:
    """``count`` quasi-uniform points on the cap, as a Fibonacci spiral (deterministic)."""
    centre = cp.asnumpy(ctx.nodes_mm.mean(axis=0))
    i = np.arange(count) + 0.5
    z = 1.0 - (1.0 - _CAP_BOTTOM) * i / count
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    return centre + _CAP_RADIUS_MM * np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _basis_transforms(ctx: SolverContext, size: int) -> np.ndarray:
    """Coil frames for the basis: points on the cap, each taken at two handle angles 90 apart.

    The field under a figure-of-eight coil turns with its handle, to first order as
    (cos t, sin t), so a pair of orthogonal handles at each point spans the rotations there.
    """
    count = (size + 1) // 2
    points = _cap_points(ctx, count)
    dist = np.full(count, _BASIS_DISTANCE_MM, dtype=np.float64)
    # Handle-free frames give each point an in-plane pair of axes to take the angles from.
    base = cp.asnumpy(compute_coil_transforms(ctx, points, None, dist))
    idx = np.repeat(np.arange(count), 2)[:size]
    angles = np.tile([0.0, np.pi / 2], count)[:size]
    x, y = base[idx, :3, 0], base[idx, :3, 1]
    handles = points[idx] + _HANDLE_MM * (
        np.cos(angles)[:, None] * x + np.sin(angles)[:, None] * y
    )
    return cp.asnumpy(compute_coil_transforms(ctx, points[idx], handles, dist[idx]))


def build_sweep_basis(ctx: SolverContext, coil: Coil, size: int = 128) -> SweepBasis:
    """Solve ``size`` placements spread over the scalp and A-orthonormalise their solutions.

    They are solved in blocks, each seeded from the basis so far, which takes a later block's
    solve to about half the iterations of the first. Each block is orthogonalised against the
    basis twice (classical Gram-Schmidt run twice keeps it orthogonal in finite precision), then
    within itself through its own Gram matrix, dropping any direction the basis already holds.
    """
    from cunibs.fem.solve import MAX_BLOCK, _assemble_rhs, _pad_block

    if size < 1:
        raise ValueError(f"build_sweep_basis: size must be positive, got {size}")
    solver = ctx.solver
    stream = cp.cuda.get_current_stream().ptr
    n = int(solver.idx.shape[0])
    a = _stiffness(ctx)
    transforms = _basis_transforms(ctx, size)
    q = cp.empty((n, 0), dtype=cp.float32)
    for start in range(0, size, MAX_BLOCK):
        k = min(MAX_BLOCK, size - start)
        dadt = cp.empty((ctx.n_nodes, k, 3), dtype=cp.float32)
        for i in range(k):
            coil_dadt_at_nodes(
                coil.positions_m,
                coil.moments,
                transforms[start + i],
                1.0,
                ctx.nodes_mm,
                out=dadt,
                column=i,
            )
        b = _pad_block(_assemble_rhs(ctx, dadt)[solver.idx, :].astype(cp.float64), k)
        del dadt
        x0 = SweepBasis(q).seed(b) if q.shape[1] else None
        x = cp.empty_like(b)
        solver.pcg.solve_mixed_block(
            solver.precond, b, x, solver.tolerance, solver.max_iters, stream, x0
        )
        x = x[:, :k].astype(cp.float32)
        for _ in range(2):
            if q.shape[1]:
                x -= q @ (q.T @ csr_matmul(a, x))
        g = x.T @ csr_matmul(a, x)
        w, u = cp.linalg.eigh(0.5 * (g + g.T))
        keep = w > _SPAN_RTOL * float(w[-1])
        if bool(keep.any()):
            new = x @ (u[:, keep] / cp.sqrt(w[keep]))
            q = cp.ascontiguousarray(cp.concatenate([q, new.astype(cp.float32)], axis=1))
    return SweepBasis(q, coil.name, coil_digest(coil), size)


def coil_digest(coil: Coil) -> str:
    """A content hash of the coil's dipoles, which is what a basis depends on."""
    h = hashlib.sha256(np.ascontiguousarray(coil.positions_m, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(coil.moments, dtype=np.float64).tobytes())
    return h.hexdigest()


def _stiffness(ctx: SolverContext):
    """The reduced stiffness as an fp32 CSR, for the basis's energy inner products."""
    import cupyx.scipy.sparse as csp

    solver = ctx.solver
    n = int(solver.idx.shape[0])
    return csp.csr_matrix(
        (solver.values.astype(cp.float32), solver.col_idx, solver.row_ptr), shape=(n, n)
    )


def ensure_sweep_basis(ctx: SolverContext, coil: Coil, size: int = 128) -> SweepBasis:
    """Return ``ctx``'s sweep basis, building it on first use; every later block solve seeds from it.

    A basis already present is returned for the same coil and size, and anything else raises:
    replacing it would change where the solves already run against it landed.
    """
    basis = ctx.sweep_basis
    if basis is None:
        ctx.sweep_basis = build_sweep_basis(ctx, coil, size)
        return ctx.sweep_basis
    if basis.coil_digest != coil_digest(coil) or basis.requested_size != size:
        raise ValueError(
            f"this subject already seeds its solves from a basis of {basis.requested_size} "
            f"placements of coil {basis.coil_name!r}. Call Subject.free() before preparing "
            "another."
        )
    return basis
