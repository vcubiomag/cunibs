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
import cupyx.scipy.sparse as csp
import numpy as np

from cunibs._memory import build_arena
from cunibs.fem.assembly import csr_matmul
from cunibs.fem.placement import coil_dadt_block, compute_coil_transforms
from cunibs.fem.solve import MAX_BLOCK, _assemble_rhs, _pad_block, _solve_grounded_block_mat
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
# Energy below which a new direction counts as already spanned, as a fraction of the energy the
# block held before projection. The fp32 Gram eigenvalues carry noise of about 1e-7 of that energy,
# and normalising a direction this small would blow its rounding error up.
_SPAN_RTOL = 1e-6


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
    """Coil frames for the basis: points on the cap, each taken at two handles 90 degrees apart.

    The field under a figure-of-eight coil turns with its handle, to first order as
    (cos t, sin t), so a pair of orthogonal handles at each point spans the rotations there.
    """
    count = (size + 1) // 2
    base = cp.asnumpy(
        compute_coil_transforms(ctx, _cap_points(ctx, count), None, _BASIS_DISTANCE_MM)
    )
    frames = np.repeat(base, 2, axis=0)
    # The same frame turned a quarter about its normal.
    frames[0::2, :3, 0] = -base[:, :3, 1]
    frames[0::2, :3, 1] = base[:, :3, 0]
    return frames[:size]


def build_sweep_basis(ctx: SolverContext, coil: Coil, size: int = 128) -> SweepBasis:
    """Solve ``size`` placements spread over the scalp and A-orthonormalise their solutions.

    They are solved in blocks, each seeded from the basis so far, which takes a later block's
    solve to about half the iterations of the first. Each block is orthogonalised against the
    basis twice (classical Gram-Schmidt run twice keeps it orthogonal in finite precision), then
    within itself through its own Gram matrix, dropping any direction the basis already holds.
    """
    if size < 1:
        raise ValueError(f"build_sweep_basis: size must be positive, got {size}")
    with build_arena() as arena:
        solver = ctx.solver
        n = int(solver.idx.shape[0])
        a = _stiffness(ctx)
        transforms = _basis_transforms(ctx, size)
        q = cp.empty((n, 0), dtype=cp.float32)
        for start in range(0, size, MAX_BLOCK):
            k = min(MAX_BLOCK, size - start)
            dadt = coil_dadt_block(
                coil.positions_m, coil.moments, transforms[start : start + k], 1.0, ctx.nodes_mm
            )
            b = _pad_block(_assemble_rhs(ctx, dadt)[solver.idx, :].astype(cp.float64), k)
            del dadt
            x0 = SweepBasis(q).seed(b) if q.shape[1] else None
            x, _ = _solve_grounded_block_mat(solver, b, k, x0)
            x = x[:, :k].astype(cp.float32)
            spanned = 0.0
            if q.shape[1]:
                c = q.T @ csr_matmul(a, x)
                x -= q @ c
                x -= q @ (q.T @ csr_matmul(a, x))
                spanned = float((c * c).sum(axis=0).max())
            g = x.T @ csr_matmul(a, x)
            w, u = cp.linalg.eigh(0.5 * (g + g.T))
            keep = w > _SPAN_RTOL * (spanned + float(w[-1]))
            if bool(keep.any()):
                new = x @ (u[:, keep] / cp.sqrt(w[keep]))
                q = cp.ascontiguousarray(cp.concatenate([q, new.astype(cp.float32)], axis=1))
        q = arena.keep(q)
    return SweepBasis(q, coil.name, coil_digest(coil), size)


def coil_digest(coil: Coil) -> str:
    """A content hash of the coil's dipoles, which is what a basis depends on."""
    h = hashlib.sha256(np.ascontiguousarray(coil.positions_m, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(coil.moments, dtype=np.float64).tobytes())
    return h.hexdigest()


def _stiffness(ctx: SolverContext) -> csp.csr_matrix:
    """The reduced stiffness as an fp32 CSR, for the basis's energy inner products."""
    a = ctx.solver.matrix()
    return csp.csr_matrix((a.data.astype(cp.float32), a.indices, a.indptr), shape=a.shape)


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
