"""The Monte Carlo conductivity-UQ sampling loop."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np

from cunibs import metrics
from cunibs.fem.assembly import GM_TAG, conductivity_per_tet
from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform
from cunibs.fem.recovery import RecoveredField, RecoveryOperator, apply_recovery_into
from cunibs.fem.solve import (
    MAX_BLOCK,
    SolverConvergenceError,
    _check_operator,
    _pad_block,
)
from cunibs.solver import (
    accumulate_moments_block,
    dadt_node_to_element,
    reconstruct_e_block,
    rhs_assemble,
    row_dot,
)
from cunibs.uq.conductivity.config import ConductivityUQConfig, sample_conductivities
from cunibs.uq.conductivity.result import ConductivityUQResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cunibs.adm.target import ResolvedTarget
    from cunibs.coil import Coil
    from cunibs.fem.solve import SolverContext
    from cunibs.simulation import Placement
    from cunibs.uq.conductivity.assembly import ConductivityUQPrecompute


_NO_ROIS: Mapping[str, ResolvedTarget] = MappingProxyType({})

# Tolerance for the sensitivity solves that span the projection basis. Deliberately far looser
# than the draw tolerance: the basis only has to point in the right directions for an initial
# guess that CG then refines to `pre.tolerance` anyway, so its accuracy sets the rate and never
# the answer.
_SENSITIVITY_TOL = 1e-3


def _dadt_node_to_elm(dadt_nodes: cp.ndarray, tet_nodes: cp.ndarray) -> cp.ndarray:
    """Average nodal dA/dt over each tetrahedron (σ-independent, computed once)."""
    dadt_elm = cp.empty((int(tet_nodes.shape[0]), 3), dtype=cp.float32)
    dadt_node_to_element(
        cp.ascontiguousarray(dadt_nodes),
        tet_nodes,
        dadt_elm,
        cp.cuda.get_current_stream().ptr,
    )
    return dadt_elm


def _placement_rhs(
    ctx: SolverContext, pre: ConductivityUQPrecompute, dadt_elm: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """Per-tissue RHS decomposition for one placement.

    The RHS is linear in σ exactly like the matrix: ``b(σ) = b_base + Σ_t σ_t·b_t``, where ``b_t``
    is the RHS assembled from tissue ``t``'s elements alone. Precomputing these ``P+1`` vectors once
    turns the per-sample RHS from a full ``rhs_assemble`` over every tet into a small GEMV.
    """
    stream = cp.cuda.get_current_stream().ptr
    ptr, idx = ctx.node2corner_ptr, ctx.node2corner_idx
    p = len(pre.perturbed_tags)
    b_tissue = cp.empty((p, ctx.n_nodes), dtype=cp.float32)
    for i, tag in enumerate(pre.perturbed_tags):
        neg_vc = cp.ascontiguousarray(-(ctx.vols * (ctx.tet_tags == tag).astype(cp.float32)))
        rhs_assemble(dadt_elm, ctx.g, neg_vc, ptr, idx, b_tissue[i], stream)

    base_cond = conductivity_per_tet(ctx.tet_tags)
    base_cond[cp.isin(ctx.tet_tags, cp.asarray(pre.perturbed_tags))] = 0.0
    b_base = cp.empty(ctx.n_nodes, dtype=cp.float32)
    neg_vc0 = cp.ascontiguousarray(-(ctx.vols * base_cond.astype(cp.float32)))
    rhs_assemble(dadt_elm, ctx.g, neg_vc0, ptr, idx, b_base, stream)
    return b_base, b_tissue


def _solve_columns(
    pre: ConductivityUQPrecompute, rhs: cp.ndarray, tolerance: float, stream: int
) -> cp.ndarray:
    """Solve the frozen nominal system against each column of ``rhs``, in block widths.

    The columns are independent, so they go through the block solver purely to share the
    stiffness and hierarchy reads.
    """
    n, k = rhs.shape
    out = cp.empty((n, k), dtype=cp.float64)
    for start in range(0, k, MAX_BLOCK):
        width = min(MAX_BLOCK, k - start)
        b = _pad_block(cp.ascontiguousarray(rhs[:, start : start + width]), width)
        x = cp.empty_like(b)
        pre.pcg.solve_mixed_block(pre.precond, b, x, tolerance, pre.max_iters, stream)
        out[:, start : start + width] = x[:, :width]
    return out


def _dominant_left_basis(a: cp.ndarray, rtol: float = 1e-8) -> cp.ndarray:
    """Orthonormal basis for the dominant column space of a tall, skinny ``a``.

    ``a = QR`` with ``Q`` orthonormal, so ``R`` carries ``a``'s singular values exactly and ``a``'s
    left singular vectors are ``Q·U_R``; only the P-by-P ``R`` reaches an SVD. Taking the SVD of
    ``a`` itself is the same factorisation, but cuSOLVER sizes ``gesvd``'s workspace off the *row*
    count and reserves gigabytes at mesh-sized rows, where ``geqrf``'s workspace is row-independent.
    Going through ``aᵀa`` would be cheaper still and is not backward stable: it squares the
    condition number, and the sensitivities decay steeply enough across tissues to lose the tail.
    """
    q, r = cp.linalg.qr(a, mode="reduced")
    u, s, _ = cp.linalg.svd(r, full_matrices=False)
    keep = int((s > rtol * float(s[0])).sum())
    return cp.ascontiguousarray(q @ u[:, :keep])


def _sensitivity_basis(
    pre: ConductivityUQPrecompute,
    b_tissue: cp.ndarray,
    ax_tissue: cp.ndarray,
    stream: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Orthonormal basis for the span of dx/dsigma at the ensemble centre, and its frozen E^-1.

    Both the matrix and the RHS are linear in sigma, so differentiating ``A(s)x(s) = b(s)`` at
    nominal sigma gives one system per perturbed tissue:

        A_nom * dx/ds_t = b_t - A_t * x_nom.

    That is the subspace a draw's correction lives in, to first order, and it is P-dimensional
    for P perturbed tissues. Both right-hand sides are already to hand: ``b_t`` is a row of the
    per-tissue RHS decomposition, and ``A_t x_nom`` is the same product the per-draw residual
    shortcut needs.

    ``_dominant_left_basis`` drops directions two tissues share, so the basis is P columns wide in
    the normal case. A placement that couples to nothing leaves every sensitivity zero and the
    basis empty; an empty basis contributes an empty correction, so there is nothing to guard.
    """
    n_red = int(pre.idx.shape[0])
    rhs = cp.ascontiguousarray(
        b_tissue[:, pre.idx].astype(cp.float64).T - ax_tissue  # (n_red, P)
    )
    sens = _solve_columns(pre, rhs, _SENSITIVITY_TOL, stream)

    w = _dominant_left_basis(sens)
    a_nom = csp.csr_matrix((pre.nominal_data, pre.indices, pre.indptr), shape=(n_red, n_red))
    return w, cp.linalg.inv(w.T @ (a_nom @ w))


class _InitialGuess:
    """The x0 the sampling loop starts every draw from.

    A draw starts at the solved ensemble-centre solution, corrected by the Galerkin projection of
    its own residual onto the sensitivity subspace:

        x0 = x_nominal + W·E⁻¹·Wᵀ·r0,   r0 = b(σ) − A(σ)·x_nominal,   E = Wᵀ·A_nominal·W.

    The correction is A_nominal-optimal over ``span(W)`` for the residual it is handed. The drawn
    system is A(σ), so that is optimal for the wrong operator, but the samples cluster at the
    centre.

    ``x_nominal`` and the basis are pure functions of the mesh, the placement and the nominal
    conductivities, and the correction reads nothing else but the draw's own σ. Keep it that way.
    x0 cannot move a draw outside ``pre.tolerance``, but it does decide where inside it the draw
    lands, so anything reaching x0 from the ensemble — a size-dependent cost heuristic, a basis
    accumulated over draws — makes a draw's last digits depend on the batch it was solved in.
    """

    def __init__(
        self,
        pre: ConductivityUQPrecompute,
        b_tissue: cp.ndarray,
        x_nominal: cp.ndarray,
        stream: int,
    ) -> None:
        self._x_nominal = x_nominal
        n_red = int(pre.idx.shape[0])
        shape = (n_red, n_red)
        # r0 is linear in σ, so A_base·x_nom and each A_t·x_nom are formed once here rather
        # than per draw; _sensitivity_basis needs the same products for its right-hand sides.
        a_base = csp.csr_matrix((pre.base_data, pre.indices, pre.indptr), shape=shape)
        self._ax_base = a_base @ x_nominal
        # Held (n_red, P), so a draw's sum over tissues is a short reduction along each row.
        self._ax_tissue = cp.ascontiguousarray(
            cp.stack(
                [
                    csp.csr_matrix((data, pre.indices, pre.indptr), shape=shape) @ x_nominal
                    for data in pre.tissue_data
                ],
                axis=1,
            )
        )
        self._w, self._einv = _sensitivity_basis(pre, b_tissue, self._ax_tissue, stream)
        self._buf = cp.empty(n_red, dtype=cp.float64)
        self._tmp = cp.empty(n_red, dtype=cp.float64)

    def for_draw(self, sigma: cp.ndarray, b_red: cp.ndarray) -> cp.ndarray:
        """The initial guess for one draw."""
        # The two products with one output per row go through row_dot: matmul sends this shape to
        # a tensor-op GEMM that costs a millisecond each on a part without fp64 tensor throughput.
        stream = cp.cuda.get_current_stream().ptr
        row_dot(
            self._ax_tissue, cp.ascontiguousarray(sigma, dtype=cp.float64), self._tmp, stream
        )
        r0 = b_red - (self._ax_base + self._tmp)
        y = cp.ascontiguousarray(self._einv @ (self._w.T @ r0))
        row_dot(self._w, y, self._tmp, stream)
        self._buf[:] = self._x_nominal + self._tmp
        return self._buf


def run_conductivity_uq(
    ctx: SolverContext,
    pre: ConductivityUQPrecompute,
    coil: Coil,
    placement: Placement,
    config: ConductivityUQConfig,
    didt: float = 1e6,
    record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
    focality_frac: float = 0.5,
    *,
    recovery: RecoveryOperator | None = None,
) -> ConductivityUQResult:
    """Solve one placement across ``config.n_samples`` conductivity draws; return |E| moments.

    ``didt`` is the coil current's rate of change in A/s; the field is linear in it.

    The coil field (``dadt_elm``) and the per-tissue RHS/stiffness components are σ-independent and
    built once. Each sample re-weights the matrix and RHS by the sampled conductivities (two small
    GEMVs), then solves against a preconditioner built once at the nominal (ensemble-centre) σ,
    which the i.i.d. samples stay clustered around. A draw that misses tolerance falls back to a
    per-draw preconditioner rebuild, so accuracy never depends on the preconditioner tracking the
    sample.

    Every draw's gray-matter peak ``|E|``, focality, and peak location are recorded — the
    distributional quantities a metric of the mean field cannot provide. ``record_rois``, a
    ``{name: ResolvedTarget}`` mapping, adds each draw's volume-weighted mean ``|E|`` per named
    ROI. The reductions run in-place into device-resident per-sample arrays, so they add no
    host sync inside the loop.

    ``recovery`` post-processes each draw's field before any statistic is taken, so the moment
    arrays and the per-draw scalars all describe the same field. It is a prebuilt operator from
    :func:`~cunibs.fem.ensure_recovery`, or ``None`` for the raw per-tetrahedron field.
    """
    _check_operator(ctx, recovery, False, "run_conductivity_uq")
    sigmas = sample_conductivities(config, pre.perturbed_tags)  # (N, P) f64
    sig_f32 = sigmas.astype(cp.float32)

    transform = compute_coil_transform(
        ctx, placement.center_mm, placement.handle_mm, placement.distance_mm
    )
    dadt_nodes = coil_dadt_at_nodes(
        coil.positions_m, coil.moments, transform, didt, ctx.nodes_mm
    )
    dadt_elm = _dadt_node_to_elm(dadt_nodes, ctx.tet_nodes)
    b_base, b_tissue = _placement_rhs(ctx, pre, dadt_elm)

    # The preconditioner stays frozen at nominal σ for every draw; a draw that misses tolerance
    # builds a matched one for itself (``pre.preconditioner_for``) and throws it away.
    pcg = pre.pcg
    precond = pre.precond

    n_tet = int(ctx.tet_nodes.shape[0])
    n_red = int(pre.idx.shape[0])
    stream = cp.cuda.get_current_stream().ptr
    # Draws are solved MAX_BLOCK at a time as columns of one affine block solve: column c's matrix
    # is A(sigma_c), applied from the per-tissue parts without ever being assembled, and every
    # part's read is shared across the block. A short last block repeats its final draw, which
    # costs a column and changes nothing, since columns are independent and the repeat is never
    # read back.
    width = MAX_BLOCK
    b_blk = cp.empty((n_red, width), dtype=cp.float64)
    x_blk = cp.empty((n_red, width), dtype=cp.float64)
    x0_blk = cp.empty((n_red, width), dtype=cp.float64)
    v_blk = cp.zeros((ctx.n_nodes, width), dtype=cp.float64)
    sum_e = cp.zeros(n_tet, dtype=cp.float64)
    sumsq_e = cp.zeros(n_tet, dtype=cp.float64)

    # Every buffer below is allocated once and rewritten in place per block, like everything else
    # in the loop, so the recovery call's arguments are loop-invariant too.
    e_bufs: list[cp.ndarray] = []
    magns: list[cp.ndarray] = []
    dadt_elms = [dadt_elm] * width
    rec = None
    if recovery is None or not recovery.on_potential:
        e_bufs = [cp.empty((n_tet, 3), dtype=cp.float32) for _ in range(width)]
        magns = [cp.empty(n_tet, dtype=cp.float32) for _ in range(width)]
    if recovery is not None:
        rec = RecoveredField.allocate(recovery, n_tet, width)
    # magn_stat is what every statistic below reads: binding it once is what keeps the moments
    # and the per-draw scalars describing the same field.
    magn_stat = magns if rec is None else rec.magnE
    # A potential-consuming operator reads the nodal potential and the exact nodal dA/dt rather
    # than the per-element field, so the reconstruct is skipped on every draw.
    dadt_nodal = [cp.ascontiguousarray(dadt_nodes)] * width
    on_potential = recovery is not None and recovery.on_potential

    # Warm-start seed: solve the nominal (ensemble-centre) problem once and reuse x_nominal as the
    # initial guess for every draw. solve_mixed measures convergence against ‖b‖ rather than the
    # initial residual, so a warm-started draw still reaches the same accuracy criterion.
    # pcg is cached on `pre` and reused across placements, so its values may hold a previous
    # placement's last fallback draw — reset the matrix to nominal before seeding (the
    # preconditioner affects only iteration count, not the solution, so it needs no reset).
    x_nominal = cp.empty(n_red, dtype=cp.float64)
    pcg.update_values(cp.ascontiguousarray(pre.nominal_data), stream)
    b_nom = cp.ascontiguousarray(
        (b_base + pre.nominal_sigma.astype(cp.float32) @ b_tissue)[pre.idx], dtype=cp.float64
    )
    pcg.solve_mixed(precond, b_nom, x_nominal, pre.tolerance, pre.max_iters, stream)

    guess = _InitialGuess(pre, b_tissue, x_nominal, stream)

    gm_idx = cp.where(ctx.tet_tags == GM_TAG)[0]
    vols_gm = ctx.vols[gm_idx].astype(cp.float64)
    bary_gm = ctx.tet_barycenters_mm[gm_idx]
    anchor_q = cp.asarray([metrics.FOCALITY_ANCHOR_PERCENTILE / 100.0], dtype=cp.float64)
    roi_names = list(record_rois)
    probes = [
        (
            cp.ascontiguousarray(record_rois[name].elem_idx.astype(cp.int64)),
            cp.ascontiguousarray(record_rois[name].weights.astype(cp.float64)),
        )
        for name in roi_names
    ]
    roi_s = cp.empty((config.n_samples, len(roi_names)), dtype=cp.float64)
    peak_s = cp.empty(config.n_samples, dtype=cp.float64)
    foc_s = cp.empty(config.n_samples, dtype=cp.float64)
    peakloc_s = cp.empty((config.n_samples, 3), dtype=cp.float64)

    # A block runs until its slowest column converges, so draws are blocked with those that will
    # take about as long. How far a draw's sigma sits from the nominal point the warm start and
    # the preconditioner were built at predicts that well enough to take the block's mean
    # iteration count from 18.6 to 16.6 on a head mesh (the per-draw mean is 14.2). A draw's own
    # result does not depend on the block it lands in, so the order changes no draw; the moments
    # are summed in it.
    log_dist = cp.asnumpy(
        cp.sqrt((cp.log(sigmas / pre.nominal_sigma[None, :]) ** 2).sum(axis=1))
    )
    schedule = [int(k) for k in np.argsort(log_dist, kind="stable")]
    for start in range(0, config.n_samples, width):
        draws = schedule[start : start + width]
        cols = draws + [draws[-1]] * (width - len(draws))
        # Each column's right-hand side and initial guess are the arithmetic a lone draw gets,
        # column by column, so neither depends on which block the draw landed in.
        for c, k in enumerate(cols):
            b_blk[:, c] = (b_base + sig_f32[k] @ b_tissue)[pre.idx]
            x0_blk[:, c] = guess.for_draw(sigmas[k], b_blk[:, c])
        _, rels, _ = pcg.solve_affine_block(
            precond,
            b_blk,
            x_blk,
            pre.affine_coefficients(sigmas[cp.asarray(cols)]),
            pre.tolerance,
            pre.max_iters,
            stream,
            x0_blk,
        )
        for c, k in enumerate(draws):
            if rels[c] <= pre.tolerance:
                continue
            # Rare extreme draw: the nominal-σ preconditioner missed tolerance, so build one
            # matched to this sample, restart from the failed iterate, and discard it.
            # pre.precond stays at nominal for the following draws.
            sample_data = cp.ascontiguousarray(pre.combine(sigmas[k]))
            pcg.update_values(sample_data, stream)
            draw_precond = pre.preconditioner_for(sample_data)
            b_red = cp.ascontiguousarray(b_blk[:, c])
            x_red = cp.ascontiguousarray(x_blk[:, c])
            retry_iters, rel = pcg.solve_mixed(
                draw_precond, b_red, x_red, pre.tolerance, pre.max_iters, stream, x_red
            )
            if rel > pre.tolerance:
                raise SolverConvergenceError(int(retry_iters), float(rel), pre.tolerance)
            x_blk[:, c] = x_red

        v_blk[pre.idx] = x_blk
        if not on_potential:
            reconstruct_e_block(v_blk, ctx.tet_nodes, ctx.g, dadt_elms, e_bufs, magns, stream)
        if rec is not None:
            apply_recovery_into(
                recovery,
                rec,
                elements=None if on_potential else e_bufs,
                potential=v_blk,
                dadt_nodes=dadt_nodal if on_potential else None,
            )

        accumulate_moments_block(magn_stat[: len(draws)], sum_e, sumsq_e, stream)
        for c, k in enumerate(draws):
            magn_k = magn_stat[c]
            magn_gm = magn_k[gm_idx]
            peak_s[k] = magn_gm.max()
            # Same percentile anchor as metrics.focality, so a per-draw focality and the
            # summary's measure the same thing. Kept on device: reading the anchor would sync.
            anchor = metrics.weighted_quantiles(magn_gm, vols_gm, anchor_q)
            foc_s[k] = vols_gm[magn_gm >= focality_frac * anchor[0]].sum()
            peakloc_s[k] = bary_gm[cp.argmax(magn_gm)]
            for j, (idx, w) in enumerate(probes):
                roi_s[k, j] = (magn_k[idx].astype(cp.float64) * w).sum()

    n = config.n_samples
    mean = sum_e / n
    var = (sumsq_e - sum_e * sum_e / n) / max(n - 1, 1)
    std = cp.sqrt(cp.clip(var, 0.0, None))
    cov = cp.where(mean > 1e-12 * float(mean.max()), std / mean, 0.0)

    return ConductivityUQResult(
        mean_magnE=mean.astype(cp.float32),
        std_magnE=std.astype(cp.float32),
        cov_magnE=cov.astype(cp.float32),
        n_samples=n,
        perturbed_tags=pre.perturbed_tags,
        sigma_samples=cp.asnumpy(sigmas),
        vols=ctx.vols,
        tet_tags=ctx.tet_tags,
        barycenters_mm=ctx.tet_barycenters_mm,
        placement=placement,
        coil_name=coil.name,
        didt=didt,
        recovery=recovery.mode if recovery is not None else "raw",
        peak_samples=cp.asnumpy(peak_s),
        focality_samples=cp.asnumpy(foc_s),
        peak_location_samples=cp.asnumpy(peakloc_s),
        roi_samples={name: cp.asnumpy(roi_s[:, j]) for j, name in enumerate(roi_names)},
    )
