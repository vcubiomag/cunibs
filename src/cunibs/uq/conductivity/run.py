"""The Monte Carlo conductivity-UQ sampling loop."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import cupy as cp
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

# Tolerance for the sensitivity solves that span the warm-start basis. Deliberately far looser
# than the draw tolerance: the basis only has to point in the right directions for an initial
# guess that CG then refines to `pre.tolerance` anyway, so its accuracy sets the rate and never
# the answer.
_SENSITIVITY_TOL = 1e-3

# Second-order sensitivities kept in the warm-start basis: two block solves. On sub-004 the
# strongest 16 of 45 take a block from 15.5 iterations (first order only) to 12.6; all 45 reach
# 12.5 for three times the setup.
_SECOND_ORDER_PAIRS = 16

# Relative eigenvalue below which a basis direction counts as dependent on the others and is
# dropped, in the nominal energy norm.
_BASIS_RTOL = 1e-12


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


def _cholesky_solve(e: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``e[d] y[d] = rhs[d]`` for every draw ``d``, ``e`` (N, m, m) SPD.

    Written as elementwise operations along the draw axis, with every sum accumulated in a fixed
    order: a batched LAPACK or BLAS call may pick its summation order by size or alignment, and a
    draw's guess must not depend on how many draws it was formed with.
    """
    n, m = rhs.shape
    lo = np.zeros((n, m, m))
    for j in range(m):
        acc = e[:, j, j].copy()
        for q in range(j):
            acc = acc - lo[:, j, q] * lo[:, j, q]
        lo[:, j, j] = np.sqrt(acc)
        for i in range(j + 1, m):
            acc = e[:, i, j].copy()
            for q in range(j):
                acc = acc - lo[:, i, q] * lo[:, j, q]
            lo[:, i, j] = acc / lo[:, j, j]
    z = np.zeros((n, m))
    for i in range(m):
        acc = rhs[:, i].copy()
        for q in range(i):
            acc = acc - lo[:, i, q] * z[:, q]
        z[:, i] = acc / lo[:, i, i]
    y = np.zeros((n, m))
    for i in reversed(range(m)):
        acc = z[:, i].copy()
        for q in range(i + 1, m):
            acc = acc - lo[:, q, i] * y[:, q]
        y[:, i] = acc / lo[:, i, i]
    return y


class _ReducedBasis:
    """The x0 the sampling loop starts every draw from: a reduced-basis Galerkin solve.

    Both A(sigma) and b(sigma) are affine in sigma, so the solution x(sigma) is a smooth function of
    P conductivities and lies close to a low-dimensional space. The basis is the ensemble-centre
    solution, its first derivatives and the strongest second derivatives:

        A_nom dx/ds_t = b_t - A_t x_nom,
        A_nom d2x/ds_s ds_t = -(A_s dx/ds_t + A_t dx/ds_s),

    and a draw's guess is the Galerkin solution in that span with the draw's own operator,
    ``x0 = V y``, ``(V^T A(sigma) V) y = V^T b(sigma)``, which is the A(sigma)-best point of the span.
    Both reduced quantities are affine in sigma too, so ``V^T A_t V`` and ``V^T b_t`` are formed once
    and a draw costs an m-by-m solve. The second derivatives kept are those with the largest
    ``std(s_s) std(s_t) |rhs|``, weighted by how often the pair appears in the Taylor series.

    Every per-placement quantity is a function of the mesh, the placement, the nominal
    conductivities and the configured spreads, and a draw's guess reads nothing else but its own
    sigma. Keep it that way. x0 cannot move a draw outside ``pre.tolerance``, but it does decide
    where inside it the draw lands, so anything reaching x0 from the ensemble (a basis grown from
    earlier draws, a size-dependent batched solve) makes a draw's last digits depend on the batch.
    """

    def __init__(
        self,
        pre: ConductivityUQPrecompute,
        b_base: cp.ndarray,
        b_tissue: cp.ndarray,
        x_nominal: cp.ndarray,
        spreads: np.ndarray,
        stream: int,
    ) -> None:
        idx = pre.idx
        p = len(pre.tissue_ops)
        ax = cp.stack([op @ x_nominal for op in pre.tissue_ops], axis=1)  # (n_red, P)
        rhs1 = cp.ascontiguousarray(b_tissue[:, idx].astype(cp.float64).T - ax)
        sens = _solve_columns(pre, rhs1, _SENSITIVITY_TOL, stream)
        a_sens = [op @ sens for op in pre.tissue_ops]

        pairs = [(s, t) for s in range(p) for t in range(s, p)]
        std = cp.asnumpy(pre.nominal_sigma) * spreads
        score = np.array(
            [
                (1.0 if s != t else 0.5)
                * std[s]
                * std[t]
                * float(cp.linalg.norm(a_sens[s][:, t] + a_sens[t][:, s]))
                for s, t in pairs
            ]
        )
        keep = np.sort(np.argsort(-score, kind="stable")[:_SECOND_ORDER_PAIRS])
        rhs2 = cp.ascontiguousarray(
            cp.stack(
                [
                    -(a_sens[pairs[j][0]][:, pairs[j][1]] + a_sens[pairs[j][1]][:, pairs[j][0]])
                    for j in keep
                ],
                axis=1,
            )
        )
        sens2 = _solve_columns(pre, rhs2, _SENSITIVITY_TOL, stream)
        v = cp.ascontiguousarray(cp.concatenate([x_nominal[:, None], sens, sens2], axis=1))
        del rhs1, rhs2

        # V^T A_t V per tissue. A_t x_nom and A_t dx/ds are already to hand from the right-hand
        # sides; only the second-order columns need a product of their own.
        e_tissue = []
        for t, op in enumerate(pre.tissue_ops):
            av = cp.concatenate([ax[:, t : t + 1], a_sens[t], op @ sens2], axis=1)
            e_tissue.append(cp.asnumpy(v.T @ av))
        e_terms = np.stack(e_tissue)
        if pre.base_op is not None:
            e_terms = np.concatenate([cp.asnumpy(v.T @ (pre.base_op @ v))[None], e_terms])
        e_terms = 0.5 * (e_terms + e_terms.transpose(0, 2, 1))
        b_cols = cp.concatenate([b_base[idx][:, None], b_tissue[:, idx].T], axis=1)
        b_terms = cp.asnumpy(v.T @ b_cols.astype(cp.float64)).T  # (1 + P, m)

        # An A_nom-orthonormal basis, once per placement: a draw's reduced matrix is then the
        # identity plus the draw's departure from nominal, so a plain Cholesky solves it, and
        # directions the others already span are dropped here rather than per draw.
        nominal = np.concatenate([[1.0], cp.asnumpy(pre.nominal_sigma)])
        coef_nom = nominal if pre.base_op is not None else nominal[1:]
        # The columns differ in scale by orders of magnitude, so they are equilibrated first and
        # the cutoff only sees dependence, not size.
        g = np.tensordot(coef_nom, e_terms, axes=1)
        d = 1.0 / np.sqrt(np.diag(g))
        w, u = np.linalg.eigh(g * d[:, None] * d[None, :])
        live = w > _BASIS_RTOL * w[-1]
        tr = d[:, None] * u[:, live] / np.sqrt(w[live])
        self._e_terms = np.einsum("ji,tjk,kl->til", tr, e_terms, tr)
        self._b_terms = b_terms @ tr
        self._v = cp.ascontiguousarray(v @ cp.asarray(tr))
        self._has_base = pre.base_op is not None

    @property
    def width(self) -> int:
        return int(self._v.shape[1])

    def coefficients(self, sigmas: np.ndarray) -> np.ndarray:
        """Each draw's reduced coordinates ``y`` (N, m), its own arithmetic whatever N is."""
        n = sigmas.shape[0]
        e = np.zeros((n, self.width, self.width))
        rhs = np.zeros((n, self.width))
        first = 1 if self._has_base else 0
        if self._has_base:
            e = e + self._e_terms[0][None]
        rhs = rhs + self._b_terms[0][None]
        for t in range(sigmas.shape[1]):
            e = e + sigmas[:, t, None, None] * self._e_terms[first + t][None]
            rhs = rhs + sigmas[:, t, None] * self._b_terms[1 + t][None]
        return _cholesky_solve(e, rhs)

    def guess_into(self, y: cp.ndarray, out: cp.ndarray, stream: int) -> None:
        """``out[:, c] = V y[c]`` for the k rows of ``y``."""
        row_dot(self._v, y, out, stream)


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

    spreads = np.array([config.cov_for(t) for t in pre.perturbed_tags], dtype=np.float64)
    basis = _ReducedBasis(pre, b_base, b_tissue, x_nominal, spreads, stream)
    y_all = cp.asarray(basis.coefficients(cp.asnumpy(sigmas)))

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
    # the preconditioner were built at predicts that. With every tissue perturbed, A and b both
    # scale with sigma, so x(c sigma) = x(sigma), and CG against a fixed preconditioner takes the
    # same iterates for a scaled operator: only the ratios matter, and the common scale is
    # projected out of the distance. A draw's own result does not depend on the block it lands
    # in, so the order changes no draw; the moments are summed in it.
    log_ratio = cp.log(sigmas / pre.nominal_sigma[None, :])
    if pre.base_op is None:
        log_ratio = log_ratio - log_ratio.mean(axis=1, keepdims=True)
    log_dist = cp.asnumpy(cp.sqrt((log_ratio**2).sum(axis=1)))
    schedule = [int(k) for k in np.argsort(log_dist, kind="stable")]
    for start in range(0, config.n_samples, width):
        draws = schedule[start : start + width]
        cols = draws + [draws[-1]] * (width - len(draws))
        # Each column's right-hand side and initial guess are the arithmetic a lone draw gets,
        # column by column, so neither depends on which block the draw landed in.
        for c, k in enumerate(cols):
            b_blk[:, c] = (b_base + sig_f32[k] @ b_tissue)[pre.idx]
        basis.guess_into(cp.ascontiguousarray(y_all[cp.asarray(cols)]), x0_blk, stream)
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
