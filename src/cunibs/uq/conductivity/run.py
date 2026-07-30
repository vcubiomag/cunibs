"""The Monte Carlo conductivity-UQ sampling loop."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as csp

from cunibs import metrics
from cunibs.fem.assembly import GM_TAG, conductivity_per_tet
from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform
from cunibs.fem.solve import SolverConvergenceError
from cunibs.solver import (
    accumulate_moments,
    dadt_node_to_element,
    reconstruct_e,
    rhs_assemble,
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


def run_conductivity_uq(
    ctx: SolverContext,
    pre: ConductivityUQPrecompute,
    coil: Coil,
    placement: Placement,
    config: ConductivityUQConfig,
    didt: float = 1e6,
    record_rois: Mapping[str, ResolvedTarget] = _NO_ROIS,
    focality_frac: float = 0.5,
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
    """
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
    b_red = cp.empty(n_red, dtype=cp.float64)
    x_red = cp.empty(n_red, dtype=cp.float64)
    v = cp.zeros(ctx.n_nodes, dtype=cp.float64)
    e_buf = cp.empty((n_tet, 3), dtype=cp.float32)
    magn = cp.empty(n_tet, dtype=cp.float32)
    sum_e = cp.zeros(n_tet, dtype=cp.float64)
    sumsq_e = cp.zeros(n_tet, dtype=cp.float64)

    # Warm-start seed: solve the nominal (ensemble-centre) problem once and reuse x_nominal as the
    # initial guess for every draw. solve_mixed measures convergence against ‖b‖ rather than the
    # initial residual, so a warm-started draw still reaches the same accuracy criterion.
    # pcg is cached on `pre` and reused across placements, so its values may hold a previous
    # placement's last draw — reset the matrix to nominal before seeding (the preconditioner
    # affects only iteration count, not the solution, so it needs no reset).
    x_nominal = cp.empty(n_red, dtype=cp.float64)
    pcg.update_values(cp.ascontiguousarray(pre.nominal_data), stream)
    b_nom = cp.ascontiguousarray(
        (b_base + pre.nominal_sigma.astype(cp.float32) @ b_tissue)[pre.idx], dtype=cp.float64
    )
    pcg.solve_mixed(precond, b_nom, x_nominal, pre.tolerance, pre.max_iters, stream)

    # Subspace recycling: only P conductivities are perturbed, so x(σ)−x_nominal spans an ≈P-dim
    # subspace. Build an A-orthonormal basis W from the first RECYCLE_BUILD draws, then Galerkin-
    # project each later draw's initial guess onto it: x0 = x_nominal + W·E⁻¹·Wᵀr0, with
    # r0 = b(σ)−A(σ)x_nominal and E = Wᵀ·A_nominal·W (frozen). Only a better x0 into the same solve.
    RECYCLE_BUILD = 16
    recycle = config.n_samples >= 2 * RECYCLE_BUILD
    if recycle:
        n_build = RECYCLE_BUILD
        d_basis = cp.empty((n_red, n_build), dtype=cp.float64)
        x0_buf = cp.empty(n_red, dtype=cp.float64)
        recycle_w = None  # (n_red, k)
        einv_nom = None  # (k, k)
        # r0 = b − A(σ)x_nominal is linear in σ: precompute A_base·x_nom and each A_t·x_nom once.
        a_base = csp.csr_matrix((pre.base_data, pre.indices, pre.indptr), shape=(n_red, n_red))
        ax_base = a_base @ x_nominal
        ax_tissue = cp.empty((len(pre.perturbed_tags), n_red), dtype=cp.float64)
        for t in range(len(pre.perturbed_tags)):
            a_t = csp.csr_matrix(
                (pre.tissue_data[t], pre.indices, pre.indptr), shape=(n_red, n_red)
            )
            ax_tissue[t] = a_t @ x_nominal

    gm_idx = cp.where(ctx.tet_tags == GM_TAG)[0]
    vols_gm = ctx.vols[gm_idx].astype(cp.float64)
    bary_gm = cp.asarray(ctx.mesh.tet_barycenters_mm)[gm_idx]
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

    for k in range(config.n_samples):
        sample_data = cp.ascontiguousarray(pre.combine(sigmas[k]))
        pcg.update_values(sample_data, stream)

        b_red[:] = (b_base + sig_f32[k] @ b_tissue)[pre.idx]
        x0 = x_nominal
        if recycle and recycle_w is not None:
            r0 = b_red - (ax_base + sigmas[k] @ ax_tissue)
            x0_buf[:] = x_nominal + recycle_w @ (einv_nom @ (recycle_w.T @ r0))
            x0 = x0_buf
        _, rel = pcg.solve_mixed(
            precond, b_red, x_red, pre.tolerance, pre.max_iters, stream, x0
        )
        if rel > pre.tolerance:
            # Rare extreme draw: the nominal-σ preconditioner missed tolerance, so build one
            # matched to this sample, restart from the failed iterate, and discard it. pre.precond
            # stays at nominal for the following draws.
            draw_precond = pre.preconditioner_for(sample_data)
            retry_iters, rel = pcg.solve_mixed(
                draw_precond, b_red, x_red, pre.tolerance, pre.max_iters, stream, x_red
            )
            if rel > pre.tolerance:
                raise SolverConvergenceError(int(retry_iters), float(rel), pre.tolerance)

        if recycle and recycle_w is None:
            d_basis[:, k] = x_red - x_nominal
            if k == n_build - 1:
                # A-orthonormal basis for the perturbation subspace; drop near-dependent directions.
                q, s, _ = cp.linalg.svd(d_basis, full_matrices=False)
                keep = int((s > 1e-8 * float(s[0])).sum())
                recycle_w = cp.ascontiguousarray(q[:, :keep])
                a_nom = csp.csr_matrix(
                    (pre.nominal_data, pre.indices, pre.indptr), shape=(n_red, n_red)
                )
                einv_nom = cp.linalg.inv(recycle_w.T @ (a_nom @ recycle_w))

        v[pre.idx] = x_red
        reconstruct_e(v, ctx.tet_nodes, ctx.g, dadt_elm, e_buf, magn, stream)
        accumulate_moments(magn, sum_e, sumsq_e, stream)

        magn_gm = magn[gm_idx]
        peak_s[k] = magn_gm.max()
        # Same percentile anchor as metrics.focality, so a per-draw focality and the summary's
        # measure the same thing. Kept on device: reading the anchor would sync the loop.
        anchor = metrics.weighted_quantiles(magn_gm, vols_gm, anchor_q)
        foc_s[k] = vols_gm[magn_gm >= focality_frac * anchor[0]].sum()
        peakloc_s[k] = bary_gm[cp.argmax(magn_gm)]
        for j, (idx, w) in enumerate(probes):
            roi_s[k, j] = (magn[idx].astype(cp.float64) * w).sum()

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
        barycenters_mm=cp.asarray(ctx.mesh.tet_barycenters_mm),
        placement=placement,
        coil_name=coil.name,
        didt=didt,
        peak_samples=cp.asnumpy(peak_s),
        focality_samples=cp.asnumpy(foc_s),
        peak_location_samples=cp.asnumpy(peakloc_s),
        roi_samples={name: cp.asnumpy(roi_s[:, j]) for j, name in enumerate(roi_names)},
    )
