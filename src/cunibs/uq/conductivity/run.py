"""The Monte Carlo conductivity-UQ sampling loop."""

from __future__ import annotations

import cupy as cp

from cunibs.coil import Coil
from cunibs.fem.assembly import conductivity_per_tet
from cunibs.fem.placement import coil_dadt_at_nodes, compute_coil_transform
from cunibs.fem.solve import SolverContext
from cunibs.simulation import Placement
from cunibs.solver import (
    accumulate_moments,
    dadt_node_to_element,
    reconstruct_e,
    rhs_assemble,
)
from cunibs.uq.conductivity.assembly import ConductivityUQPrecompute
from cunibs.uq.conductivity.config import ConductivityUQConfig, sample_conductivities
from cunibs.uq.conductivity.result import ConductivityUQResult


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
) -> ConductivityUQResult:
    """Solve one placement across ``config.n_samples`` conductivity draws; return |E| moments.

    The coil field (``dadt_elm``) and the per-tissue RHS/stiffness components are σ-independent and
    built once. Each sample re-weights the matrix and RHS by the sampled conductivities (two small
    GEMVs), then solves against a preconditioner frozen at the nominal (ensemble-centre) σ — the
    cheapest robust choice, since i.i.d. samples give a fixed central preconditioner no drift to
    chase. ``preconditioner_refresh`` only controls the rare recovery/robustness behaviour.
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

    # The double AMGx solver stays frozen at nominal σ as the mixed-solve fallback.
    solver = pre.solver
    pcg = pre.pcg
    float_precond = pre.float_preconditioner
    nominal_f32 = cp.ascontiguousarray(pre.nominal_data.astype(cp.float32))
    policy = config.preconditioner_refresh
    periodic = policy if isinstance(policy, int) and not isinstance(policy, bool) else 0

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

    for k in range(config.n_samples):
        sample_data = cp.ascontiguousarray(pre.combine(sigmas[k]))
        pcg.update_values(sample_data)
        if policy == "always" or (periodic and k > 0 and k % periodic == 0):
            float_precond.setup(pre.indptr, pre.indices, sample_data.astype(cp.float32))

        b_red[:] = (b_base + sig_f32[k] @ b_tissue)[pre.idx]
        _, rel = pcg.solve_mixed(float_precond, b_red, x_red, pre.tolerance, pre.max_iters)
        if rel > pre.tolerance:
            if policy == "never":
                raise RuntimeError(
                    f"UQ mixed solve did not converge (rel={rel:.2e}) with a frozen "
                    "preconditioner; use preconditioner_refresh='adaptive'."
                )
            # Rare extreme draw: match the preconditioner to this sample, solve, then restore
            # the nominal-frozen hierarchy for the remaining (i.i.d.) samples.
            solver.update_coefficients(sample_data)
            solver.resetup()
            solver.solve(b_red, x_red)
            solver.update_coefficients(pre.nominal_data)
            solver.resetup()
            if policy == "always" or periodic:
                float_precond.setup(pre.indptr, pre.indices, nominal_f32)

        v[pre.idx] = x_red
        reconstruct_e(v, ctx.tet_nodes, ctx.g, dadt_elm, e_buf, magn, stream)
        accumulate_moments(magn, sum_e, sumsq_e, stream)

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
    )
