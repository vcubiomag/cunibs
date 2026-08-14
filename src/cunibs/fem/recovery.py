"""Patch-recovery post-processing for the per-tetrahedron E-field.

The raw FEM field is constant over each tetrahedron. Recovery fits a local patch around every
node and interpolates from it, which is what SimNIBS's surface- and volume-interpolation path
does before reporting a field.

For node ``n`` the SPR modes least-squares fit ``p(x) = [1, x, y, z]`` over the barycentres of
the incident tetrahedra and evaluate it at the node::

    A_n  = sum_{e in n} p(b_e) p(b_e)^T          (4x4 SPD)
    E*_n = p(x_n)^T A_n^-1 sum_{e in n} p(b_e) E_e

That is linear in ``E_e``, so it collapses to one *scalar* weight per (node, element)
incidence::

    w_c  = p(x_n)^T A_n^-1 p(b_e)     for corner c = 4e + i with tet_nodes[e, i] = n
    E*_n = sum_{c in n} w_c E_{c>>2}

The same weight serves all three components and none of it depends on the placement, so the
whole operator is one float per corner laid over a corner CSR, built once per mesh and reused.
Nodes on the boundary of the volume the patch is drawn from take a volume-weighted average
instead, which is also a scalar per corner, so one array covers both rules. That volume is the
whole mesh for ``"spr_global"`` and one tissue for ``"spr_tissue"``.

The fit runs in the node-centred, radius-scaled frame ``[1, (b_e - x_n) / h]``. That is an
affine change of basis, so the recovered value is unchanged in exact arithmetic, but it leaves
the normal matrix far better conditioned on head-mesh coordinates.

Harmonic recovery
-----------------

Two properties of this PDE make recovering ``v`` better than recovering ``E`` directly.

``dA/dt`` is analytic and perfectly smooth, so every discontinuity in ``E`` lives in
``grad(v)``; recovering the potential and adding the exact nodal ``dA/dt`` afterwards keeps the
smooth part exact. And inside a region of constant sigma, ``v`` is *harmonic*: the weak form
gives ``div(sigma grad v) = -div(sigma dA/dt)``, and a magnetic dipole's ``A`` is
divergence-free away from its source, so ``lap(v) = 0``. Restricting the fit to the harmonic
polynomials, 9 of the 10 quadratics, 16 of the 20 cubics, since those of degree exactly ``l``
span ``2l + 1`` dimensions, therefore encodes the solution rather than regularising it.

That constraint is also what makes the fit well posed where the plain one is not. At an
interface or outer-boundary node the same-tissue patch is a half-ball, which cannot determine
the curvature normal to the flat side; Laplace's equation ties that unknown to the in-plane
curvature the patch does resolve.

The assumption is specific to the magneto-quasistatic TMS formulation, where the source sits
outside the head. It would not carry to a problem with interior current sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, get_args

import cupy as cp

from cunibs.fem.assembly import build_node2corner
from cunibs.solver import (
    count_incident_node_csr,
    count_patch_csr,
    fill_incident_node_csr,
    fill_patch_csr,
    hpr_grad,
    hpr_weights,
    mark_outer_boundary,
    recover_elements,
    recover_nodes,
    spr_weights,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cunibs.fem.solve import SolverContext

type Recovery = Literal["raw", "spr_global", "spr_tissue", "harmonic"]
"""How the reported E-field is derived from the FEM solution.

``"raw"``
    The per-tetrahedron field straight from the solve, constant over each element. Its peak is
    the largest element *average* rather than a value at a point, so it overstates the pointwise
    maximum and moves with mesh resolution.
``"spr_global"``
    Zienkiewicz-Zhu superconvergent patch recovery over patches spanning every incident
    tetrahedron regardless of tissue, smearing the genuine E discontinuity at a conductivity
    jump. Corresponds to SimNIBS's ``interpolate_scattered(..., continuous=True)``.
``"spr_tissue"``
    The same fit with each patch restricted to one tissue, so the discontinuity survives. ``E``
    is genuinely two-valued at a conductivity jump -- the tangential component is continuous
    while the normal one jumps by the conductivity ratio -- and a patch spanning the jump fits a
    discontinuous function with a linear polynomial. This is what SimNIBS's surface and volume
    overlays report, down to the boundary rule: a tissue-interface node lies on the cropped
    tissue's boundary, so each of its slots takes that tissue's own volume-weighted average
    rather than a fit, and what survives the jump is two one-sided averages.
``"harmonic"``
    Recovers the *potential* over a same-tissue node patch and differentiates it, in the
    16-dimensional space of harmonic cubics, dropping to the 9-dimensional harmonic quadratics on
    a patch too small to carry them. The default. See the module docstring.
"""

RECOVERY_MODES: tuple[Recovery, ...] = get_args(Recovery.__value__)

#: Modes whose slots are (node, tissue) pairs rather than plain nodes.
TISSUE_SLOT_MODES: tuple[Recovery, ...] = ("spr_tissue", "harmonic")

#: Modes that consume the nodal potential rather than the per-element field.
POTENTIAL_MODES: tuple[Recovery, ...] = ("harmonic",)


def validate_recovery(recovery: str) -> Recovery:
    """Normalise and check a recovery mode."""
    if recovery not in RECOVERY_MODES:
        raise ValueError(
            f"Unknown recovery {recovery!r}; use one of {', '.join(map(repr, RECOVERY_MODES))}."
        )
    return recovery


def outer_boundary_nodes(
    tet_nodes: cp.ndarray,
    n_nodes: int,
    node2corner: tuple[cp.ndarray, cp.ndarray] | None = None,
) -> cp.ndarray:
    """Mark nodes lying on the outer boundary of the tetrahedral volume.

    A boundary face is one that only a single tetrahedron owns, and every owner of a face holds
    each of its three nodes, so the owners are counted by walking one node's segment of the
    node-to-corner incidence CSR. ``node2corner`` is that CSR, ``build_node2corner``'s result;
    pass it when the caller already holds one, otherwise it is built here.

    ``mesh.skin_tris`` is not a substitute: it carries surface tag 1005 alone, while the
    volume's boundary also includes whatever the loader's volume-tag filtering exposed.
    """
    is_boundary = cp.zeros(n_nodes, dtype=cp.int32)
    if int(tet_nodes.shape[0]) == 0:
        return is_boundary

    ptr, idx = node2corner if node2corner is not None else build_node2corner(tet_nodes, n_nodes)
    mark_outer_boundary(tet_nodes, ptr, idx, is_boundary, cp.cuda.get_current_stream().ptr)
    return is_boundary


def tissue_boundary_nodes(
    tet_nodes: cp.ndarray,
    n_nodes: int,
    slot_node: cp.ndarray,
    node2corner: tuple[cp.ndarray, cp.ndarray] | None = None,
) -> cp.ndarray:
    """Mark nodes lying on the boundary of *their own tissue's* submesh.

    This is the set SimNIBS's tissue path sees: it crops the mesh to one tag and runs
    ``elm_data2node_data`` there, where ``get_outside_faces`` returns the cropped volume's
    boundary, so a tissue-interface node takes the volume-weighted average just as an
    outer-surface node does.

    One node mask serves every tag, and no per-tag face count is needed, because a node is on the
    tag-``T`` submesh's boundary exactly when it is on the whole volume's outer boundary or it
    touches more than one tag, whichever ``T`` is. Forwards: a face with a single tag-``T`` owner
    either has one owner in total, or a second owner carrying another tag. Backwards: were a mixed
    node's tag-``T`` faces all doubly owned, its tag-``T`` tetrahedra would close into a
    neighbourhood ball around it and leave a conforming mesh nowhere to put the other tag's.

    ``slot_node`` is :func:`tissue_slots`' sorted map from slot to node, so a node carrying more
    than one tag repeats.
    """
    is_boundary = outer_boundary_nodes(tet_nodes, n_nodes, node2corner=node2corner)
    if int(slot_node.shape[0]) < 2:
        return is_boundary

    repeated = slot_node[:-1][slot_node[1:] == slot_node[:-1]]
    is_boundary[repeated] = 1
    return is_boundary


class TissueSlots(NamedTuple):
    """A corner CSR grouped by (node, tissue), plus the maps back to node and tag."""

    ptr: cp.ndarray
    idx: cp.ndarray
    slot_of_corner: cp.ndarray
    slot_node: cp.ndarray
    slot_tag: cp.ndarray


def tissue_slots(tet_nodes: cp.ndarray, tet_tags: cp.ndarray) -> TissueSlots:
    """Group corners by (node, tissue) instead of by node alone.

    The sort key is ``node * K + tag``, a refinement of the plain node order, so this is the
    CSR ``build_node2corner`` produces with each node's segment further split by tissue. The
    sort is stable for the same reason that one is: it fixes which corner contributes when, and
    so keeps the weights and every field built from them reproducible run to run.
    """
    n_tet = int(tet_nodes.shape[0])
    if n_tet == 0:
        empty = cp.empty(0, dtype=cp.int32)
        return TissueSlots(cp.zeros(1, dtype=cp.int32), empty, empty, empty, empty)

    corner_node = tet_nodes.reshape(-1).astype(cp.int64)
    corner_tag = cp.repeat(tet_tags.astype(cp.int64), 4)
    stride = cp.int64(int(tet_tags.max()) + 1)
    key = corner_node * stride + corner_tag

    order = cp.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = cp.flatnonzero(
        cp.concatenate([cp.ones(1, dtype=bool), sorted_key[1:] != sorted_key[:-1]])
    )
    n_slots = int(starts.shape[0])

    slot_of_sorted = cp.zeros(sorted_key.shape[0], dtype=cp.int32)
    slot_of_sorted[starts] = 1
    slot_of_sorted = cp.cumsum(slot_of_sorted, dtype=cp.int32) - 1

    ptr = cp.zeros(n_slots + 1, dtype=cp.int32)
    ptr[:-1] = starts.astype(cp.int32)
    ptr[-1] = cp.int32(sorted_key.shape[0])
    idx = cp.ascontiguousarray(cp.arange(4 * n_tet, dtype=cp.int32)[order])

    slot_of_corner = cp.empty(4 * n_tet, dtype=cp.int32)
    slot_of_corner[order] = slot_of_sorted
    return TissueSlots(
        ptr=ptr,
        idx=idx,
        slot_of_corner=slot_of_corner,
        slot_node=cp.ascontiguousarray((sorted_key[starts] // stride).astype(cp.int32)),
        slot_tag=cp.ascontiguousarray((sorted_key[starts] % stride).astype(cp.int32)),
    )


# Escalate to the 2-ring below this many patch nodes. A fit over a patch barely larger than its
# basis interpolates rather than averages, so a determined but too-flat patch passes the pivot test
# and still returns weights of order 1e10. 16 gives the quadratic rung's 9 terms that margin, and
# leaves the great majority of a head mesh's slots on the cubic rung, which wants 20 nodes for its
# 16 terms.
_MIN_PATCH_NODES = 16


def _first_ring(
    tet_nodes: cp.ndarray, ptr: cp.ndarray, idx: cp.ndarray, n_slots: int
) -> tuple[cp.ndarray, cp.ndarray]:
    """The distinct nodes of each slot's own tetrahedra, as a CSR over slots."""
    stream = cp.cuda.get_current_stream().ptr
    r1_ptr = cp.empty(n_slots + 1, dtype=cp.int32)
    nnz = count_incident_node_csr(tet_nodes, ptr, idx, r1_ptr, stream)
    r1_idx = cp.empty(nnz, dtype=cp.int32)
    fill_incident_node_csr(tet_nodes, ptr, idx, r1_ptr, r1_idx, stream)
    return r1_ptr, r1_idx


def _neighbour_slots(
    r1_ptr: cp.ndarray,
    r1_idx: cp.ndarray,
    slot_key: cp.ndarray,
    slot_tag: cp.ndarray,
    stride: int,
) -> cp.ndarray:
    """For each first-ring entry, the slot of the same tissue centred on that node.

    Every such slot exists: a first-ring node was reached through a tetrahedron of the slot's own
    tissue, so it carries that tissue itself.
    """
    counts = cp.diff(r1_ptr)
    owner = cp.repeat(cp.arange(int(counts.shape[0]), dtype=cp.int32), counts)
    key = r1_idx.astype(cp.int64) * stride + slot_tag[owner].astype(cp.int64)
    return cp.searchsorted(slot_key, key).astype(cp.int32)


def node_patches(ctx: SolverContext, slots: TissueSlots) -> tuple[cp.ndarray, cp.ndarray]:
    """Build the per-slot node patch the potential is fitted over.

    Starts from the nodes of the slot's own tetrahedra and grows to the next ring wherever that
    holds fewer than ``_MIN_PATCH_NODES``. Growing means taking the union of the neighbours' own
    first rings, which reaches the same nodes as re-walking the same-tissue tetrahedra around the
    slot without touching the mesh a second time.

    Returns ``(pptr, pidx)``. The patch is emitted sorted by node within each slot, so the
    reduction order every weight and field is built on is fixed by construction.
    """
    n_slots = int(slots.slot_node.shape[0])
    if n_slots == 0:
        return cp.zeros(1, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

    r1_ptr, r1_idx = _first_ring(ctx.tet_nodes, slots.ptr, slots.idx, n_slots)
    stride = int(slots.slot_tag.max()) + 1
    slot_key = slots.slot_node.astype(cp.int64) * stride + slots.slot_tag.astype(cp.int64)
    neighbour = _neighbour_slots(r1_ptr, r1_idx, slot_key, slots.slot_tag, stride)

    stream = cp.cuda.get_current_stream().ptr
    pptr = cp.empty(n_slots + 1, dtype=cp.int32)
    nnz = count_patch_csr(r1_ptr, r1_idx, neighbour, _MIN_PATCH_NODES, pptr, stream)
    pidx = cp.empty(nnz, dtype=cp.int32)
    fill_patch_csr(r1_ptr, r1_idx, neighbour, _MIN_PATCH_NODES, pptr, pidx, stream)
    return pptr, pidx


@dataclass(frozen=True)
class RecoveryOperator:
    """The precomputed patch weights for one mesh and one recovery mode.

    ``ptr``/``idx`` is a CSR over slots, where a slot is whatever a patch is fitted around: a
    node for ``"spr_global"``, a (node, tissue) pair for the tissue-restricted modes. What it
    indexes differs by mode, and so does the shape of ``w``:

    * SPR modes index *corners* (``c = 4e + i``, the map the RHS assembly already builds) and
      ``w`` is one scalar per corner.
    * ``"harmonic"`` indexes patch *nodes* and ``w`` is ``(nnz, 3)``, since it differentiates.

    ``slot_of_corner`` always says which slot each corner reads back from when the recovered
    field is sampled at the barycentres.

    ``n_fallback`` counts slots that did not take the mode's own fit: for the SPR modes the
    volume-weighted average, for ``"harmonic"`` the linear rung or nothing at all. A slot lands
    there either because it sits on its patch volume's boundary or because the patch could not
    support the fit. For the SPR modes the boundary term dominates and is a rule rather than a
    failure, so subtract ``is_boundary[slot_node].sum()`` to isolate the rejections;
    ``"harmonic"`` has no boundary rule, so every one of its fallbacks is a rejection.

    A harmonic rejection means the patch could not carry either harmonic rung even after growing,
    which on a head mesh is a tissue fragment too small to fit anything to rather than a bad patch
    around a good one; the patch is typically the whole same-tissue connected component, so the
    linear rung is the only thing available there.

    ``n_undetermined`` counts slots where no gradient was determined at all, leaving ``-dA/dt``
    there; that takes an all-coplanar patch, so anything but zero is worth investigating.

    ``status`` is the harmonic mode's per-slot rung and is ``None`` elsewhere: 0 harmonic cubic,
    1 harmonic quadratic, 2 linear, 3 undetermined. It is the array the two counts above are
    reductions of, and what makes a rejected fit attributable to a place on the mesh rather than
    only countable.
    """

    mode: Recovery
    ptr: cp.ndarray
    idx: cp.ndarray
    w: cp.ndarray
    slot_of_corner: cp.ndarray
    slot_node: cp.ndarray
    n_slots: int
    n_fallback: int
    slot_tag: cp.ndarray | None = None
    n_undetermined: int = 0
    status: cp.ndarray | None = None

    @property
    def on_potential(self) -> bool:
        """Whether the operator consumes the nodal potential rather than the per-element field."""
        return self.mode in POTENTIAL_MODES


def _build_harmonic_operator(ctx: SolverContext) -> RecoveryOperator:
    """Fit the potential over same-tissue node patches, in the harmonic-polynomial space."""
    slots = tissue_slots(ctx.tet_nodes, ctx.tet_tags)
    pptr, pidx = node_patches(ctx, slots)
    n_slots = int(slots.slot_node.shape[0])

    w = cp.empty((int(pidx.shape[0]), 3), dtype=cp.float32)
    status = cp.empty(n_slots, dtype=cp.int32)
    # In metres, not millimetres. These weights differentiate, so unlike the SPR ones they carry
    # a length unit: v is in volts and E has to come out in V/m. The scaling cancels in any check
    # whose reference gradient is also per millimetre, so it stays invisible until dA/dt is added.
    nodes_m = cp.ascontiguousarray(ctx.nodes_mm * 1e-3)
    hpr_weights(
        nodes_m, pptr, pidx, slots.slot_node, w, status, cp.cuda.get_current_stream().ptr
    )
    del nodes_m

    return RecoveryOperator(
        mode="harmonic",
        ptr=pptr,
        idx=pidx,
        w=w,
        slot_of_corner=slots.slot_of_corner,
        slot_node=slots.slot_node,
        slot_tag=slots.slot_tag,
        n_slots=n_slots,
        # Dropping from the cubic rung to the quadratic one is still a harmonic fit, so it is not
        # a fallback; only leaving the harmonic ladder is.
        n_fallback=int((status >= 2).sum()),
        n_undetermined=int((status == 3).sum()),
        status=status,
    )


def _build_spr_operator(ctx: SolverContext, mode: Recovery) -> RecoveryOperator:
    """Fit the per-element field over each slot's incident tetrahedra, linearly."""
    n_nodes = int(ctx.n_nodes)
    if mode in TISSUE_SLOT_MODES:
        slots = tissue_slots(ctx.tet_nodes, ctx.tet_tags)
        ptr, idx, slot_of_corner = slots.ptr, slots.idx, slots.slot_of_corner
        slot_node, slot_tag = slots.slot_node, slots.slot_tag
        slot_arg = slot_node
    else:
        # A slot is a node, so the corner CSR is the one the RHS assembly already carries and a
        # corner reads back from its own node. No new index arrays at all, and the kernel reads
        # the slot index directly rather than through an identity map.
        ptr, idx = ctx.node2corner_ptr, ctx.node2corner_idx
        slot_of_corner = cp.ascontiguousarray(ctx.tet_nodes.reshape(-1))
        slot_node = cp.arange(n_nodes, dtype=cp.int32)
        slot_tag = None
        slot_arg = None

    node2corner = (ctx.node2corner_ptr, ctx.node2corner_idx)
    is_boundary = (
        tissue_boundary_nodes(ctx.tet_nodes, n_nodes, slot_node, node2corner=node2corner)
        if mode in TISSUE_SLOT_MODES
        else outer_boundary_nodes(ctx.tet_nodes, n_nodes, node2corner=node2corner)
    )
    w = cp.empty(int(idx.shape[0]), dtype=cp.float32)
    counter = cp.zeros(1, dtype=cp.int32)
    spr_weights(
        ctx.nodes_mm,
        ctx.tet_nodes,
        ctx.vols,
        ptr,
        idx,
        slot_arg,
        is_boundary,
        w,
        counter,
        cp.cuda.get_current_stream().ptr,
    )
    return RecoveryOperator(
        mode=mode,
        ptr=ptr,
        idx=idx,
        w=w,
        slot_of_corner=slot_of_corner,
        slot_node=slot_node,
        slot_tag=slot_tag,
        n_slots=int(slot_node.shape[0]),
        n_fallback=int(counter[0]),
    )


def build_recovery_operator(ctx: SolverContext, mode: Recovery) -> RecoveryOperator:
    """Build the patch weights for ``mode``. Costs one pass over the mesh, once per subject."""
    mode = validate_recovery(mode)
    if mode == "raw":
        raise ValueError("The raw mode has no recovery operator to build.")
    if mode in POTENTIAL_MODES:
        return _build_harmonic_operator(ctx)
    return _build_spr_operator(ctx, mode)


def ensure_recovery(ctx: SolverContext, mode: Recovery) -> RecoveryOperator:
    """Return ``mode``'s operator for ``ctx``, building and caching it on first use.

    Callers that hold a :class:`~cunibs.simulation.Subject` should go through it instead, so the
    build is serialised by its lock and lands outside any scratch allocator.
    """
    mode = validate_recovery(mode)
    cached = ctx.recovery.get(mode)
    if cached is None:
        cached = build_recovery_operator(ctx, mode)
        ctx.recovery[mode] = cached
    return cached


class RecoveredField(NamedTuple):
    """One recovered field per placement, on slots and sampled back at the barycentres.

    The names match what these arrays become downstream -- ``PlacementResult["E_slots"]`` and
    :attr:`~cunibs.FieldResult.E_slots`, and likewise for ``E`` and ``magnE`` -- so nothing has
    to be renamed as a field flows out of the solver.
    """

    E_slots: list[cp.ndarray]
    E: list[cp.ndarray]
    magnE: list[cp.ndarray]

    @classmethod
    def allocate(cls, op: RecoveryOperator, n_tet: int, k: int = 1) -> RecoveredField:
        """Buffers for ``k`` placements against ``op``."""
        return cls(
            E_slots=[cp.empty((op.n_slots, 3), dtype=cp.float32) for _ in range(k)],
            E=[cp.empty((n_tet, 3), dtype=cp.float32) for _ in range(k)],
            magnE=[cp.empty(n_tet, dtype=cp.float32) for _ in range(k)],
        )


def apply_recovery_into(
    op: RecoveryOperator,
    out: RecoveredField,
    *,
    elements: Sequence[cp.ndarray] | None = None,
    potential: cp.ndarray | None = None,
    dadt_nodes: Sequence[cp.ndarray] | None = None,
) -> None:
    """Recover a block of placements into caller-allocated buffers.

    The operator decides which inputs it reads, so a caller never branches on the mode. An
    element-consuming operator needs ``elements``, the per-tetrahedron field; a
    potential-consuming one needs ``potential``, the row-major ``(n_nodes, stride)`` float64
    block the solve already produces, together with ``dadt_nodes``, the exact nodal dA/dt each
    placement was built from. Passing all three is fine.

    Both kernels read the shared operator once for the whole block, and each column's reduction
    order is the one the width-1 path would use, so a placement's recovered field does not
    depend on who it was batched with.

    The buffer-taking form exists for the UQ loop, which reuses one set of arrays across every
    draw rather than allocating per sample.
    """
    stream = cp.cuda.get_current_stream().ptr
    if op.on_potential:
        if potential is None or dadt_nodes is None:
            missing = "potential" if potential is None else "dadt_nodes"
            raise ValueError(
                f"recovery={op.mode!r} reads the nodal potential, so apply_recovery needs "
                f"{missing}=. Pass the solve's v block and the nodal dA/dt it was built from."
            )
        _check_width(len(dadt_nodes), out, op.mode)
        hpr_grad(
            potential, op.w, op.ptr, op.idx, op.slot_node, list(dadt_nodes), out.E_slots, stream
        )
    else:
        if elements is None:
            raise ValueError(
                f"recovery={op.mode!r} reads the per-element field, so apply_recovery needs "
                "elements=."
            )
        _check_width(len(elements), out, op.mode)
        recover_nodes(list(elements), op.w, op.ptr, op.idx, out.E_slots, stream)
    recover_elements(out.E_slots, op.slot_of_corner, out.E, out.magnE, stream)


def _check_width(k: int, out: RecoveredField, mode: Recovery) -> None:
    if k != len(out.E_slots):
        raise ValueError(
            f"recovery={mode!r} was given {k} placements but buffers for {len(out.E_slots)}. "
            "Allocate with RecoveredField.allocate(op, n_tet, k)."
        )


def apply_recovery(
    op: RecoveryOperator,
    n_tet: int,
    *,
    elements: Sequence[cp.ndarray] | None = None,
    potential: cp.ndarray | None = None,
    dadt_nodes: Sequence[cp.ndarray] | None = None,
) -> RecoveredField:
    """Recover a block of placements, allocating the outputs.

    The allocating form of :func:`apply_recovery_into`; see it for which inputs an operator
    reads.
    """
    k = len(dadt_nodes or ()) if op.on_potential else len(elements or ())
    out = RecoveredField.allocate(op, n_tet, k)
    apply_recovery_into(op, out, elements=elements, potential=potential, dadt_nodes=dadt_nodes)
    return out
