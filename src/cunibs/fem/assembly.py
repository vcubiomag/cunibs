"""Assemble the P1 FEM system on the GPU."""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as csp

from cunibs.solver import (
    assemble_stiffness_values,
    count_incident_node_csr,
    fill_incident_node_csr,
    p1_gradients,
)

TISSUE_CONDUCTIVITY: dict[int, float] = {
    1: 0.126,  # white matter
    2: 0.275,  # gray matter
    3: 1.654,  # CSF
    4: 0.010,  # average bone
    5: 0.465,  # scalp
    6: 0.5,  # eye balls
    7: 0.008,  # compact bone
    8: 0.025,  # spongy bone
    9: 0.6,  # blood
    10: 0.16,  # muscle
}

GM_TAG = 2


def conductivity_per_tet(tet_tags: cp.ndarray) -> cp.ndarray:
    """Map tetrahedron tags to isotropic conductivities (S/m)."""
    # NaN entries detect missing tags after the table lookup.
    max_tag = max(TISSUE_CONDUCTIVITY)
    lut = cp.full(max_tag + 1, cp.nan, dtype=cp.float64)
    for tag, sigma in TISSUE_CONDUCTIVITY.items():
        lut[tag] = sigma
    cond = lut[cp.clip(tet_tags, 0, max_tag)]
    unknown = cp.isnan(cond) | (tet_tags < 0) | (tet_tags > max_tag)
    if bool(unknown.any()):
        raise ValueError(f"Tetrahedra with unknown tags: {cp.unique(tet_tags[unknown])}")
    return cond


def gradient_operator(
    nodes_mm: cp.ndarray, tet_nodes: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """Per-tetrahedron P1 basis-function gradients G (M,4,3) in 1/m and volumes (M,) in m³.

    Node coordinates are in millimetres; the metre conversion is folded into the kernel. See
    gradient.cu for the closed form.
    """
    n_tet = int(tet_nodes.shape[0])
    g = cp.empty((n_tet, 4, 3), dtype=cp.float64)
    vols = cp.empty(n_tet, dtype=cp.float64)
    p1_gradients(
        cp.ascontiguousarray(nodes_mm, dtype=cp.float64),
        as_int32_tets(tet_nodes),
        g,
        vols,
        cp.cuda.get_current_stream().ptr,
    )
    return g, vols


def as_int32_tets(tet_nodes: cp.ndarray) -> cp.ndarray:
    """Contiguous int32 connectivity, the layout the solver kernels index with."""
    return cp.ascontiguousarray(tet_nodes.astype(cp.int32, copy=False))


def assemble_stiffness(
    g: cp.ndarray,
    vols: cp.ndarray,
    cond: cp.ndarray,
    n_nodes: int,
    tet_nodes: cp.ndarray,
) -> csp.csr_matrix:
    """Assemble the symmetric P1 conductivity stiffness matrix.

    K_e[i,j] = vol_e · σ_e · (∇λ_i · ∇λ_j).

    Two passes: ``stiffness_pattern`` derives row and column indices from the connectivity alone,
    then ``fill_stiffness_values`` writes the values one thread per row over the node2corner map,
    which is what fixes the summation order.
    """
    tets = as_int32_tets(tet_nodes)
    ptr, idx = build_node2corner(tets, n_nodes)
    a = stiffness_pattern(tets, n_nodes, ptr, idx)
    fill_stiffness_values(a, g, vols, cond, tets, ptr, idx)
    return a


def stiffness_pattern(
    tet_nodes: cp.ndarray, n_nodes: int, ptr: cp.ndarray, idx: cp.ndarray
) -> csp.csr_matrix:
    """CSR sparsity pattern of the stiffness matrix, with uninitialised values.

    The pattern depends only on connectivity, so a caller assembling several matrices over one
    mesh builds it once and refills ``.data`` per conductivity with ``fill_stiffness_values``.
    ``ptr``/``idx`` is the mesh's node2corner CSR.

    Sizing the rows takes a pass of its own, because nothing knows how many distinct columns a row
    has until its candidates are sorted, so the column index is allocated at exactly its own
    length. The candidates themselves never leave the kernel's shared memory.
    """
    stream = cp.cuda.get_current_stream().ptr
    indptr = cp.empty(n_nodes + 1, dtype=cp.int32)
    nnz = count_incident_node_csr(tet_nodes, ptr, idx, indptr, stream)
    indices = cp.empty(nnz, dtype=cp.int32)
    fill_incident_node_csr(tet_nodes, ptr, idx, indptr, indices, stream)
    return csp.csr_matrix(
        (cp.empty(nnz, dtype=cp.float64), indices, indptr), shape=(n_nodes, n_nodes)
    )


def fill_stiffness_values(
    a: csp.csr_matrix,
    g: cp.ndarray,
    vols: cp.ndarray,
    cond: cp.ndarray,
    tet_nodes: cp.ndarray,
    ptr: cp.ndarray,
    idx: cp.ndarray,
) -> None:
    """Overwrite ``a.data`` in place with the values for conductivity ``cond``.

    ``a`` keeps whatever pattern it already holds; ``ptr``/``idx`` are the mesh's node2corner CSR.
    """
    assemble_stiffness_values(
        cp.ascontiguousarray(g),
        cp.ascontiguousarray(vols * cond),
        tet_nodes,
        ptr,
        idx,
        a.indptr,
        a.indices,
        a.data,
        cp.cuda.get_current_stream().ptr,
    )


def build_node2corner(tet_nodes: cp.ndarray, n_nodes: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Build the node-to-corner CSR used by RHS assembly.

    A stable sort fixes the reduction order for each node and makes the result reproducible.
    ``idx`` stores corner IDs ``c = k*e + i``, where ``k`` is the element's corner count: four for
    the tetrahedra this is usually called on, three for the skin triangles the smoothed normals
    reduce over.
    """
    n_corner = int(tet_nodes.shape[0]) * int(tet_nodes.shape[1])
    ptr = cp.zeros(n_nodes + 1, dtype=cp.int32)
    if n_corner == 0:
        # cupy's bincount reduces over the keys to size its output rather than trusting
        # minlength, so it cannot take an empty array. A tet subset is empty whenever a UQ
        # per-tissue component covers a tag the mesh does not use.
        return ptr, cp.empty(0, dtype=cp.int32)
    corners = cp.arange(n_corner, dtype=cp.int32)
    keys = tet_nodes.ravel()
    order = cp.argsort(keys, kind="stable")
    idx = cp.ascontiguousarray(corners[order])
    ptr[1:] = cp.cumsum(cp.bincount(keys, minlength=n_nodes)).astype(cp.int32)
    return ptr, idx
