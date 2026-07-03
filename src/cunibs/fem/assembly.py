"""Assemble the P1 FEM system on the GPU."""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as csp

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

GRADIENT_TILE_TETS = 1 << 20
STIFFNESS_TILE_TETS = 1 << 18


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
    nodes_m: cp.ndarray, tet_nodes: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """Per-tetrahedron P1 basis-function gradients G (M,4,3) in 1/m and volumes (M,) in m³.

    G = (T^{-1} A)^T with A = [[-1,1,0,0],[-1,0,1,0],[-1,0,0,1]] and T the edge matrix to
    barycentric coords.
    """
    n_tet = tet_nodes.shape[0]
    g = cp.empty((n_tet, 4, 3), dtype=nodes_m.dtype)
    vols = cp.empty(n_tet, dtype=nodes_m.dtype)
    a = cp.hstack([-cp.ones((3, 1)), cp.eye(3)])
    for lo in range(0, n_tet, GRADIENT_TILE_TETS):
        hi = min(lo + GRADIENT_TILE_TETS, n_tet)
        th = nodes_m[tet_nodes[lo:hi]]
        edges = th[:, 1:4] - th[:, 0, None]
        vols[lo:hi] = cp.abs(cp.linalg.det(edges)) / 6.0
        solved = cp.linalg.solve(edges, cp.broadcast_to(a, (hi - lo, 3, 4)))
        g[lo:hi] = cp.transpose(solved, (0, 2, 1))
        del th, edges, solved
    return g, vols


def assemble_stiffness(
    g: cp.ndarray,
    vols: cp.ndarray,
    cond: cp.ndarray,
    n_nodes: int,
    tet_nodes: cp.ndarray,
) -> csp.csr_matrix:
    """Assemble the symmetric P1 conductivity stiffness matrix.

    K_e[i,j] = vol_e · σ_e · (∇λ_i · ∇λ_j).

    Tiling limits temporary COO storage and the memory used to merge duplicate entries.
    AMGx requires int32 indices.
    """
    scale = vols * cond
    n_tet = tet_nodes.shape[0]
    tile = STIFFNESS_TILE_TETS
    a = csp.csr_matrix((n_nodes, n_nodes), dtype=g.dtype)
    for lo in range(0, n_tet, tile):
        gt = g[lo : lo + tile]
        tn = tet_nodes[lo : lo + tile].astype(cp.int32, copy=False)
        gij = cp.einsum("eik,ejk->eij", gt, gt)
        k_local = scale[lo : lo + tile][:, None, None] * gij
        rows = cp.broadcast_to(tn[:, :, None], k_local.shape)
        cols = cp.broadcast_to(tn[:, None, :], k_local.shape)
        block = csp.coo_matrix(
            (k_local.ravel(), (rows.ravel(), cols.ravel())), shape=(n_nodes, n_nodes)
        ).tocsr()
        a = a + block
        del gt, tn, gij, k_local, rows, cols, block
    a.sum_duplicates()
    return a


def build_node2corner(tet_nodes: cp.ndarray, n_nodes: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Build the node-to-corner CSR used by RHS assembly.

    A stable sort fixes the reduction order for each node and makes the result reproducible.
    ``idx`` stores corner IDs ``c = 4e + i``.
    """
    n_corner = int(tet_nodes.shape[0]) * 4
    corners = cp.arange(n_corner, dtype=cp.int32)
    keys = tet_nodes.ravel()
    order = cp.argsort(keys, kind="stable")
    idx = cp.ascontiguousarray(corners[order])
    ptr = cp.zeros(n_nodes + 1, dtype=cp.int32)
    ptr[1:] = cp.cumsum(cp.bincount(keys, minlength=n_nodes)).astype(cp.int32)
    return ptr, idx
