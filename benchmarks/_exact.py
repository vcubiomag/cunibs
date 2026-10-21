"""Problems with known solutions, and the meshes they are solved on.

The induced field of magnetic dipoles outside a spherically symmetric conductor is known in
closed form (Heller and van Hulsteyn 1992). It does not depend on the radial conductivity
profile and has no radial component. ``ball_mesh`` builds the layered ball it is scored on.

Two manufactured interface problems have an exact field on both sides of a conductivity jump:
a box split by the plane ``z = Z0``, with ``v = f(z) cos(kx)``, and a spherical shell split at
``R_MID``, with ``v`` an ``l = 4`` solid harmonic. In both, ``v`` is harmonic in each region and
``v`` and ``sigma dv/dn`` are continuous across the interface, so ``E`` jumps by the
conductivity ratio. The source is the Dirichlet data on the outer boundary and dA/dt is zero.
``solve`` computes the P1 solution of either and the exact field at every (node, tissue) slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cunibs.fem import TISSUE_CONDUCTIVITY
from cunibs.mesh import SKIN_SURFACE_TAG, HeadMesh

MU0_OVER_4PI = 1e-7

BALL_RADII_MM = (55.0, 70.0, 85.0)
BALL_TAGS = (2, 7, 5)

INNER, OUTER = 2, 3
BOX_MM, Z0, K = 100.0, 50.0, 0.02
R_IN, R_MID, R_OUT, DEGREE = 30.0, 50.0, 70.0, 4
PLANAR_CELLS = (10, 14, 20, 28, 40)
SPHERE_RUNGS = ((1, 4), (2, 8), (3, 16), (4, 32), (5, 64))
JITTER = 0.25


def induced_e(points_mm, dipole_pos_mm, dipole_moment, didt: float) -> np.ndarray:
    """E in V/m at ``points_mm`` inside a sphere centred on the origin, from dipoles outside it."""
    r1 = np.asarray(points_mm, np.float64) * 1e-3
    e = np.zeros_like(r1)
    for r2, m in zip(np.asarray(dipole_pos_mm) * 1e-3, np.asarray(dipole_moment), strict=True):
        a = r2 - r1
        na = np.linalg.norm(a, axis=1)[:, None]
        nr2 = np.linalg.norm(r2)
        r2a = np.sum(r2 * a, axis=1)[:, None]
        f = na * (nr2 * na + r2a)
        grad_f = (na**2 / nr2 + 2 * na + 2 * nr2 + r2a / na) * r2 - (
            na + 2 * nr2 + r2a / na
        ) * r1
        m_grad_f = np.sum(m * grad_f, axis=1)[:, None]
        e += -didt * MU0_OVER_4PI / f**2 * (f * np.cross(r1, m) - np.cross(m_grad_f * r1, r2))
    return e


def radial_ratio(points_mm: np.ndarray, e: np.ndarray) -> float:
    """``max |E . r_hat| / max |E|``, which is zero for the exact field."""
    norms = np.linalg.norm(points_mm, axis=1)[:, None]
    radial = np.abs(np.sum(e * points_mm / norms, axis=1))
    return float(radial.max() / np.linalg.norm(e, axis=1).max())


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit-sphere vertices and triangles of a refined icosahedron."""
    p = (1 + 5**0.5) / 2
    verts = [np.array(v, float) for v in (
        (-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p),
        (0, -1, -p), (0, 1, -p), (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1),
    )]  # fmt: skip
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
        (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
        (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    for _ in range(subdivisions):
        middle: dict[tuple[int, int], int] = {}
        refined = []
        for a, b, c in faces:
            ab, bc, ca = (_midpoint(verts, middle, u, w) for u, w in ((a, b), (b, c), (c, a)))
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    return np.array(verts), np.array(faces, np.int64)


def _midpoint(verts: list, middle: dict, a: int, b: int) -> int:
    key = (min(a, b), max(a, b))
    if key not in middle:
        v = verts[a] + verts[b]
        verts.append(v / np.linalg.norm(v))
        middle[key] = len(verts) - 1
    return middle[key]


def _prisms(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Three tetrahedra per triangular prism. Rows sorted by node index keep shared faces conforming."""
    (a0, b0, c0), (a1, b1, c1) = lo.T, hi.T
    return np.concatenate(
        [np.stack(t, -1) for t in ((a0, b0, c0, c1), (a0, b0, c1, b1), (a0, b1, c1, a1))]
    )


def _jitter(points: np.ndarray, movable: np.ndarray, step: float, rng, jitter: float) -> None:
    if jitter:
        points[movable] += rng.uniform(-jitter, jitter, (int(movable.sum()), 3)) * step


def ball_mesh(subdivisions: int, tags=BALL_TAGS, jitter: float = JITTER) -> HeadMesh:
    """A solid ball of concentric shells, with conforming interfaces at ``BALL_RADII_MM``.

    Each shell gets as many radial layers as keep its elements near isotropic. Nodes between
    interfaces are jittered. ``tags=(2, 2, 2)`` gives the same nodes and elements, homogeneous.
    """
    sphere, tris = icosphere(subdivisions)
    n_v = len(sphere)
    edge = float(np.linalg.norm(sphere[tris[:, 0]] - sphere[tris[:, 1]], axis=1).mean())
    tris = np.sort(tris, axis=1)
    rng = np.random.default_rng(0)
    nodes, tets, tet_tags = [np.zeros((1, 3))], [], []
    inner, start = 0.0, 1
    for shell, (outer, tag) in enumerate(zip(BALL_RADII_MM, tags, strict=True)):
        n_layers = max(1, round((outer - inner) / (edge * outer)))
        radii = np.linspace(inner, outer, n_layers + 1)[1:]
        rings = (radii[:, None, None] * sphere).reshape(-1, 3)
        step = min(radii[0] - inner, edge * max(inner, radii[0]))
        _jitter(rings, np.arange(len(rings)) < (n_layers - 1) * n_v, step, rng, jitter)
        nodes.append(rings)
        block = (
            [np.column_stack([np.zeros(len(tris), np.int64), tris + start])]
            if shell == 0
            else []
        )
        first = 1 if shell == 0 else 0
        block += [
            _prisms(start + (k - 1) * n_v + tris, start + k * n_v + tris)
            for k in range(first, n_layers)
        ]
        tets += block
        tet_tags += [np.full(sum(len(b) for b in block), tag)]
        inner, start = outer, start + n_layers * n_v
    surface = (tris + start - n_v).astype(np.int32)
    return HeadMesh(
        np.concatenate(nodes),
        np.concatenate(tets).astype(np.int32),
        np.concatenate(tet_tags).astype(np.int32),
        surface,
        np.full(len(surface), SKIN_SURFACE_TAG, np.int32),
    )


def mean_edge_mm(nodes: np.ndarray, tets: np.ndarray) -> float:
    p = nodes[tets]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return float(np.mean([np.linalg.norm(p[:, i] - p[:, j], axis=1).mean() for i, j in pairs]))


def _mesh(nodes: np.ndarray, tets: np.ndarray, tags: np.ndarray) -> HeadMesh:
    # The manufactured problems place no coil, so they need no scalp surface.
    empty = np.zeros((0, 3), np.int32)
    return HeadMesh(nodes, tets.astype(np.int32), tags.astype(np.int32), empty, empty[:, 0])


@dataclass(frozen=True)
class Planar:
    """The box ``[0, BOX_MM]^3``, gray matter below ``Z0`` and CSF above."""

    cells: int
    jitter: float = JITTER

    @property
    def label(self) -> str:
        return f"planar_c{self.cells}"

    @property
    def h_mm(self) -> float:
        return BOX_MM / self.cells

    def mesh(self) -> HeadMesh:
        n = self.cells + 1
        lin = np.linspace(0.0, BOX_MM, n)
        nodes = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
        inside = ((nodes > 0) & (nodes < BOX_MM)).all(axis=1) & ~np.isclose(nodes[:, 2], Z0)
        _jitter(nodes, inside, self.h_mm, np.random.default_rng(0), self.jitter)
        corner = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])
        cube = [
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ]
        ijk = np.stack(np.meshgrid(*(np.arange(self.cells),) * 3, indexing="ij"), -1)
        ids = ijk.reshape(-1, 1, 3) + corner
        verts = (ids[..., 0] * n + ids[..., 1]) * n + ids[..., 2]
        tets = verts[:, cube].reshape(-1, 4)
        return _mesh(nodes, tets, np.where(nodes[tets].mean(1)[:, 2] < Z0, INNER, OUTER))

    def _f(self, z: np.ndarray, tag: int) -> tuple[np.ndarray, np.ndarray]:
        if tag == INNER:
            return np.exp(K * z), K * np.exp(K * z)
        ratio = TISSUE_CONDUCTIVITY[INNER] / TISSUE_CONDUCTIVITY[OUTER]
        a, b = (1 + ratio) / 2, np.exp(2 * K * Z0) * (1 - ratio) / 2
        up, down = a * np.exp(K * z), b * np.exp(-K * z)
        return up + down, K * (up - down)

    def field(self, points: np.ndarray, tag: int) -> np.ndarray:
        """E in V/m, from the branch of region ``tag`` even on the interface."""
        f, fz = self._f(points[:, 2], tag)
        x = points[:, 0]
        grad = np.stack([-K * f * np.sin(K * x), np.zeros_like(x), fz * np.cos(K * x)], -1)
        return -grad * 1e3

    def potential(self, nodes: np.ndarray) -> np.ndarray:
        below = self._f(nodes[:, 2], INNER)[0]
        return np.where(nodes[:, 2] < Z0, below, self._f(nodes[:, 2], OUTER)[0]) * np.cos(
            K * nodes[:, 0]
        )

    def on_interface(self, points: np.ndarray) -> np.ndarray:
        return np.isclose(points[:, 2], Z0)


@dataclass(frozen=True)
class Shell:
    """The shell ``R_IN < r < R_OUT``, gray matter inside ``R_MID`` and CSF outside."""

    subdivisions: int
    layers: int
    jitter: float = JITTER

    @property
    def label(self) -> str:
        return f"sphere_s{self.subdivisions}_l{self.layers}"

    @property
    def h_mm(self) -> float:
        return (R_OUT - R_IN) / self.layers

    def mesh(self) -> HeadMesh:
        sphere, tris = icosphere(self.subdivisions)
        n_v = len(sphere)
        radii = np.linspace(R_IN, R_OUT, self.layers + 1)
        nodes = (radii[:, None, None] * sphere).reshape(-1, 3)
        held = np.isin(np.arange(self.layers + 1), (0, self.layers // 2, self.layers))
        edge = np.linalg.norm(sphere[tris[:, 0]] - sphere[tris[:, 1]], axis=1).mean()
        step = min(radii[1] - radii[0], edge * R_IN)
        _jitter(nodes, np.repeat(~held, n_v), step, np.random.default_rng(0), self.jitter)
        lo = np.arange(self.layers)[:, None, None] * n_v + np.sort(tris, axis=1)
        tets = np.concatenate([_prisms(a, a + n_v) for a in lo])
        radius = np.linalg.norm(nodes[tets].mean(1), axis=1)
        return _mesh(nodes, tets, np.where(radius < R_MID, INNER, OUTER))

    def _harmonic(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``r^4 P_4(cos theta)`` and its gradient."""
        x, y, z = p.T
        r2 = (p * p).sum(1)
        val = (35 * z**4 - 30 * z * z * r2 + 3 * r2 * r2) / 8
        common = (12 * r2 - 60 * z * z) / 8
        gz = (80 * z**3 - 48 * z * r2) / 8
        return val, np.stack([common * x, common * y, gz], -1)

    def _solution(self, p: np.ndarray, tag: int) -> tuple[np.ndarray, np.ndarray]:
        val, grad = self._harmonic(p)
        if tag == INNER:
            return val, grad
        sin, sout = TISSUE_CONDUCTIVITY[INNER], TISSUE_CONDUCTIVITY[OUTER]
        b = DEGREE * (sout - sin) / ((2 * DEGREE + 1) * sout)
        r = np.linalg.norm(p, axis=1)
        f = b * (R_MID / r) ** (2 * DEGREE + 1)
        dfdr = -(2 * DEGREE + 1) * f / r
        scale = 1 - b + f
        return scale * val, scale[:, None] * grad + (dfdr * val / r)[:, None] * p

    def field(self, points: np.ndarray, tag: int) -> np.ndarray:
        return -self._solution(points, tag)[1] * 1e3

    def potential(self, nodes: np.ndarray) -> np.ndarray:
        inside = np.linalg.norm(nodes, axis=1) < R_MID
        return np.where(
            inside, self._solution(nodes, INNER)[0], self._solution(nodes, OUTER)[0]
        )

    def on_interface(self, points: np.ndarray) -> np.ndarray:
        return np.isclose(np.linalg.norm(points, axis=1), R_MID)


def ladder(geometry: str, quick: bool, jitter: float = JITTER) -> list[Planar | Shell]:
    """A refinement ladder, coarsest first."""
    n = 4 if quick else None
    if geometry == "planar":
        return [Planar(c, jitter) for c in PLANAR_CELLS[:n]]
    return [Shell(s, layers, jitter) for s, layers in SPHERE_RUNGS[:n]]


def sigma_of(tags: np.ndarray) -> np.ndarray:
    lut = np.zeros(max(TISSUE_CONDUCTIVITY) + 1)
    lut[list(TISSUE_CONDUCTIVITY)] = list(TISSUE_CONDUCTIVITY.values())
    return lut[tags]


def p1_gradients(nodes_mm: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape-function gradients ``(n_tet, 4, 3)`` in 1/m and volumes in m^3, in float64."""
    p = nodes_mm[tets] * 1e-3
    edges = p[:, 1:] - p[:, :1]
    g = np.empty((len(tets), 4, 3))
    g[:, 1:] = np.linalg.inv(edges).transpose(0, 2, 1)
    g[:, 0] = -g[:, 1:].sum(axis=1)
    return g, np.abs(np.linalg.det(edges)) / 6


def p1_dirichlet(nodes_mm, tets, tags, fixed, values) -> tuple[np.ndarray, float]:
    """The P1 potential with ``values`` held at the ``fixed`` nodes, and its relative residual.

    The solve runs to 1e-13 so that the algebraic error stays far below the discretisation
    error a convergence order measures.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csl

    n = len(nodes_mm)
    g, vol = p1_gradients(nodes_mm, tets)
    sigma = sigma_of(tags)
    local = (vol * sigma)[:, None, None] * np.einsum("eik,ejk->eij", g, g)
    rows, cols = np.repeat(tets, 4, axis=1).ravel(), np.tile(tets, (1, 4)).ravel()
    # Duplicates are summed on the host in a fixed order. CuPy's COO-to-CSR conversion sums them
    # in an order that changes between runs, and the solution then changes in its last bits, which
    # gives the SimNIBS request built from it a new name on every run.
    order = np.lexsort((cols, rows))
    rows, cols, values_ = rows[order], cols[order], local.ravel()[order]
    first = np.flatnonzero(np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])])
    a = csp.csr_matrix(
        (
            cp.asarray(np.add.reduceat(values_, first)),
            cp.asarray(cols[first], cp.int32),
            cp.asarray(np.searchsorted(rows[first], np.arange(n + 1)), cp.int32),
        ),
        shape=(n, n),
    )
    del rows, cols, local, values_
    free = cp.asarray(~fixed, cp.float64)
    held = cp.asarray(np.where(fixed, values, 0.0))
    b = -free * (a @ held)
    inv_diag = cp.where(free > 0, 1.0 / a.diagonal(), 1.0)
    op = csl.LinearOperator((n, n), lambda x: free * (a @ (free * x)), dtype=cp.float64)
    precond = csl.LinearOperator((n, n), lambda x: inv_diag * x, dtype=cp.float64)
    x, _ = csl.cg(op, b, rtol=1e-13, atol=0.0, maxiter=50_000, M=precond)
    residual = float(cp.linalg.norm(op.matvec(x) - b) / cp.linalg.norm(b))
    return np.where(fixed, values, cp.asnumpy(x)), residual


@dataclass
class Solved:
    """One manufactured problem solved on one mesh, in the context's element order.

    Slots are (node, tissue) pairs, the set the tissue-restricted rules recover on. ``truth``
    is the exact field at each slot from its own tissue's side, so an interface node carries
    both limits.
    """

    problem: Planar | Shell
    ctx: object
    nodes: np.ndarray
    tets: np.ndarray
    tags: np.ndarray
    on_surface: np.ndarray
    v: np.ndarray
    e_raw: np.ndarray
    solve_residual: float
    slot_node: np.ndarray = field(init=False)
    slot_tag: np.ndarray = field(init=False)
    truth: np.ndarray = field(init=False)
    at_interface: np.ndarray = field(init=False)
    interior: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        import cupy as cp

        from cunibs.fem import ensure_recovery

        op = ensure_recovery(self.ctx, "spr_tissue")
        self.slot_node = cp.asnumpy(op.slot_node).astype(np.int64)
        self.slot_tag = cp.asnumpy(op.slot_tag).astype(np.int64)
        self.truth = np.zeros((len(self.slot_node), 3))
        for tag in (INNER, OUTER):
            sel = self.slot_tag == tag
            self.truth[sel] = self.problem.field(self.nodes[self.slot_node[sel]], tag)
        off = ~self.on_surface[self.slot_node]
        self.at_interface = off & self.problem.on_interface(self.nodes[self.slot_node])
        self.interior = off & ~self.at_interface

    def recover(self, mode: str) -> np.ndarray:
        """``mode``'s field on the tissue slots; a node-valued rule repeats its value per tissue."""
        import cupy as cp

        from cunibs.fem import apply_recovery, ensure_recovery

        op = ensure_recovery(self.ctx, mode)
        if op.on_potential:
            zero = [cp.zeros((len(self.nodes), 3), cp.float32)]
            v = cp.ascontiguousarray(cp.asarray(self.v).reshape(-1, 1))
            out = apply_recovery(op, len(self.tets), potential=v, dadt_nodes=zero)
        else:
            out = apply_recovery(
                op, len(self.tets), elements=[cp.asarray(self.e_raw, cp.float32)]
            )
        slots = cp.asnumpy(out.E_slots[0]).astype(np.float64)
        return slots[self.slot_node] if op.slot_tag is None else slots


def solve(problem: Planar | Shell) -> Solved:
    import cupy as cp

    from cunibs.fem import build_context, outer_boundary_nodes

    ctx = build_context(problem.mesh())
    nodes, tets, tags = (
        ctx.mesh.nodes_mm,
        ctx.mesh.tet_nodes.astype(np.int64),
        ctx.mesh.tet_tags,
    )
    on_surface = cp.asnumpy(outer_boundary_nodes(ctx.tet_nodes, len(nodes))).astype(bool)
    v, residual = p1_dirichlet(nodes, tets, tags, on_surface, problem.potential(nodes))
    e_raw = -np.einsum("ei,eik->ek", v[tets], p1_gradients(nodes, tets)[0])
    return Solved(problem, ctx, nodes, tets, tags, on_surface, v, e_raw, residual)


def rel_rms(values: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """RMS error over the masked rows, relative to the RMS of the truth there; NaN if none."""
    if not mask.any():
        return float("nan")
    d, t = values[mask] - truth[mask], truth[mask]
    return float(np.sqrt((d * d).sum(1).mean() / (t * t).sum(1).mean()))


def _slope(h: np.ndarray, e: np.ndarray) -> dict:
    lx, ly = np.log(h), np.log(e)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    sxx = ((lx - lx.mean()) ** 2).sum()
    se = float(np.sqrt(resid @ resid / ((len(h) - 2) * sxx))) if len(h) > 2 else None
    return {"order": float(slope), "order_se": se}


def order(h, errors) -> dict:
    """Slope of log error against log h with its standard error, the successive orders between
    neighbouring rungs, and (from four rungs) the slope over the three finest.

    Rungs come coarsest first. A rung with no error (an empty region) is left out.
    """
    h, e = np.asarray(h, float), np.asarray(errors, float)
    keep = np.isfinite(e) & (e > 0)
    h, e = h[keep], e[keep]
    if len(h) < 2:
        return {"order": None, "order_se": None, "local": []}
    out = _slope(h, e) | {"local": [float(v) for v in np.diff(np.log(e)) / np.diff(np.log(h))]}
    if len(h) >= 4:
        out["tail"] = _slope(h[-3:], e[-3:])
    return out
