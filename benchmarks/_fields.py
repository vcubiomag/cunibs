"""How far apart two fields are, and the fits that turn a series of them into an order.

Norms are volume weighted: ``||x||_w = sqrt(sum_i w_i |x_i|^2)``, summing a vector over its
components first. Head meshes are graded, so an unweighted norm would measure agreement over the
mesher's size field rather than over the head.

Two fields differ in shape and in scale. ``rdm`` (relative difference measure, 0 to 2) compares
shape with the scale divided out, and ``mag`` is the ratio of norms (Meijs et al. 1989).
"""

from __future__ import annotations

import numpy as np


def tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = nodes[tets]
    return np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6.0


def wnorm(x: np.ndarray, w: np.ndarray | None = None) -> float:
    sq = np.asarray(x, dtype=np.float64) ** 2
    if sq.ndim > 1:
        sq = sq.sum(axis=1)
    return float(np.sqrt(sq @ w if w is not None else sq.sum()))


def compare(test: np.ndarray, ref: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    """``rdm``, ``mag``, ``rel_l2`` and ``max_abs_over_peak`` of ``test`` against ``ref``."""
    test = np.asarray(test, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    nt, nr = wnorm(test, w), wnorm(ref, w)
    diff = test - ref
    mag_diff = np.linalg.norm(diff, axis=1) if diff.ndim > 1 else np.abs(diff)
    mag_ref = np.linalg.norm(ref, axis=1) if ref.ndim > 1 else np.abs(ref)
    return {
        "rdm": wnorm(test / nt - ref / nr, w),
        "mag": nt / nr,
        "rel_l2": wnorm(diff, w) / nr,
        "max_abs_over_peak": float(mag_diff.max() / mag_ref.max()),
    }


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q, weights=w, method="inverted_cdf"))


def roi_mean(x: np.ndarray, elements: np.ndarray, w: np.ndarray) -> float:
    return float(x[elements] @ w[elements] / w[elements].sum())


def fit_power(x, y) -> dict[str, float]:
    """``y = a * x**slope`` by least squares in log-log; ``slope`` is the order or exponent."""
    line = fit_line(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)))
    return {"slope": line["slope"], "a": float(np.exp(line["intercept"])), "r2": line["r2"]}


def fit_line(x, y) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def spearman(a, b) -> float:
    def ranks(v):
        _, inverse, counts = np.unique(
            np.asarray(v, float), return_inverse=True, return_counts=True
        )
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def quantiles(x) -> dict[str, float]:
    q = np.percentile(np.asarray(x, float), [0, 25, 50, 75, 100])
    return dict(zip(("min", "q1", "median", "q3", "max"), map(float, q), strict=True))
