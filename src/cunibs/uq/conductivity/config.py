"""Configuration and conductivity sampling for Monte Carlo conductivity UQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cupy as cp

from cunibs.fem.assembly import TISSUE_CONDUCTIVITY

# Illustrative lognormal coefficients of variation for conductivity sensitivity analyses.
# Override via ConductivityUQConfig.tissue_cov for a study-specific uncertainty model.
DEFAULT_TISSUE_COV: dict[int, float] = {
    1: 0.15,  # white matter
    2: 0.15,  # gray matter
    3: 0.05,  # CSF
    4: 0.35,  # average bone
    5: 0.20,  # scalp
    6: 0.20,  # eye balls
    7: 0.35,  # compact bone
    8: 0.35,  # spongy bone
    9: 0.15,  # blood
    10: 0.20,  # muscle
}

RefreshPolicy = Literal["adaptive", "always", "never"]


@dataclass(frozen=True)
class ConductivityUQConfig:
    """Monte Carlo conductivity-uncertainty settings.

    Each perturbed tissue's conductivity is an independent random variable with the given
    coefficient of variation. The lognormal model keeps σ strictly positive and is parameterised
    so its mean equals the nominal ``TISSUE_CONDUCTIVITY`` value.

    The AMG preconditioner is frozen at the nominal (ensemble-centre) σ for the whole run: with
    i.i.d. samples there is no drift to chase, and the converged field is exact for any
    preconditioner (only the iteration count changes). ``preconditioner_refresh`` only tunes
    robustness/cost:
    - ``"adaptive"`` (default) — frozen; on the rare draw that fails to converge, resetup, solve,
      then restore the nominal hierarchy. Fastest robust choice.
    - ``"never"``  — frozen with no recovery (a pathological draw would raise). Marginally faster.
    - ``"always"`` — ``resetup`` every sample (most robust, ~20% slower; a diagnostic baseline).
    - an ``int`` k — ``resetup`` every k-th sample.
    """

    n_samples: int = 500
    seed: int = 0
    cov: float = 0.1
    tissue_cov: dict[int, float] = field(default_factory=dict)
    perturbed_tags: tuple[int, ...] | None = None
    distribution: Literal["lognormal", "normal"] = "lognormal"
    preconditioner_refresh: RefreshPolicy | int = "adaptive"

    def cov_for(self, tag: int) -> float:
        """Resolve the CoV for a tissue: explicit override, then default table, then global."""
        if tag in self.tissue_cov:
            return self.tissue_cov[tag]
        return DEFAULT_TISSUE_COV.get(tag, self.cov)


def sample_conductivities(
    config: ConductivityUQConfig, perturbed_tags: tuple[int, ...]
) -> cp.ndarray:
    """Draw ``(n_samples, len(perturbed_tags))`` conductivities on the GPU.

    Lognormal draws use the median parameterisation ``σ0·exp(s·z − s²/2)`` so ``E[σ] = σ0``.
    """
    rng = cp.random.default_rng(config.seed)
    nominal = cp.asarray([TISSUE_CONDUCTIVITY[t] for t in perturbed_tags], dtype=cp.float64)
    covs = cp.asarray([config.cov_for(t) for t in perturbed_tags], dtype=cp.float64)
    z = rng.standard_normal((config.n_samples, len(perturbed_tags)), dtype=cp.float64)
    if config.distribution == "lognormal":
        s = cp.sqrt(cp.log1p(covs**2))
        return nominal * cp.exp(z * s - 0.5 * s**2)
    sigmas = nominal * (1.0 + covs * z)
    # A normal model can wander non-physical; clamp to a small positive floor.
    return cp.maximum(sigmas, 1e-6 * nominal)
