"""Configuration and conductivity sampling for Monte Carlo conductivity UQ."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

import cupy as cp

from cunibs.fem.assembly import TISSUE_CONDUCTIVITY
from cunibs.metrics import region_tag

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cunibs.mesh import TissueLabel

type Distribution = Literal["lognormal", "normal"]

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


def _as_tag(key: int | TissueLabel, what: str) -> int:
    """A tissue named by its volume tag or its label, as the tag."""
    tag = region_tag(key) if isinstance(key, str) else int(key)
    if tag not in TISSUE_CONDUCTIVITY:
        raise ValueError(
            f"{what} names tissue {key!r}, which has no built-in conductivity; use one of the "
            f"tags {sorted(TISSUE_CONDUCTIVITY)} or their tissue labels."
        )
    return tag


@dataclass(frozen=True, init=False)
class ConductivityUQConfig:
    """Monte Carlo conductivity-uncertainty settings.

    Each perturbed tissue's conductivity is an independent random variable with the given
    coefficient of variation. The lognormal model keeps σ strictly positive and is parameterised
    so its mean equals the nominal ``TISSUE_CONDUCTIVITY`` value.

    Tissues are named by volume tag or by label (``"gray_matter"``), and are stored as tags. The
    perturbed set is ``perturbed_tags`` when given, otherwise the tissues ``tissue_cov`` names,
    otherwise every tissue in the mesh. A perturbed tissue ``tissue_cov`` does not name takes its
    CoV from ``DEFAULT_TISSUE_COV``.

    The AMG preconditioner is built once at the nominal (ensemble-centre) σ and reused for the
    whole run. The samples are i.i.d. about that centre, and the converged field is the same for
    any preconditioner (only the iteration count changes). A draw that fails to reach tolerance
    against it retries against a preconditioner rebuilt for that draw, so accuracy never depends
    on the preconditioner tracking the sample.
    """

    n_samples: int = 500
    seed: int = 0
    tissue_cov: dict[int, float] = field(default_factory=dict)
    perturbed_tags: tuple[int, ...] | None = None
    distribution: Distribution = "lognormal"

    def __init__(
        self,
        n_samples: int = 500,
        seed: int = 0,
        tissue_cov: Mapping[int | TissueLabel, float] | None = None,
        perturbed_tags: Iterable[int | TissueLabel] | None = None,
        distribution: Distribution = "lognormal",
    ) -> None:
        if isinstance(n_samples, bool) or int(n_samples) != n_samples or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
        if distribution not in get_args(Distribution.__value__):
            raise ValueError(
                f"distribution must be one of {get_args(Distribution.__value__)}, got "
                f"{distribution!r}."
            )
        covs: dict[int, float] = {}
        for key, value in (tissue_cov or {}).items():
            tag = _as_tag(key, "tissue_cov")
            if tag in covs:
                raise ValueError(f"tissue_cov names tissue {tag} twice.")
            cov = float(value)
            if not (math.isfinite(cov) and cov >= 0.0):
                raise ValueError(f"tissue_cov[{key!r}] must be finite and >= 0, got {value!r}.")
            covs[tag] = cov
        tags: tuple[int, ...] | None = None
        if perturbed_tags is not None:
            tags = tuple(_as_tag(t, "perturbed_tags") for t in perturbed_tags)
            if not tags:
                raise ValueError("perturbed_tags must name at least one tissue.")
            if len(set(tags)) != len(tags):
                raise ValueError(f"perturbed_tags names a tissue twice: {tags}.")
            # A CoV for a tissue that is held fixed would be silently ignored.
            unused = sorted(set(covs) - set(tags))
            if unused:
                raise ValueError(
                    f"tissue_cov sets a CoV for tissues {unused} that perturbed_tags {tags} holds "
                    "fixed. Add them to perturbed_tags or drop them from tissue_cov."
                )
        object.__setattr__(self, "n_samples", int(n_samples))
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "tissue_cov", covs)
        object.__setattr__(self, "perturbed_tags", tags)
        object.__setattr__(self, "distribution", distribution)

    def cov_for(self, tag: int) -> float:
        """Resolve the CoV for a tissue: explicit override, then the default table."""
        if tag in self.tissue_cov:
            return self.tissue_cov[tag]
        return DEFAULT_TISSUE_COV[tag]

    def tags_for(self, mesh_tags: Iterable[int]) -> tuple[int, ...]:
        """The tissues this configuration perturbs on a mesh holding ``mesh_tags``, sorted.

        Raises when a named tissue is not in the mesh: its stiffness part would be empty, and
        its draws would change nothing while still being reported.
        """
        present = {int(t) for t in mesh_tags}
        if self.perturbed_tags is not None:
            tags = set(self.perturbed_tags)
        elif self.tissue_cov:
            tags = set(self.tissue_cov)
        else:
            tags = present
        missing = sorted(tags - present)
        if missing:
            raise ValueError(
                f"tissues {missing} are named for perturbation but the mesh holds none of their "
                f"elements (it holds {sorted(present)})."
            )
        return tuple(sorted(tags))


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
