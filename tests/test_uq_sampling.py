"""Statistics of ``sample_conductivities`` — the conductivity draws feeding every UQ run.

The lognormal branch uses the median parameterisation σ₀·exp(s·z − s²/2) with
s = √log(1+cov²), chosen so E[σ] = σ₀ *and* std/mean = cov exactly. Both identities are
checked here, since a plausible-looking reparameterisation would silently bias every
ensemble mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from cunibs.fem.assembly import TISSUE_CONDUCTIVITY
from cunibs.uq.conductivity.config import (
    DEFAULT_TISSUE_COV,
    ConductivityUQConfig,
    sample_conductivities,
)

pytestmark = pytest.mark.gpu

BIG = 200_000


def draw(cp, tags, n=BIG, seed=0, **kw):
    cfg = ConductivityUQConfig(n_samples=n, seed=seed, **kw)
    return cp.asnumpy(sample_conductivities(cfg, tags))


def test_shape_and_dtype(cp):
    s = draw(cp, (1, 2, 3), n=17, tissue_cov={1: 0.1, 2: 0.2, 3: 0.3})
    assert s.shape == (17, 3)
    assert s.dtype == np.float64


def test_lognormal_mean_is_nominal(cp):
    """E[σ] = σ₀ and std/mean = cov — the two properties the parameterisation exists for."""
    cov = 0.3
    s = draw(cp, (2,), tissue_cov={2: cov})[:, 0]
    nominal = TISSUE_CONDUCTIVITY[2]
    assert np.all(s > 0)
    assert s.mean() / nominal == pytest.approx(1.0, rel=1e-2)
    assert s.std() / s.mean() == pytest.approx(cov, rel=3e-2)


def test_lognormal_is_lognormal(cp):
    """log σ must be Gaussian with std √log(1+cov²) and zero skew."""
    cov = 0.4
    logs = np.log(draw(cp, (2,), tissue_cov={2: cov})[:, 0])
    centred = logs - logs.mean()
    skew = float((centred**3).mean() / centred.std() ** 3)
    assert abs(skew) < 0.03
    assert logs.std() == pytest.approx(np.sqrt(np.log1p(cov**2)), rel=2e-2)
    assert logs.mean() == pytest.approx(
        np.log(TISSUE_CONDUCTIVITY[2]) - 0.5 * np.log1p(cov**2), abs=5e-3
    )


def test_zero_cov_is_degenerate(cp):
    """Zero variance must be exactly nominal, not merely close — UQ tests rely on this."""
    s = draw(cp, (2, 3), n=64, tissue_cov={2: 0.0, 3: 0.0})
    np.testing.assert_array_equal(s[:, 0], np.full(64, TISSUE_CONDUCTIVITY[2]))
    np.testing.assert_array_equal(s[:, 1], np.full(64, TISSUE_CONDUCTIVITY[3]))


def test_normal_distribution_is_symmetric(cp):
    cov = 0.1
    s = draw(cp, (2,), tissue_cov={2: cov}, distribution="normal")[:, 0]
    nominal = TISSUE_CONDUCTIVITY[2]
    assert s.mean() / nominal == pytest.approx(1.0, rel=1e-2)
    assert s.std() / nominal == pytest.approx(cov, rel=2e-2)


def test_normal_distribution_clamped(cp):
    """A wide normal model wanders non-physical; the floor must actually engage."""
    nominal = TISSUE_CONDUCTIVITY[2]
    s = draw(cp, (2,), n=100_000, tissue_cov={2: 5.0}, distribution="normal")[:, 0]
    floor = 1e-6 * nominal
    assert s.min() == floor
    assert np.all(s >= floor)
    assert (s == floor).mean() > 0.01  # the clamp branch is genuinely exercised


def test_seed_reproducibility(cp):
    a = draw(cp, (1, 2), n=1000, seed=7)
    b = draw(cp, (1, 2), n=1000, seed=7)
    c = draw(cp, (1, 2), n=1000, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_columns_are_independent(cp):
    """Tissues are modelled as independent random variables; the draws must reflect that."""
    s = np.log(draw(cp, (1, 2, 3), n=100_000, tissue_cov={1: 0.2, 2: 0.2, 3: 0.2}))
    corr = np.corrcoef(s.T)
    off = corr[~np.eye(3, dtype=bool)]
    assert np.abs(off).max() < 0.02


def test_each_column_uses_its_own_nominal_and_cov(cp):
    tags = (1, 2, 3)
    covs = {1: 0.05, 2: 0.2, 3: 0.5}
    s = draw(cp, tags, tissue_cov=covs)
    for i, tag in enumerate(tags):
        assert s[:, i].mean() / TISSUE_CONDUCTIVITY[tag] == pytest.approx(1.0, rel=1e-2)
        assert s[:, i].std() / s[:, i].mean() == pytest.approx(covs[tag], rel=5e-2)


def test_cov_for_override_precedence():
    """An explicit tissue_cov entry wins over the default table; unknown tags raise."""
    cfg = ConductivityUQConfig(tissue_cov={2: 0.99})
    assert cfg.cov_for(2) == 0.99
    assert cfg.cov_for(3) == DEFAULT_TISSUE_COV[3]
    with pytest.raises(KeyError):
        ConductivityUQConfig().cov_for(999)


def test_default_cov_table_is_used_when_unset(cp):
    s = draw(cp, (3,))[:, 0]
    assert s.std() / s.mean() == pytest.approx(DEFAULT_TISSUE_COV[3], rel=5e-2)
