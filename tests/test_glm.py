"""Tests for the Negative Binomial GLM offset-model fitter.

The offset model fixes beta1 = ln(10) and fits only beta0 + theta per gene.

Validation strategy:
  - beta0 is a closed-form Poisson MLE → test exact recovery against the
    analytical formula (tolerance from finite-sample variance only).
  - theta is fit via Newton-Raphson → test against scipy's NB profile
    log-likelihood maximiser as an independent oracle.
  - Roundtrip test: fit → residuals → verify statistical properties.
"""

import numpy as np
import pytest
from scipy.special import gammaln, digamma
from scipy.optimize import minimize_scalar

import sctransform_rs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_nb(
    n_genes: int,
    n_cells: int,
    theta_true: np.ndarray,
    beta0_true: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate UMI counts from the SCTransform offset NB model."""
    rng = np.random.default_rng(seed)
    total_umi = rng.lognormal(mean=np.log(5000.0), sigma=0.3, size=n_cells)

    umi = np.empty((n_genes, n_cells), dtype=np.float64)
    for g in range(n_genes):
        mu = np.exp(beta0_true[g]) * total_umi
        p = theta_true[g] / (theta_true[g] + mu)
        umi[g] = rng.negative_binomial(n=theta_true[g], p=p).astype(np.float64)
    return umi, total_umi


def _nb_loglik(theta: float, x: np.ndarray, mu: np.ndarray) -> float:
    """Negative Binomial log-likelihood (summed over cells) for fixed mu."""
    return np.sum(
        gammaln(x + theta)
        - gammaln(theta)
        - gammaln(x + 1)
        + theta * np.log(theta / (theta + mu))
        + x * np.log(mu / (theta + mu))
    )


def _scipy_theta_mle(x: np.ndarray, mu: np.ndarray) -> float:
    """Independent theta MLE via scipy (golden-section on log-likelihood)."""
    result = minimize_scalar(
        lambda log_t: -_nb_loglik(np.exp(log_t), x, mu),
        bounds=(np.log(1e-4), np.log(1e6)),
        method="bounded",
    )
    return np.exp(result.x)


# ---------------------------------------------------------------------------
# Shape / API tests
# ---------------------------------------------------------------------------

def test_fit_glm_offset_returns_expected_keys_and_shapes():
    rng = np.random.default_rng(0)
    umi = rng.poisson(lam=1.0, size=(5, 100)).astype(np.float64)
    out = sctransform_rs.fit_glm_offset(umi)
    assert set(out) == {"theta", "beta0", "beta1"}
    assert out["theta"].shape == (5,)
    assert out["beta0"].shape == (5,)
    assert out["beta1"].shape == (5,)
    assert np.allclose(out["beta1"], np.log(10.0))


def test_total_umi_none_uses_column_sum():
    rng = np.random.default_rng(0)
    umi = rng.poisson(lam=2.0, size=(3, 200)).astype(np.float64)
    out_auto = sctransform_rs.fit_glm_offset(umi)
    out_manual = sctransform_rs.fit_glm_offset(umi, umi.sum(axis=0))
    assert np.allclose(out_auto["theta"], out_manual["theta"])
    assert np.allclose(out_auto["beta0"], out_manual["beta0"])


# ---------------------------------------------------------------------------
# beta0 recovery — closed-form, so should be near-exact
# ---------------------------------------------------------------------------

class TestBeta0Recovery:
    """beta0 = log(sum_x / sum_total) is a closed-form Poisson MLE.

    The only source of error is sampling noise in the NB draws.
    With 50k cells the sampling variance of log(mean(x)/mean(total)) is
    negligible, so we can use tight tolerances.
    """

    @pytest.mark.parametrize("n_cells", [10_000, 50_000])
    def test_beta0_tight_recovery(self, n_cells):
        n_genes = 20
        rng = np.random.default_rng(42)
        theta_true = rng.uniform(1.0, 50.0, size=n_genes)
        beta0_true = rng.uniform(-9.0, -5.0, size=n_genes)

        umi, total_umi = _simulate_nb(n_genes, n_cells, theta_true, beta0_true, seed=42)

        # Analytical beta0 from the data (what our Rust code should compute).
        beta0_analytical = np.log(umi.sum(axis=1) / total_umi.sum())

        out = sctransform_rs.fit_glm_offset(umi, total_umi)

        # Rust beta0 must match the analytical formula to machine precision.
        np.testing.assert_allclose(
            out["beta0"],
            beta0_analytical,
            rtol=1e-12,
            err_msg="beta0 diverges from closed-form Poisson MLE",
        )

    def test_beta0_matches_analytical_formula_exactly(self):
        """Deterministic check: beta0 = log(row_sum / col_sum_total)."""
        # Use fixed small data so there's no randomness.
        umi = np.array([[10.0, 20.0, 30.0],
                        [0.0,   5.0, 15.0],
                        [100.0, 200.0, 300.0]])
        total = np.array([1000.0, 2000.0, 3000.0])
        expected_beta0 = np.log(umi.sum(axis=1) / total.sum())

        out = sctransform_rs.fit_glm_offset(umi, total)
        np.testing.assert_allclose(
            out["beta0"],
            expected_beta0,
            rtol=1e-14,
            err_msg="beta0 is not the exact Poisson MLE",
        )


# ---------------------------------------------------------------------------
# theta recovery — compare Rust Newton-Raphson vs scipy oracle
# ---------------------------------------------------------------------------

class TestThetaRecovery:
    """Theta (dispersion) tests compare Rust against scipy's NB MLE.

    This is algorithm-vs-algorithm, not algorithm-vs-ground-truth, so the
    tolerance reflects optimizer agreement, not sampling variance.
    """

    def _fit_and_compare_theta(
        self, n_genes, n_cells, theta_true, beta0_true, seed, rtol
    ):
        umi, total_umi = _simulate_nb(n_genes, n_cells, theta_true, beta0_true, seed=seed)
        out = sctransform_rs.fit_glm_offset(umi, total_umi, max_iter=100)

        for g in range(n_genes):
            mu_g = np.exp(out["beta0"][g]) * total_umi
            theta_scipy = _scipy_theta_mle(umi[g], mu_g)

            np.testing.assert_allclose(
                out["theta"][g],
                theta_scipy,
                rtol=rtol,
                err_msg=(
                    f"gene {g}: Rust theta={out['theta'][g]:.6f}, "
                    f"scipy theta={theta_scipy:.6f}, "
                    f"true theta={theta_true[g]:.6f}"
                ),
            )

    def test_theta_vs_scipy_moderate_dispersion(self):
        """theta in [1, 50] — typical scRNA-seq range."""
        n_genes, n_cells = 30, 20_000
        rng = np.random.default_rng(7)
        theta_true = rng.uniform(1.0, 50.0, size=n_genes)
        beta0_true = rng.uniform(-8.0, -5.0, size=n_genes)
        self._fit_and_compare_theta(
            n_genes, n_cells, theta_true, beta0_true, seed=7, rtol=1e-3
        )

    def test_theta_vs_scipy_high_dispersion(self):
        """theta in [0.1, 1] — highly overdispersed genes."""
        n_genes, n_cells = 20, 20_000
        rng = np.random.default_rng(99)
        theta_true = rng.uniform(0.1, 1.0, size=n_genes)
        beta0_true = rng.uniform(-7.0, -5.0, size=n_genes)
        self._fit_and_compare_theta(
            n_genes, n_cells, theta_true, beta0_true, seed=99, rtol=1e-3
        )

    def test_theta_vs_scipy_low_dispersion(self):
        """theta in [50, 500] — near-Poisson genes."""
        n_genes, n_cells = 20, 20_000
        rng = np.random.default_rng(13)
        theta_true = rng.uniform(50.0, 500.0, size=n_genes)
        beta0_true = rng.uniform(-7.0, -5.0, size=n_genes)
        self._fit_and_compare_theta(
            n_genes, n_cells, theta_true, beta0_true, seed=13, rtol=1e-3
        )

    def test_theta_vs_scipy_extreme_range(self):
        """theta spanning 4 orders of magnitude [0.01, 100]."""
        n_genes, n_cells = 40, 30_000
        rng = np.random.default_rng(0)
        theta_true = 10 ** rng.uniform(-2.0, 2.0, size=n_genes)
        beta0_true = rng.uniform(-8.0, -5.0, size=n_genes)
        self._fit_and_compare_theta(
            n_genes, n_cells, theta_true, beta0_true, seed=0, rtol=1e-3
        )

    def test_theta_recovery_improves_with_more_cells(self):
        """Sanity: doubling cells should reduce Rust-vs-scipy gap."""
        n_genes = 10
        rng = np.random.default_rng(55)
        theta_true = rng.uniform(2.0, 20.0, size=n_genes)
        beta0_true = rng.uniform(-7.0, -6.0, size=n_genes)

        gaps = []
        for n_cells in [5_000, 20_000]:
            umi, total_umi = _simulate_nb(n_genes, n_cells, theta_true, beta0_true, seed=55)
            out = sctransform_rs.fit_glm_offset(umi, total_umi, max_iter=100)
            rel_errors = []
            for g in range(n_genes):
                mu_g = np.exp(out["beta0"][g]) * total_umi
                theta_scipy = _scipy_theta_mle(umi[g], mu_g)
                rel_errors.append(abs(out["theta"][g] - theta_scipy) / theta_scipy)
            gaps.append(np.median(rel_errors))

        assert gaps[1] <= gaps[0], (
            f"More cells should improve agreement: "
            f"5k median rel_err={gaps[0]:.4f}, 20k={gaps[1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Degenerate / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_gene_produces_finite_params(self):
        rng = np.random.default_rng(0)
        umi = rng.poisson(lam=1.0, size=(3, 500)).astype(np.float64)
        umi[1] = 0.0
        out = sctransform_rs.fit_glm_offset(umi)
        assert np.all(np.isfinite(out["theta"]))
        assert np.all(np.isfinite(out["beta0"]))

    def test_single_cell(self):
        umi = np.array([[5.0], [0.0], [100.0]])
        total = np.array([1000.0])
        out = sctransform_rs.fit_glm_offset(umi, total)
        assert np.all(np.isfinite(out["theta"]))
        assert np.all(np.isfinite(out["beta0"]))

    def test_very_sparse_gene(self):
        """Gene detected in only 1 out of 10k cells."""
        rng = np.random.default_rng(0)
        n_cells = 10_000
        umi = np.zeros((1, n_cells), dtype=np.float64)
        umi[0, 42] = 1.0
        total = rng.lognormal(mean=np.log(5000.0), sigma=0.3, size=n_cells)
        out = sctransform_rs.fit_glm_offset(umi, total)
        assert np.all(np.isfinite(out["theta"]))
        assert np.all(np.isfinite(out["beta0"]))

    def test_constant_total_umi(self):
        """All cells have the same library size."""
        rng = np.random.default_rng(0)
        n_genes, n_cells = 5, 5000
        total = np.full(n_cells, 3000.0)
        theta_true = np.full(n_genes, 10.0)
        beta0_true = np.full(n_genes, -7.0)
        umi, _ = _simulate_nb(n_genes, n_cells, theta_true, beta0_true, seed=0)
        out = sctransform_rs.fit_glm_offset(umi, total)
        assert np.all(np.isfinite(out["theta"]))
        assert np.all(np.isfinite(out["beta0"]))


# ---------------------------------------------------------------------------
# Roundtrip: fit → residuals → statistical properties
# ---------------------------------------------------------------------------

class TestRoundtrip:

    def test_residuals_shape_and_finite(self):
        n_genes, n_cells = 10, 5000
        theta_true = np.full(n_genes, 10.0)
        beta0_true = np.full(n_genes, -7.0)
        umi, total_umi = _simulate_nb(n_genes, n_cells, theta_true, beta0_true, seed=0)

        params = sctransform_rs.fit_glm_offset(umi, total_umi)
        residuals = sctransform_rs.compute_residuals(
            umi, params["theta"], params["beta0"], params["beta1"], total_umi,
        )
        assert residuals.shape == (n_genes, n_cells)
        assert np.all(np.isfinite(residuals))

    def test_residuals_approximately_standardised(self):
        """If the model is correct, unclipped residuals should have mean≈0, std≈1."""
        n_genes, n_cells = 20, 50_000
        rng = np.random.default_rng(12)
        theta_true = rng.uniform(5.0, 30.0, size=n_genes)
        beta0_true = rng.uniform(-7.0, -5.5, size=n_genes)
        umi, total_umi = _simulate_nb(n_genes, n_cells, theta_true, beta0_true, seed=12)

        params = sctransform_rs.fit_glm_offset(umi, total_umi, max_iter=100)
        residuals = sctransform_rs.compute_residuals(
            umi, params["theta"], params["beta0"], params["beta1"],
            total_umi, clip=False,
        )

        gene_means = residuals.mean(axis=1)
        gene_stds = residuals.std(axis=1)

        # With 50k cells and correct model, mean should be very close to 0.
        np.testing.assert_allclose(
            gene_means, 0.0, atol=0.05,
            err_msg=f"Residual means far from 0: {gene_means}",
        )
        # Std should be close to 1 (NB Pearson residuals are not exactly
        # unit-variance, but for moderate theta they're close).
        assert np.all(gene_stds > 0.7), f"Some gene stds too low: {gene_stds}"
        assert np.all(gene_stds < 1.5), f"Some gene stds too high: {gene_stds}"
