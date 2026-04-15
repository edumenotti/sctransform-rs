"""Correctness tests for compute_residuals().

Ground truth is computed with a straightforward numpy implementation of the
SCTransform v2 Pearson residual formula. The Rust kernel must match this
within rtol=1e-10 (pure arithmetic, no IRLS or other iterative step here, so
tolerance is tight).

End-to-end cross-validation against R sctransform::vst() is deferred to
v0.4.0 (see project memory: validation_r_deferred.md).
"""
from __future__ import annotations

import numpy as np
import pytest

import sctransform_rs as st


def numpy_reference(
    umi: np.ndarray,
    theta: np.ndarray,
    beta0: np.ndarray,
    beta1: np.ndarray,
    total_umi: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """Reference Pearson residual implementation in pure numpy.

    mu = exp(beta0 + beta1 * log10(total_umi))  → shape (n_genes, n_cells)
    var = mu + mu^2 / theta
    z = (x - mu) / sqrt(var), clipped to ±sqrt(n_cells / 30)
    """
    log10_total = np.log10(total_umi)[np.newaxis, :]      # (1, n_cells)
    lin = beta0[:, np.newaxis] + beta1[:, np.newaxis] * log10_total
    mu = np.exp(lin)                                      # (n_genes, n_cells)
    var = mu + (mu ** 2) / theta[:, np.newaxis]
    z = (umi - mu) / np.sqrt(var)
    if clip:
        k = np.sqrt(umi.shape[1] / 30.0)
        np.clip(z, -k, k, out=z)
    return z


def _synthetic_dataset(n_genes: int, n_cells: int, seed: int = 0):
    """Synthetic UMI matrix + plausible per-gene params for testing."""
    rng = np.random.default_rng(seed)

    # Per-cell library size varies across 2 orders of magnitude, typical of
    # 10x Chromium scRNA-seq.
    total_umi = rng.integers(500, 50_000, size=n_cells).astype(np.float64)

    # Gene-level parameters spanning the realistic sctransform v2 range.
    theta = rng.uniform(0.5, 50.0, size=n_genes)
    beta0 = rng.uniform(-12.0, -3.0, size=n_genes)
    beta1 = np.full(n_genes, np.log(10.0) / np.log(10.0))  # fixed slope (offset model)

    # Draw UMI counts from a negative binomial matching these params.
    mu = np.exp(beta0[:, None] + beta1[:, None] * np.log10(total_umi)[None, :])
    # scipy-style NB parameterization: number of successes = theta, prob = theta/(theta+mu)
    p = theta[:, None] / (theta[:, None] + mu)
    umi = rng.negative_binomial(theta[:, None], p).astype(np.float64)

    return umi, theta, beta0, beta1, total_umi


@pytest.mark.parametrize("clip", [True, False])
def test_matches_numpy_reference_small(clip):
    umi, theta, beta0, beta1, total = _synthetic_dataset(n_genes=20, n_cells=100, seed=42)

    rust = st.compute_residuals(umi, theta, beta0, beta1, total, clip=clip)
    ref = numpy_reference(umi, theta, beta0, beta1, total, clip=clip)

    np.testing.assert_allclose(rust, ref, rtol=1e-10, atol=1e-12)


def test_matches_numpy_reference_pbmc_like():
    """Realistic-sized test: ~2k genes × 3k cells, similar to PBMC 3k HVGs."""
    umi, theta, beta0, beta1, total = _synthetic_dataset(
        n_genes=2000, n_cells=3000, seed=7
    )

    rust = st.compute_residuals(umi, theta, beta0, beta1, total)
    ref = numpy_reference(umi, theta, beta0, beta1, total)

    np.testing.assert_allclose(rust, ref, rtol=1e-10, atol=1e-12)


def test_clip_bounds():
    """Residuals must lie in [-sqrt(N/30), sqrt(N/30)] when clip=True."""
    umi, theta, beta0, beta1, total = _synthetic_dataset(n_genes=100, n_cells=600, seed=1)
    z = st.compute_residuals(umi, theta, beta0, beta1, total, clip=True)
    k = np.sqrt(600 / 30.0)
    assert z.max() <= k + 1e-12
    assert z.min() >= -k - 1e-12


def test_total_umi_defaults_to_column_sum():
    """If total_umi is None, it should be computed as umi.sum(axis=0)."""
    umi, theta, beta0, beta1, total = _synthetic_dataset(n_genes=10, n_cells=50, seed=3)
    actual_total = umi.sum(axis=0)

    z_explicit = st.compute_residuals(umi, theta, beta0, beta1, actual_total)
    z_implicit = st.compute_residuals(umi, theta, beta0, beta1, None)

    np.testing.assert_array_equal(z_explicit, z_implicit)


def test_shape_mismatch_raises():
    umi = np.zeros((5, 10))
    with pytest.raises(ValueError, match="theta has length"):
        st.compute_residuals(umi, np.ones(4), np.zeros(5), np.ones(5), np.ones(10))
    with pytest.raises(ValueError, match="total_umi has length"):
        st.compute_residuals(umi, np.ones(5), np.zeros(5), np.ones(5), np.ones(9))


def test_dtype_coercion():
    """Integer input should be silently cast to float64."""
    umi_int = np.arange(30, dtype=np.int32).reshape(3, 10)
    theta = np.ones(3)
    beta0 = np.full(3, -3.0)
    beta1 = np.ones(3)
    total = np.full(10, 1000.0)

    z = st.compute_residuals(umi_int, theta, beta0, beta1, total)
    assert z.dtype == np.float64
    assert z.shape == (3, 10)
