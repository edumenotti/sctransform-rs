"""Tests for the high-level vst() function and AnnData integration."""

import numpy as np
import pytest

import sctransform_rs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_nb(n_genes, n_cells, theta, beta0, seed=0):
    rng = np.random.default_rng(seed)
    total_umi = rng.lognormal(mean=np.log(5000.0), sigma=0.3, size=n_cells)
    umi = np.empty((n_genes, n_cells), dtype=np.float64)
    for g in range(n_genes):
        mu = np.exp(beta0[g]) * total_umi
        p = theta[g] / (theta[g] + mu)
        umi[g] = rng.negative_binomial(n=theta[g], p=p).astype(np.float64)
    return umi, total_umi


# ---------------------------------------------------------------------------
# vst() tests
# ---------------------------------------------------------------------------

class TestVst:

    def test_returns_all_keys(self):
        rng = np.random.default_rng(0)
        umi = rng.poisson(lam=2.0, size=(5, 200)).astype(np.float64)
        out = sctransform_rs.vst(umi)
        assert set(out) == {"residuals", "theta", "beta0", "beta1"}
        assert out["residuals"].shape == (5, 200)
        assert out["theta"].shape == (5,)

    def test_equivalent_to_manual_pipeline(self):
        """vst() must produce identical output to fit_glm_offset + compute_residuals."""
        n_genes, n_cells = 15, 3000
        rng = np.random.default_rng(42)
        theta_true = rng.uniform(2.0, 30.0, size=n_genes)
        beta0_true = rng.uniform(-8.0, -5.0, size=n_genes)
        umi, total_umi = _simulate_nb(n_genes, n_cells, theta_true, beta0_true, seed=42)

        # Manual pipeline
        params = sctransform_rs.fit_glm_offset(umi, total_umi)
        residuals_manual = sctransform_rs.compute_residuals(
            umi, params["theta"], params["beta0"], params["beta1"], total_umi,
        )

        # vst()
        out = sctransform_rs.vst(umi, total_umi)

        np.testing.assert_array_equal(out["residuals"], residuals_manual)
        np.testing.assert_array_equal(out["theta"], params["theta"])
        np.testing.assert_array_equal(out["beta0"], params["beta0"])

    def test_clip_false(self):
        rng = np.random.default_rng(0)
        umi = rng.poisson(lam=1.0, size=(3, 500)).astype(np.float64)
        out_clip = sctransform_rs.vst(umi, clip=True)
        out_noclip = sctransform_rs.vst(umi, clip=False)
        # Parameters should be identical regardless of clip.
        np.testing.assert_array_equal(out_clip["theta"], out_noclip["theta"])
        # Residuals may differ where clipping is active.
        assert np.all(np.isfinite(out_noclip["residuals"]))

    def test_total_umi_auto(self):
        rng = np.random.default_rng(0)
        umi = rng.poisson(lam=2.0, size=(5, 300)).astype(np.float64)
        out_auto = sctransform_rs.vst(umi)
        out_manual = sctransform_rs.vst(umi, total_umi=umi.sum(axis=0))
        np.testing.assert_array_equal(out_auto["residuals"], out_manual["residuals"])

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            sctransform_rs.vst(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# AnnData integration tests
# ---------------------------------------------------------------------------

class TestAnnDataIntegration:

    @pytest.fixture
    def adata(self):
        import anndata as ad

        rng = np.random.default_rng(0)
        n_cells, n_genes = 500, 20
        X = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float64)
        return ad.AnnData(X=X)

    def test_inplace_writes_layer_and_var(self, adata):
        from sctransform_rs.anndata import vst_anndata

        result = vst_anndata(adata, inplace=True)
        assert result is None
        assert "sct_residuals" in adata.layers
        assert adata.layers["sct_residuals"].shape == adata.X.shape
        assert "sct_theta" in adata.var.columns
        assert "sct_beta0" in adata.var.columns
        assert "sct_beta1" in adata.var.columns
        assert np.all(np.isfinite(adata.layers["sct_residuals"]))

    def test_not_inplace_returns_new_adata(self, adata):
        from sctransform_rs.anndata import vst_anndata

        adata_out = vst_anndata(adata, inplace=False)
        assert adata_out is not adata
        assert adata_out.X.shape == adata.X.shape
        assert "sct_theta" in adata_out.var.columns
        # Original unchanged
        assert "sct_residuals" not in adata.layers

    def test_matches_raw_vst(self, adata):
        """AnnData wrapper must produce identical residuals to raw vst()."""
        from sctransform_rs.anndata import vst_anndata

        # Raw vst: genes × cells
        umi_gc = np.ascontiguousarray(adata.X.T, dtype=np.float64)
        raw = sctransform_rs.vst(umi_gc)

        vst_anndata(adata, inplace=True)

        # adata.layers is cells × genes, raw residuals is genes × cells
        np.testing.assert_array_equal(
            adata.layers["sct_residuals"],
            raw["residuals"].T,
        )

    def test_sparse_input(self):
        """Sparse CSR input should be densified and produce valid output."""
        import anndata as ad
        import scipy.sparse as sp

        rng = np.random.default_rng(0)
        X = sp.csr_matrix(rng.poisson(lam=0.5, size=(200, 10)).astype(np.float64))
        adata = ad.AnnData(X=X)

        from sctransform_rs.anndata import vst_anndata

        vst_anndata(adata, inplace=True)
        assert np.all(np.isfinite(adata.layers["sct_residuals"]))

    def test_layer_parameter(self):
        import anndata as ad
        from sctransform_rs.anndata import vst_anndata

        rng = np.random.default_rng(0)
        n_cells, n_genes = 300, 10
        X_normalized = rng.random((n_cells, n_genes))
        counts = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float64)
        adata = ad.AnnData(X=X_normalized)
        adata.layers["counts"] = counts

        vst_anndata(adata, layer="counts", inplace=True)

        # Should have used the counts layer, not X
        umi_gc = np.ascontiguousarray(counts.T, dtype=np.float64)
        raw = sctransform_rs.vst(umi_gc)
        np.testing.assert_array_equal(
            adata.layers["sct_residuals"],
            raw["residuals"].T,
        )
