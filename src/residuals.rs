//! Pearson residual computation for SCTransform v2.
//!
//! For each gene `g` and cell `c`:
//!   μ_gc = exp(β₀_g + β₁_g · log10(total_umi_c))
//!   σ_gc = sqrt(μ_gc + μ_gc² / θ_g)
//!   z_gc = (x_gc − μ_gc) / σ_gc
//!
//! Residuals are optionally clipped to ±sqrt(N/30), where N = number of cells,
//! matching `sctransform::vst(vst.flavor = "v2")`.
//!
//! Parallelism: each gene is an independent row of the output, so we parallelize
//! over rows with rayon. `axis_iter_mut(Axis(0))` + `into_par_iter()` is the
//! canonical ndarray+rayon pattern for row-wise independent work.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use rayon::prelude::*;

/// Compute Pearson residuals for a dense genes × cells UMI matrix.
///
/// All per-gene parameter arrays (`theta`, `beta0`, `beta1`) must have length
/// equal to the number of rows of `umi`. `total_umi` must have length equal to
/// the number of columns. Length mismatches are caller's responsibility — the
/// PyO3 wrapper validates them before calling this function.
pub fn compute_residuals_dense(
    umi: &Array2<f64>,
    theta: ArrayView1<f64>,
    beta0: ArrayView1<f64>,
    beta1: ArrayView1<f64>,
    total_umi: ArrayView1<f64>,
    clip: bool,
) -> Array2<f64> {
    let (n_genes, n_cells) = umi.dim();

    // log10(total_umi) is reused for every gene — precompute once.
    let log10_total: Array1<f64> = total_umi.mapv(f64::log10);

    let clip_val = if clip {
        Some((n_cells as f64 / 30.0).sqrt())
    } else {
        None
    };

    let mut residuals: Array2<f64> = Array2::zeros((n_genes, n_cells));

    residuals
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(g, row)| {
            compute_gene_row(
                umi.row(g),
                row,
                log10_total.view(),
                theta[g],
                beta0[g],
                beta1[g],
                clip_val,
            );
        });

    residuals
}

#[inline]
fn compute_gene_row(
    umi_row: ArrayView1<f64>,
    mut out_row: ArrayViewMut1<f64>,
    log10_total: ArrayView1<f64>,
    theta_g: f64,
    beta0_g: f64,
    beta1_g: f64,
    clip: Option<f64>,
) {
    let inv_theta = 1.0 / theta_g;
    for (c, &x) in umi_row.iter().enumerate() {
        // Cells with zero total UMI have no information → residual = 0.
        if !log10_total[c].is_finite() {
            out_row[c] = 0.0;
            continue;
        }
        let mu = (beta0_g + beta1_g * log10_total[c]).exp();
        let var = mu + mu * mu * inv_theta;
        let sigma = var.sqrt();
        let mut z = (x - mu) / sigma;
        if let Some(k) = clip {
            if z > k {
                z = k;
            } else if z < -k {
                z = -k;
            }
        }
        out_row[c] = z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn single_gene_matches_manual_formula() {
        // Single gene, 3 cells. theta=10, beta0=-3.0, beta1=1.0.
        // total_umi = [1000, 2000, 500] → log10 = [3, log10(2000), log10(500)]
        let umi = array![[1.0, 4.0, 0.0]];
        let theta = array![10.0];
        let beta0 = array![-3.0];
        let beta1 = array![1.0];
        let total = array![1000.0, 2000.0, 500.0];

        let out = compute_residuals_dense(
            &umi,
            theta.view(),
            beta0.view(),
            beta1.view(),
            total.view(),
            false,
        );

        // Cell 0: mu = exp(-3 + 1 * log10(1000)) = exp(0) = 1.0
        //         var = 1 + 1/10 = 1.1, sigma = sqrt(1.1)
        //         z = (1 - 1) / sqrt(1.1) = 0.0
        assert!((out[[0, 0]] - 0.0).abs() < 1e-12);

        // Cell 2: mu = exp(-3 + log10(500)) = exp(-3 + 2.69897) = exp(-0.30103) ≈ 0.5
        let mu_c2 = (-3.0_f64 + 500.0_f64.log10()).exp();
        let sigma_c2 = (mu_c2 + mu_c2 * mu_c2 / 10.0).sqrt();
        let expected_c2 = (0.0 - mu_c2) / sigma_c2;
        assert!((out[[0, 2]] - expected_c2).abs() < 1e-12);
    }

    #[test]
    fn clip_bounds_respected() {
        // Force an extreme residual and check clipping.
        // 300 cells → clip = sqrt(300/30) = sqrt(10) ≈ 3.162
        let n_cells = 300;
        let umi = Array2::from_elem((1, n_cells), 1000.0); // huge count
        let theta = array![100.0];
        let beta0 = array![-10.0]; // forces tiny mu
        let beta1 = array![0.0];
        let total = Array1::from_elem(n_cells, 1000.0);

        let out = compute_residuals_dense(
            &umi,
            theta.view(),
            beta0.view(),
            beta1.view(),
            total.view(),
            true,
        );

        let clip_val = (n_cells as f64 / 30.0).sqrt();
        for &z in out.iter() {
            assert!(
                z <= clip_val + 1e-12,
                "residual {} exceeds clip {}",
                z,
                clip_val
            );
            assert!(z >= -clip_val - 1e-12);
        }
        // All cells hit the positive clip
        assert!((out[[0, 0]] - clip_val).abs() < 1e-12);
    }

    #[test]
    fn parallel_matches_sequential() {
        // Generate a deterministic matrix and check that a second run produces
        // the same result (rayon should not introduce nondeterminism in this
        // embarrassingly-parallel-per-row computation).
        let n_genes = 50;
        let n_cells = 200;
        let umi: Array2<f64> =
            Array2::from_shape_fn((n_genes, n_cells), |(g, c)| ((g * 31 + c * 7) % 10) as f64);
        let theta: Array1<f64> = Array1::from_shape_fn(n_genes, |g| 1.0 + g as f64 * 0.1);
        let beta0: Array1<f64> = Array1::from_shape_fn(n_genes, |g| -3.0 + g as f64 * 0.01);
        let beta1: Array1<f64> = Array1::from_elem(n_genes, 1.0);
        let total: Array1<f64> = Array1::from_shape_fn(n_cells, |c| 500.0 + c as f64);

        let a = compute_residuals_dense(
            &umi,
            theta.view(),
            beta0.view(),
            beta1.view(),
            total.view(),
            true,
        );
        let b = compute_residuals_dense(
            &umi,
            theta.view(),
            beta0.view(),
            beta1.view(),
            total.view(),
            true,
        );
        assert_eq!(a, b);
    }
}
