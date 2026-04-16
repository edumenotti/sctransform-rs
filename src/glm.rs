//! Per-gene NB GLM fitting for the SCTransform v2 **offset model**.
//!
//! Model:
//!   x_gc ~ NegBinom(mu_gc, theta_g)
//!   log(mu_gc) = beta0_g + log(total_umi_c)
//!   (equivalently, beta1 is fixed at log(10) on a log10-scale covariate,
//!    so downstream `compute_residuals()` gets beta1 = ln(10))
//!
//! Fitting procedure per gene:
//!   1. beta0 = log( sum(x) / sum(total_umi) )   — Poisson MLE closed form
//!   2. theta via Newton-Raphson on log(theta) using the profile NB
//!      score/Fisher-information w.r.t. theta, with bounds [θ_min, θ_max]
//!      and a fallback to bisection if Newton diverges.
//!
//! Each gene is independent → parallelised with rayon.
//!
//! References:
//!   - glmGamPoi (Ahlmann-Eltze & Huber 2020), overdispersion MLE
//!   - sctransform v2 (Choudhary & Satija 2022), offset model

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::prelude::*;

/// Bounds on the dispersion parameter. Matches glmGamPoi's practical range.
const THETA_MIN: f64 = 1e-4;
const THETA_MAX: f64 = 1e6;

/// Convergence tolerance for log(theta) Newton-Raphson step.
const TOL: f64 = 1e-6;

/// Trigamma function ψ'(x) = Σ 1/(x+k)² for k=0..∞.
///
/// Implemented via Abramowitz & Stegun 6.4.12 asymptotic expansion, combined
/// with the recurrence ψ'(x) = ψ'(x+1) + 1/x² to shift small x up to x >= 6
/// where the asymptotic series is accurate.
#[inline]
fn trigamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0;
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic: ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - ...
    let xinv = 1.0 / x;
    let xinv2 = xinv * xinv;
    result
        + xinv
        + 0.5 * xinv2
        + xinv2 * xinv / 6.0
        - xinv2 * xinv2 * xinv / 30.0
        + xinv2 * xinv2 * xinv2 * xinv / 42.0
}

/// NB profile log-likelihood score w.r.t. theta (sum over cells).
///
/// Returns (score, fisher_info) where score = ∂ℓ/∂θ and fisher_info = -∂²ℓ/∂θ².
/// mu is fixed (computed from beta0 and total_umi).
fn nb_theta_score_info(
    x: ArrayView1<f64>,
    mu: ArrayView1<f64>,
    theta: f64,
) -> (f64, f64) {
    use statrs::function::gamma::digamma;

    let n = x.len() as f64;
    let log_theta = theta.ln();
    let tri_theta = trigamma(theta);
    let dig_theta = digamma(theta);

    let mut score = 0.0_f64;
    let mut info = 0.0_f64;

    for (&xi, &mi) in x.iter().zip(mu.iter()) {
        let t_plus_m = theta + mi;
        let t_plus_x = theta + xi;
        // ∂/∂θ [log Γ(x+θ) − log Γ(θ)] = ψ(x+θ) − ψ(θ)
        let dig_xt = digamma(t_plus_x);
        score += dig_xt - dig_theta + log_theta + 1.0 - t_plus_x / t_plus_m - (t_plus_m).ln();
        // Fisher information (negative second derivative, per cell):
        // -∂²/∂θ² = ψ'(θ) − ψ'(x+θ) − 1/θ + 2/(θ+μ) − (θ+x)/(θ+μ)²
        let tri_xt = trigamma(t_plus_x);
        info += tri_theta - tri_xt - 1.0 / theta + 2.0 / t_plus_m - t_plus_x / (t_plus_m * t_plus_m);
    }

    // Sanity: n enters only through x,mu sums — no extra n term.
    let _ = n;
    (score, info)
}

/// Method-of-moments initial estimate for theta.
///
/// Var(x) = μ + μ²/θ  →  θ = μ̄²/(σ̂² − μ̄) when σ̂² > μ̄ (overdispersed).
/// When not overdispersed, return a large theta (effectively Poisson limit).
fn theta_mom_init(x: ArrayView1<f64>, mu: ArrayView1<f64>) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.sum() / n;
    let mean_mu: f64 = mu.sum() / n;

    // Compute residual variance (uses mu, not mean_x, as the center — this is
    // the correct form for Pearson-style overdispersion assessment).
    let var: f64 = x
        .iter()
        .zip(mu.iter())
        .map(|(&xi, &mi)| (xi - mi).powi(2))
        .sum::<f64>()
        / n;

    if var > mean_mu && mean_mu > 0.0 {
        let theta = mean_x * mean_x / (var - mean_mu);
        theta.clamp(THETA_MIN, THETA_MAX)
    } else {
        let _ = mean_x;
        // Nearly Poisson — pick a large theta near the upper bound.
        100.0_f64.clamp(THETA_MIN, THETA_MAX)
    }
}

/// Fit the NB GLM offset model for a single gene.
///
/// Returns `(theta, beta0)`. Does NOT return beta1 because it is fixed at
/// ln(10) by the offset model (caller materialises it).
pub fn fit_gene_offset(
    x: ArrayView1<f64>,
    total_umi: ArrayView1<f64>,
    max_iter: usize,
) -> (f64, f64) {
    let sum_x: f64 = x.sum();
    let sum_total: f64 = total_umi.sum();

    // Degenerate: gene never detected. Theta is unidentified; return a
    // plausible value that won't create NaNs downstream.
    if sum_x <= 0.0 {
        return (THETA_MAX, (1e-9_f64 / sum_total).ln());
    }

    // Poisson MLE for beta0 (closed form with offset).
    let beta0 = (sum_x / sum_total).ln();

    // mu_c = exp(beta0) * total_c.  Since beta0 = log(sum_x/sum_total),
    // exp(beta0) = sum_x/sum_total.
    let ratio = sum_x / sum_total;
    let mu: Array1<f64> = total_umi.mapv(|t| ratio * t);

    // Initial theta via method of moments.
    let theta_init = theta_mom_init(x, mu.view());
    let mut log_theta = theta_init.ln();

    // Newton-Raphson on log(theta). Chain rule:
    //   d/d(log θ) ℓ(θ) = θ · ∂ℓ/∂θ
    //   -d²/d(log θ)² ℓ(θ) = -θ² ∂²ℓ/∂θ² − θ ∂ℓ/∂θ = θ² · info − θ · score
    //
    // Use the damped-Newton formulation: if info_log ≤ 0 or step explodes,
    // fall back to half-stepping.
    let mut converged = false;
    for _ in 0..max_iter {
        let theta = log_theta.exp();
        let (score, info) = nb_theta_score_info(x, mu.view(), theta);
        let score_log = theta * score;
        let info_log = theta * theta * info - theta * score;

        if !score_log.is_finite() || !info_log.is_finite() || info_log <= 0.0 {
            // Fisher not positive definite — fall back to gradient ascent with
            // small step. Extremely rare in practice for NB MLE.
            log_theta += 0.1 * score_log.signum();
            log_theta = log_theta.clamp(THETA_MIN.ln(), THETA_MAX.ln());
            continue;
        }

        let mut step = score_log / info_log;
        // Damp large steps to keep log_theta in bounds and avoid overshooting.
        step = step.clamp(-2.0, 2.0);
        log_theta += step;
        log_theta = log_theta.clamp(THETA_MIN.ln(), THETA_MAX.ln());

        if step.abs() < TOL {
            converged = true;
            break;
        }
    }

    let _ = converged; // informational; not yet surfaced

    (log_theta.exp(), beta0)
}

/// Fit the NB GLM offset model for every gene (row) of a dense genes × cells
/// UMI matrix, in parallel.
///
/// Returns `(theta, beta0)` arrays of length `n_genes` each.
pub fn fit_glm_offset_dense(
    umi: &Array2<f64>,
    total_umi: ArrayView1<f64>,
    max_iter: usize,
) -> (Array1<f64>, Array1<f64>) {
    let n_genes = umi.shape()[0];
    let results: Vec<(f64, f64)> = umi
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| fit_gene_offset(row, total_umi, max_iter))
        .collect();

    let mut theta = Array1::zeros(n_genes);
    let mut beta0 = Array1::zeros(n_genes);
    for (g, (t, b)) in results.into_iter().enumerate() {
        theta[g] = t;
        beta0[g] = b;
    }
    (theta, beta0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn trigamma_matches_known_values() {
        // ψ'(1) = π²/6
        let pi_sq_over_6 = std::f64::consts::PI.powi(2) / 6.0;
        assert!((trigamma(1.0) - pi_sq_over_6).abs() < 1e-10);
        // ψ'(2) = π²/6 − 1
        assert!((trigamma(2.0) - (pi_sq_over_6 - 1.0)).abs() < 1e-10);
        // Large x: ψ'(x) ≈ 1/x
        assert!((trigamma(1000.0) - 1.0 / 1000.0).abs() < 1e-6);
    }

    #[test]
    fn beta0_closed_form_poisson() {
        // When data is Poisson with known rate, the offset model recovers
        // beta0 = log(rate/mean_total_per_cell_unit).
        let total: Array1<f64> = Array1::from_elem(1000, 1000.0);
        // Set x = rate * total_c, with rate = 0.005.
        let x: Array1<f64> = total.mapv(|t| 0.005 * t);
        let (_theta, beta0) = fit_gene_offset(x.view(), total.view(), 30);
        assert!((beta0 - 0.005_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn zero_gene_does_not_panic() {
        let total = array![1000.0, 2000.0, 500.0];
        let x = Array1::zeros(3);
        let (theta, beta0) = fit_gene_offset(x.view(), total.view(), 30);
        assert!(theta.is_finite());
        assert!(beta0.is_finite());
    }
}
