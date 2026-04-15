// PyO3 0.22's #[pyfunction] macro expands to code that triggers
// clippy::useless_conversion on the return-type wrapping. Suppress it
// crate-wide rather than per-function — it never indicates a real issue here.
#![allow(clippy::useless_conversion)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod residuals;

#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Compute Pearson residuals for a dense genes × cells UMI matrix.
///
/// Parameters
/// ----------
/// umi : np.ndarray[float64, shape=(n_genes, n_cells)]
///     UMI count matrix. Integer counts should be passed as float64.
/// theta, beta0, beta1 : np.ndarray[float64, shape=(n_genes,)]
///     Per-gene NB GLM parameters.
/// total_umi : np.ndarray[float64, shape=(n_cells,)]
///     Per-cell total UMI (sum of the cell's column of `umi`).
/// clip : bool, default True
///     If True, clip residuals to ±sqrt(n_cells / 30) (sctransform v2 default).
///
/// Returns
/// -------
/// np.ndarray[float64, shape=(n_genes, n_cells)]
///     Dense Pearson residuals.
#[pyfunction]
#[pyo3(signature = (umi, theta, beta0, beta1, total_umi, clip=true))]
#[allow(clippy::too_many_arguments)]
fn compute_residuals<'py>(
    py: Python<'py>,
    umi: PyReadonlyArray2<'py, f64>,
    theta: PyReadonlyArray1<'py, f64>,
    beta0: PyReadonlyArray1<'py, f64>,
    beta1: PyReadonlyArray1<'py, f64>,
    total_umi: PyReadonlyArray1<'py, f64>,
    clip: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let umi = umi.as_array();
    let theta = theta.as_array();
    let beta0 = beta0.as_array();
    let beta1 = beta1.as_array();
    let total_umi = total_umi.as_array();

    let (n_genes, n_cells) = umi.dim();

    if theta.len() != n_genes {
        return Err(PyValueError::new_err(format!(
            "theta has length {}, expected n_genes = {}",
            theta.len(),
            n_genes
        )));
    }
    if beta0.len() != n_genes {
        return Err(PyValueError::new_err(format!(
            "beta0 has length {}, expected n_genes = {}",
            beta0.len(),
            n_genes
        )));
    }
    if beta1.len() != n_genes {
        return Err(PyValueError::new_err(format!(
            "beta1 has length {}, expected n_genes = {}",
            beta1.len(),
            n_genes
        )));
    }
    if total_umi.len() != n_cells {
        return Err(PyValueError::new_err(format!(
            "total_umi has length {}, expected n_cells = {}",
            total_umi.len(),
            n_cells
        )));
    }

    // Ensure the umi matrix is contiguous & owned so rayon can freely borrow rows.
    let umi_owned = umi.to_owned();

    let out = py.allow_threads(|| {
        residuals::compute_residuals_dense(&umi_owned, theta, beta0, beta1, total_umi, clip)
    });

    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residuals, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
