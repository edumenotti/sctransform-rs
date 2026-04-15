use pyo3::prelude::*;

#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
