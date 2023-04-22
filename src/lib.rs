use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;

#[macro_use]
extern crate lalrpop_util;

lalrpop_mod!(calculator1);

#[test]
fn calculator1() {
    let mut a = [1., 2., 3.];
    assert!(calculator1::ExprParser::new().parse(&mut a, "22").is_ok());
    assert!(calculator1::ExprParser::new().parse(&mut a, "(22)").is_ok());
    assert!(
        calculator1::ExprParser::new()
            .parse(&mut a, "2 * 10")
            .unwrap()
            == 20.
    );
    assert!(calculator1::ExprParser::new()
        .parse(&mut a, "((((22))))")
        .is_ok());
    assert!(calculator1::ExprParser::new()
        .parse(&mut a, "((22)")
        .is_err());
}

#[pyfunction]
pub fn parse_to_str(input: &str) -> PyResult<String> {
    let mut a = [1., 2., 3.];
    let parsed = calculator1::ExprParser::new().parse(&mut a, &input);
    match parsed {
        Ok(v) => return Ok(v.to_string()),
        Err(err) => return Err(PyErr::new::<PyTypeError, _>(err.to_string())),
    }
}

#[pymodule]
fn formula_parser(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_to_str, m)?)?;

    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    Ok(())
}
