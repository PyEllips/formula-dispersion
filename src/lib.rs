use std::error;
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python}; 
use pyo3::exceptions::PyTypeError;
use numpy::{PyReadonlyArrayDyn, PyArray1, IntoPyArray};
use numpy::ndarray::{Array1, ArrayViewD};
use num_complex::Complex64;

#[macro_use]
extern crate lalrpop_util;

lalrpop_mod!(formula_dispersion);

#[test]
fn basic_execution_test() {
    use numpy::ndarray::{aview1, aview2};

    let x_axis_name = "lbda";
    let a1 = aview1(&[0., 1., 2., 3.]);
    let a2 = aview2(&[[0., 1., 2.], [4., 5., 6.]]);
    let size = a2.len();
    assert!(formula_dispersion::ExprParser::new().parse(x_axis_name, &a1, "22").is_ok());
    assert!(formula_dispersion::ExprParser::new().parse(x_axis_name, &a1, "(22)").is_ok());
    assert!(formula_dispersion::ExprParser::new().parse(x_axis_name, &a2.into_shape([size]).unwrap(), "(22)").is_ok());
    assert!(
        formula_dispersion::ExprParser::new()
            .parse(x_axis_name, &a1, "2 * 10")
            .unwrap()
            == aview1(&[
                Complex64::new(20., 0.),
                Complex64::new(20., 0.),
                Complex64::new(20., 0.),
                Complex64::new(20., 0.)
            ])
    );
    assert!(formula_dispersion::ExprParser::new()
        .parse(x_axis_name, &a1, "((((22))))")
        .is_ok());
    assert!(formula_dispersion::ExprParser::new()
        .parse(x_axis_name, &a1, "((22)")
        .is_err());
}

fn parse_formula_dispersion<'a>(
    formula: &'a str,
    x_axis_name: &'a str,
    x_axis_values: ArrayViewD<'a, f64>
) -> Result<Array1<Complex64>, Box<dyn error::Error + 'a>> {
    let size = x_axis_values.len();
    let x_axis_1d = x_axis_values.into_shape([size])?;
    
    Ok(formula_dispersion::ExprParser::new().parse(&x_axis_name, &x_axis_1d, &formula)?)
}

#[pymodule]
fn formula_parser(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "parse_formula_dispersion")]
    fn parse_formula_dispersion_py<'py>(
        py: Python<'py>,
        formula: &str,
        x_axis_name: &str,
        x_axis_values: PyReadonlyArrayDyn<f64>,
        // single_params: &PyDict,
        // rep_params: &PyDict
    ) -> PyResult<&'py PyArray1<Complex64>> {
        let x: numpy::ndarray::ArrayBase<numpy::ndarray::ViewRepr<&f64>, numpy::ndarray::Dim<numpy::ndarray::IxDynImpl>> = x_axis_values.as_array();
        match parse_formula_dispersion(formula, x_axis_name, x) {
            Ok(arr) => return Ok(arr.into_pyarray(py)),
            Err(err) => return Err(PyErr::new::<PyTypeError, _>(err.to_string()))
        }
    }

    Ok(())
}