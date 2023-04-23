use numpy::ndarray::ArrayViewD;
use numpy::PyReadonlyArrayDyn;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
// use pyo3::types::PyDict;
use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;

#[macro_use]
extern crate lalrpop_util;

lalrpop_mod!(formula_dispersion);

#[test]
fn basic_execution_test() {
    use numpy::ndarray::aview1;

    let x_axis_name = "lbda";
    let a1 = aview1(&[0., 1., 2., 3.]);
    assert!(formula_dispersion::ExprParser::new().parse(x_axis_name, &a1, "22").is_ok());
    assert!(formula_dispersion::ExprParser::new().parse(x_axis_name, &a1, "(22)").is_ok());
    assert!(
        formula_dispersion::ExprParser::new()
            .parse(x_axis_name, &a1, "2 * 10")
            .unwrap()
            == 20.
    );
    assert!(formula_dispersion::ExprParser::new()
        .parse(x_axis_name, &a1, "((((22))))")
        .is_ok());
    assert!(formula_dispersion::ExprParser::new()
        .parse(x_axis_name, &a1, "((22)")
        .is_err());
}

#[pymodule]
fn formula_parser(_py: Python, m: &PyModule) -> PyResult<()> {

    fn parse_formula_dispersion<T>(formula: &str, x_axis_name: &str, x_axis_values: ArrayViewD<'_, T>) -> PyResult<String> {
        let size = x_axis_values.len();
        match x_axis_values.into_shape([size]) {
            Ok(x_axis) => {
                let parsed =
                    formula_dispersion::ExprParser::new().parse(&x_axis_name, &x_axis, &formula);
                
                match parsed {
                    Ok(v) => return Ok(v.to_string()),
                    Err(err) => return Err(PyErr::new::<PyTypeError, _>(err.to_string())),
                }
            },
            Err(err) => return Err(PyErr::new::<PyTypeError, _>(err.to_string()))
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "parse_formula_dispersion")]
    fn parse_formula_dispersion_py<'py>(
        formula: &str,
        x_axis_name: &str,
        x_axis_values: PyReadonlyArrayDyn<f64>,
        // single_params: &PyDict,
        // rep_params: &PyDict
    ) -> PyResult<String> {
        let x: numpy::ndarray::ArrayBase<numpy::ndarray::ViewRepr<&f64>, numpy::ndarray::Dim<numpy::ndarray::IxDynImpl>> = x_axis_values.as_array();
        return parse_formula_dispersion(formula, x_axis_name, x);
    }

    Ok(())
}