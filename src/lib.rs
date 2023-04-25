use crate::ast::Expr;
use num_complex::Complex64;
use numpy::ndarray::{Array1, ArrayView1, ArrayViewD};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::error;

#[macro_use]
extern crate lalrpop_util;

lalrpop_mod!(formula_disp);
mod ast;

#[test]
fn basic_execution_test() {
    assert!(formula_disp::FormulaParser::new()
        .parse(" eps = 22")
        .is_ok());
    assert!(formula_disp::FormulaParser::new().parse("n = (22)").is_ok());
    assert!(formula_disp::FormulaParser::new()
        .parse("eps = (22)")
        .is_ok());
    let expr = formula_disp::FormulaParser::new()
        .parse("n = 22 * 44 + 66")
        .unwrap();
    assert_eq!(&format!("{:?}", expr), "n = ((22 * 44) + 66)");
    let expr = formula_disp::FormulaParser::new()
        .parse("eps = 3 * 22 ** 4")
        .unwrap();
    assert_eq!(&format!("{:?}", expr), "eps = (3 * (22 ** 4))");
    let expr = formula_disp::FormulaParser::new()
        .parse("eps = 3 * lbda ** 4")
        .unwrap();
    assert_eq!(&format!("{:?}", expr), "eps = (3 * (lbda ** 4))");
    let expr = formula_disp::FormulaParser::new()
        .parse("eps = sum[param]")
        .unwrap();
    assert_eq!(&format!("{:?}", expr), "eps = sum[r_param]");
    assert!(formula_disp::FormulaParser::new()
        .parse("n = ((((22))))")
        .is_ok());
    assert!(formula_disp::FormulaParser::new()
        .parse("n = sum[2 * 3] + sum[4*5]")
        .is_ok());
    assert!(formula_disp::FormulaParser::new()
        .parse("n = sum[sum [ 2 * lbda ] * 3] + sum[4*5]")
        .is_err());
    assert!(formula_disp::FormulaParser::new()
        .parse("n = ((((22))))")
        .is_ok());
    assert!(formula_disp::FormulaParser::new()
        .parse("eps = ((22)")
        .is_err());
    assert!(formula_disp::FormulaParser::new()
        .parse("something = ((22)")
        .is_err());
    assert!(formula_disp::FormulaParser::new().parse("(22)").is_err());
}

#[allow(unused)]
fn evaluate(
    ast: &Expr,
    x_axis_name: &str,
    x_axis_values: &ArrayView1<'_, f64>,
) -> Array1<Complex64> {
    return Array1::<Complex64>::zeros(x_axis_values.len());
}

fn parse<'a>(
    formula: &'a str,
    x_axis_name: &'a str,
    x_axis_values: ArrayViewD<'a, f64>,
) -> Result<Array1<Complex64>, Box<dyn error::Error + 'a>> {
    let size = x_axis_values.len();
    let x_axis_1d = x_axis_values.into_shape([size])?;

    let ast = formula_disp::FormulaParser::new().parse(&formula)?;

    Ok(evaluate(&ast, &x_axis_name, &x_axis_1d))
}

#[pymodule]
fn formula_dispersion(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "parse")]
    fn parse_py<'py>(
        py: Python<'py>,
        formula: &str,
        x_axis_name: &str,
        x_axis_values: PyReadonlyArrayDyn<f64>,
        // single_params: &PyDict,
        // rep_params: &PyDict
    ) -> PyResult<&'py PyArray1<Complex64>> {
        let x: numpy::ndarray::ArrayBase<
            numpy::ndarray::ViewRepr<&f64>,
            numpy::ndarray::Dim<numpy::ndarray::IxDynImpl>,
        > = x_axis_values.as_array();

        match parse(formula, x_axis_name, x) {
            Ok(arr) => return Ok(arr.into_pyarray(py)),
            Err(err) => return Err(PyErr::new::<PyTypeError, _>(err.to_string())),
        }
    }

    Ok(())
}
