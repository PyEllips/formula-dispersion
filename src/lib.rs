use crate::ast::Expr;
use lalrpop_util::lexer::Token;
use lalrpop_util::ParseError;
use num_complex::Complex64;
use numpy::ndarray::{Array1, ArrayViewD};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::collections::HashMap;
use std::error;

#[macro_use]
extern crate lalrpop_util;

lalrpop_mod!(formula_disp);
mod ast;

#[test]
fn basic_execution_test() {
    use crate::ast::Expr::*;
    use crate::ast::Opcode;
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
    assert_eq!(
        Index(Box::new(Op(
            Box::new(Op(
                Box::new(Number(22.)),
                Opcode::Mul,
                Box::new(Number(44.))
            )),
            Opcode::Add,
            Box::new(Number(66.)),
        ))),
        *expr
    );
    assert_eq!(&format!("{:?}", expr), "n = ((22.0 * 44.0) + 66.0)");
    let expr = formula_disp::FormulaParser::new()
        .parse("eps = 3 * 22 ** 4")
        .unwrap();
    assert_eq!(&format!("{:?}", expr), "eps = (3.0 * (22.0 ** 4.0))");
    let expr = formula_disp::FormulaParser::new()
        .parse("eps = 3 * lbda ** 4")
        .unwrap();
    assert_eq!(&format!("{:?}", expr), "eps = (3.0 * (lbda ** 4.0))");
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

fn parse_ast<'a>(formula: &'a str) -> Result<Box<Expr>, ParseError<usize, Token<'a>, &'a str>> {
    formula_disp::FormulaParser::new().parse(formula)
}

fn parse<'a>(
    formula: &'a str,
    x_axis_name: &'a str,
    x_axis_values: ArrayViewD<'a, f64>,
    single_params: &HashMap<&str, f64>,
    rep_params: &HashMap<&str, Vec<f64>>,
) -> Result<Array1<Complex64>, Box<dyn error::Error + 'a>> {
    let size = x_axis_values.len();
    let x_axis_1d = x_axis_values.into_shape([size])?;

    let ast = parse_ast(formula)?;
    ast.evaluate(&x_axis_name, &x_axis_1d, single_params, rep_params)
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
        single_params: &PyDict,
        rep_params: &PyDict,
    ) -> PyResult<&'py PyArray1<Complex64>> {
        let x: numpy::ndarray::ArrayBase<
            numpy::ndarray::ViewRepr<&f64>,
            numpy::ndarray::Dim<numpy::ndarray::IxDynImpl>,
        > = x_axis_values.as_array();

        let sparams: HashMap<&str, f64> = match single_params.extract() {
            Ok(hmap) => hmap,
            Err(err) => {
                return Err(PyErr::new::<PyTypeError, _>(format!(
                    "Error while parsing single parameters: {}",
                    err.to_string()
                )))
            }
        };
        let rparams: HashMap<&str, Vec<f64>> = match rep_params.extract() {
            Ok(hmap) => hmap,
            Err(err) => {
                return Err(PyErr::new::<PyTypeError, _>(format!(
                    "Error while parsing repeated parameters: {}",
                    err.to_string()
                )))
            }
        };

        match parse(formula, x_axis_name, x, &sparams, &rparams) {
            Ok(arr) => return Ok(arr.into_pyarray(py)),
            Err(err) => return Err(PyErr::new::<PyTypeError, _>(err.to_string())),
        }
    }

    Ok(())
}
