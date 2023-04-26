use crate::ast::Expr;
use crate::ast::ExprParams;
use lalrpop_util::lexer::Token;
use lalrpop_util::ParseError;
use num_complex::Complex64;
use numpy::ndarray::Array1;
use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use numpy::{IntoPyArray, PyArray1};
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
    x_axis_values: &'a ArrayView1<'a, f64>,
    single_params: &'a HashMap<&str, f64>,
    rep_params: &'a HashMap<&str, Vec<f64>>,
) -> Result<Array1<Complex64>, Box<dyn error::Error + 'a>> {
    let ast = parse_ast(formula)?;
    ast.evaluate(&ExprParams {
        x_axis_name: &x_axis_name,
        x_axis_values: &x_axis_values,
        single_params,
        rep_params,
        is_in_sum: false,
    })
}

#[pymodule]
fn formula_dispersion(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "parse")]
    fn parse_py<'py>(
        py: Python<'py>,
        formula: &str,
        x_axis_name: &str,
        x_axis_values: PyReadonlyArray1<f64>,
        single_params: &PyDict,
        rep_params: &PyDict,
    ) -> PyResult<&'py PyArray1<Complex64>> {
        let x: numpy::ndarray::ArrayBase<numpy::ndarray::ViewRepr<&f64>, numpy::ndarray::Ix1> =
            x_axis_values.as_array();

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

        let result = match parse(formula, x_axis_name, &x, &sparams, &rparams) {
            Ok(arr) => Ok(arr.into_pyarray(py)),
            Err(err) => Err(PyErr::new::<PyTypeError, _>(err.to_string())),
        };
        result
    }

    Ok(())
}
