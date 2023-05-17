use errorfunctions::ComplexErrorFunctions;
use num_complex::Complex64;
use numpy::ndarray::{Array1, ArrayView1, Zip};
use physical_constants;
use std::collections::HashMap;
use std::error;
use std::f64::consts::PI;
use std::fmt;
use std::fmt::{Debug, Display, Error, Formatter};
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

#[derive(Clone, PartialEq)]
pub enum Expr<'input> {
    Number(f64),
    Op(Box<Expr<'input>>, Opcode, Box<Expr<'input>>),
    Index(Box<Expr<'input>>),
    Dielectric(Box<Expr<'input>>),
    KramersKronig(Box<Expr<'input>>),
    Constant(Constant),
    Sum(Box<Expr<'input>>),
    Func(Func, Box<Expr<'input>>),
    Var(&'input str),
    RepeatedVar(&'input str),
}

#[derive(Debug, Clone)]
pub struct NotImplementedError;

impl Display for NotImplementedError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?} not implemented", self)
    }
}

impl error::Error for NotImplementedError {}

#[derive(Debug, Clone)]
pub struct MissingParameter {
    message: String,
}

impl MissingParameter {
    fn new(parameter_name: &str) -> MissingParameter {
        MissingParameter {
            message: format!("The parameter {} is missing", parameter_name).to_string(),
        }
    }
}

impl Display for MissingParameter {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl error::Error for MissingParameter {
    fn description(&self) -> &str {
        &self.message
    }
}

pub struct ExprParams<'a> {
    pub x_axis_name: &'a str,
    pub x_axis_values: &'a ArrayView1<'a, f64>,
    pub single_params: &'a HashMap<&'a str, f64>,
    pub rep_params: &'a HashMap<&'a str, Vec<f64>>,
    pub sum_params: Option<HashMap<&'a str, f64>>,
}

pub enum EvaluateResult {
    Array(Array1<Complex64>),
    Number(Complex64),
}

impl EvaluateResult {
    fn power(self, other: EvaluateResult) -> EvaluateResult {
        use EvaluateResult::*;
        match (self, other) {
            (Number(b), Number(exp)) => EvaluateResult::Number(b.powc(exp)),
            (Number(b), Array(exp)) => EvaluateResult::Array(exp.map(|x| b.powc(*x))),
            (Array(b), Number(exp)) => EvaluateResult::Array(b.map(|x| x.powc(exp))),
            (Array(b), Array(exp)) => EvaluateResult::Array(
                Zip::from(&b)
                    .and(&exp)
                    .map_collect(|base, &exp| (*base).powc(exp)),
            ),
        }
    }

    fn sin(self) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.sin()),
            Array(arr) => Array(arr.map(|x| x.sin())),
        }
    }

    fn cos(self) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.cos()),
            Array(arr) => Array(arr.map(|x| x.cos())),
        }
    }

    fn tan(self) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.tan()),
            Array(arr) => Array(arr.map(|x| x.tan())),
        }
    }

    fn sqrt(self) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.sqrt()),
            Array(arr) => Array(arr.map(|x| x.sqrt())),
        }
    }

    fn ln(self) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.ln()),
            Array(arr) => Array(arr.map(|x| x.ln())),
        }
    }

    fn log(self, base: f64) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.log(base)),
            Array(arr) => Array(arr.map(|x| x.log(base))),
        }
    }

    fn dawson(self) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.dawson()),
            Array(arr) => Array(arr.map(|x| x.dawson())),
        }
    }

    fn heaviside(self, zero_val: f64) -> EvaluateResult {
        use EvaluateResult::*;
        match self {
            Number(num) => Number(num.heaviside(zero_val)),
            Array(arr) => Array(arr.map(|x| x.heaviside(zero_val))),
        }
    }
}

impl Mul for EvaluateResult {
    type Output = EvaluateResult;

    fn mul(self, other: EvaluateResult) -> EvaluateResult {
        use EvaluateResult::*;
        match (self, other) {
            (Number(x), Number(y)) => EvaluateResult::Number(x * y),
            (Number(x), Array(y)) => EvaluateResult::Array(x * y),
            (Array(x), Number(y)) => EvaluateResult::Array(x * y),
            (Array(x), Array(y)) => EvaluateResult::Array(x * y),
        }
    }
}

impl Div for EvaluateResult {
    type Output = EvaluateResult;

    fn div(self, other: EvaluateResult) -> EvaluateResult {
        use EvaluateResult::*;
        match (self, other) {
            (Number(x), Number(y)) => EvaluateResult::Number(x / y),
            (Number(x), Array(y)) => EvaluateResult::Array(x / y),
            (Array(x), Number(y)) => EvaluateResult::Array(x / y),
            (Array(x), Array(y)) => EvaluateResult::Array(x / y),
        }
    }
}

impl Add for EvaluateResult {
    type Output = EvaluateResult;

    fn add(self, other: EvaluateResult) -> EvaluateResult {
        use EvaluateResult::*;
        match (self, other) {
            (Number(x), Number(y)) => EvaluateResult::Number(x + y),
            (Number(x), Array(y)) => EvaluateResult::Array(x + y),
            (Array(x), Number(y)) => EvaluateResult::Array(x + y),
            (Array(x), Array(y)) => EvaluateResult::Array(x + y),
        }
    }
}

impl Sub for EvaluateResult {
    type Output = EvaluateResult;

    fn sub(self, other: EvaluateResult) -> EvaluateResult {
        use EvaluateResult::*;
        match (self, other) {
            (Number(x), Number(y)) => EvaluateResult::Number(x - y),
            (Number(x), Array(y)) => EvaluateResult::Array(x - y),
            (Array(x), Number(y)) => EvaluateResult::Array(x - y),
            (Array(x), Array(y)) => EvaluateResult::Array(x - y),
        }
    }
}

impl Expr<'_> {
    pub fn get_representation<'a>(self) -> Result<&'a str, Box<dyn error::Error>> {
        use Expr::*;
        match self {
            Dielectric(_) => Ok("eps"),
            Index(_) => Ok("n"),
            _ => Err("Not a valid expression".into()),
        }
    }
    pub fn evaluate<'a>(
        &self,
        params: &mut ExprParams<'a>,
    ) -> Result<EvaluateResult, Box<dyn error::Error>> {
        use Expr::*;
        match *self {
            Number(num) => Ok(EvaluateResult::Number(Complex64::from(num))),
            Op(ref l, op, ref r) => Ok(op.reduce(l.evaluate(params)?, r.evaluate(params)?)),
            Constant(c) => Ok(EvaluateResult::Number(c.get())),
            Func(func, ref expr) => Ok(func.evaluate(expr.evaluate(params)?)),
            Var(key) => match key {
                x if x == params.x_axis_name => Ok(EvaluateResult::Array(
                    params.x_axis_values.mapv(|elem| Complex64::from(elem)),
                )),
                _ => match params.single_params.get(key) {
                    Some(val) => Ok(EvaluateResult::Number(Complex64::new(*val, 0.))),
                    None => Err(MissingParameter::new(key).into()),
                },
            },
            RepeatedVar(key) => match key {
                x if x == params.x_axis_name => Ok(EvaluateResult::Array(
                    params.x_axis_values.mapv(|elem| Complex64::from(elem)),
                )),
                x if params.single_params.contains_key(x) => Ok(EvaluateResult::Number(
                    Complex64::from(params.single_params.get(key).unwrap()),
                )),
                x if params.sum_params.as_ref().unwrap().contains_key(x) => {
                    Ok(EvaluateResult::Number(Complex64::from(
                        params.sum_params.as_ref().unwrap().get(key).unwrap(),
                    )))
                }
                _ => Err(MissingParameter::new(key).into()),
            },
            Dielectric(ref expr) => expr.evaluate(params),
            Index(ref expr) => expr.evaluate(params),
            Sum(ref expr) => {
                let mut params_vec = Vec::new();
                for (key, val) in params.rep_params.iter() {
                    for (i, param) in val.iter().enumerate() {
                        if params_vec.len() <= i {
                            params_vec.push(HashMap::new())
                        }
                        params_vec[i].insert(*key, *param);
                    }
                }

                let mut result = EvaluateResult::Number(Complex64::from(0.));
                for p in params_vec.iter() {
                    params.sum_params = Some(p.clone());
                    result = result + expr.evaluate(params)?;
                }
                Ok(result)
            }
            // Missing:
            // KramersKronig(Box<Expr<'input>>),
            _ => Err(NotImplementedError.into()),
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Opcode {
    Mul,
    Div,
    Add,
    Sub,
    Pow,
}

impl Opcode {
    pub fn reduce(&self, left: EvaluateResult, right: EvaluateResult) -> EvaluateResult {
        use Opcode::*;
        match *self {
            Mul => left * right,
            Div => left / right,
            Add => left + right,
            Sub => left - right,
            Pow => left.power(right),
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Sqrt,
    Dawsn,
    Ln,
    Log,
    Heaviside,
}

trait Heaviside {
    fn heaviside(&self, zero_val: f64) -> Complex64;
}

impl Heaviside for Complex64 {
    fn heaviside(&self, zero_val: f64) -> Complex64 {
        if self.re > 0. {
            Complex64::from(1.)
        } else if self.re == 0. {
            Complex64::from(zero_val)
        } else {
            Complex64::from(0.)
        }
    }
}

trait Evaluate<T, G> {
    fn evaluate(&self, expr: T) -> G;
}

impl Evaluate<Array1<Complex64>, Array1<Complex64>> for Func {
    fn evaluate(&self, expr: Array1<Complex64>) -> Array1<Complex64> {
        expr.map(|x| self.evaluate(*x))
    }
}

impl Evaluate<EvaluateResult, EvaluateResult> for Func {
    fn evaluate(&self, expr: EvaluateResult) -> EvaluateResult {
        use Func::*;
        match *self {
            Sin => expr.sin(),
            Cos => expr.cos(),
            Tan => expr.tan(),
            Sqrt => expr.sqrt(),
            Ln => expr.ln(),
            Log => expr.log(10.),
            Dawsn => expr.dawson(),
            Heaviside => expr.heaviside(0.),
        }
    }
}

impl Evaluate<Complex64, Complex64> for Func {
    fn evaluate(&self, x: Complex64) -> Complex64 {
        use Func::*;
        match *self {
            Sin => x.sin(),
            Cos => x.cos(),
            Tan => x.tan(),
            Sqrt => x.sqrt(),
            Ln => x.ln(),
            Log => x.log(10.),
            Dawsn => x.dawson(),
            Heaviside => x.heaviside(0.),
        }
    }
}

impl Evaluate<Complex64, Complex64> for Constant {
    fn evaluate(&self, _: Complex64) -> Complex64 {
        self.get()
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Constant {
    I,
    Pi,
    Eps0,
    PlanckConstBar,
    PlanckConst,
    SpeedOfLight,
}

impl Constant {
    pub fn get(&self) -> Complex64 {
        use Constant::*;
        match *self {
            I => Complex64::new(0., 1.),
            Pi => Complex64::from(PI),
            Eps0 => Complex64::from(physical_constants::VACUUM_ELECTRIC_PERMITTIVITY),
            PlanckConstBar => Complex64::from(physical_constants::PLANCK_CONSTANT / 2. / PI),
            PlanckConst => Complex64::from(physical_constants::PLANCK_CONSTANT),
            SpeedOfLight => Complex64::from(physical_constants::SPEED_OF_LIGHT_IN_VACUUM),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseConstantError;

impl FromStr for Constant {
    type Err = ParseConstantError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1j" => Ok(Self::I),
            "pi" => Ok(Self::Pi),
            "eps_0" => Ok(Self::Eps0),
            "hbar" => Ok(Self::PlanckConstBar),
            "h" => Ok(Self::PlanckConst),
            "c" => Ok(Self::SpeedOfLight),
            _ => Err(ParseConstantError),
        }
    }
}

impl<'input> Debug for Expr<'input> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Expr::*;
        match *self {
            Number(n) => write!(fmt, "{:?}", n),
            Op(ref l, op, ref r) => write!(fmt, "({:?} {:?} {:?})", l, op, r),
            Constant(c) => write!(fmt, "{:?}", c),
            Index(ref expr) => write!(fmt, "n = {:?}", expr),
            Dielectric(ref expr) => write!(fmt, "eps = {:?}", expr),
            KramersKronig(ref expr) => write!(fmt, "<kkr> + 1j * {:?}", expr),
            Sum(ref expr) => write!(fmt, "sum[{:?}]", expr),
            Func(func, ref expr) => write!(fmt, "{:?}({:?})", func, expr),
            Var(name) => write!(fmt, "{}", name),
            RepeatedVar(name) => write!(fmt, "r_{}", name),
        }
    }
}

impl Debug for Opcode {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Opcode::*;
        match *self {
            Mul => write!(fmt, "*"),
            Div => write!(fmt, "/"),
            Add => write!(fmt, "+"),
            Sub => write!(fmt, "-"),
            Pow => write!(fmt, "**"),
        }
    }
}

impl Debug for Func {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Func::*;
        match *self {
            Sin => write!(fmt, "sin"),
            Cos => write!(fmt, "cos"),
            Tan => write!(fmt, "tan"),
            Sqrt => write!(fmt, "sqrt"),
            Dawsn => write!(fmt, "dawsn"),
            Ln => write!(fmt, "ln"),
            Log => write!(fmt, "log"),
            Heaviside => write!(fmt, "heaviside"),
        }
    }
}

impl Debug for Constant {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Constant::*;
        match *self {
            I => write!(fmt, "1j"),
            Pi => write!(fmt, "pi"),
            Eps0 => write!(fmt, "eps_0"),
            PlanckConstBar => write!(fmt, "hbar"),
            PlanckConst => write!(fmt, "h"),
            SpeedOfLight => write!(fmt, "c"),
        }
    }
}
