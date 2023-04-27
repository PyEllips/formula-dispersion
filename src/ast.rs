use errorfunctions::ComplexErrorFunctions;
use num_complex::Complex64;
use numpy::ndarray::{Array1, ArrayView1, Zip};
use physical_constants;
use std::collections::HashMap;
use std::error;
use std::f64::consts::PI;
use std::fmt;
use std::fmt::{Debug, Display, Error, Formatter};
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
    pub sum_params: &'a Option<HashMap<&'a str, f64>>,
}

// type Elem = Box<dyn Mul<Array1<Complex64>, Array1<Complex64>>>;

impl Expr<'_> {
    pub fn evaluate<'a>(
        &self,
        params: &ExprParams<'a>,
    ) -> Result<Array1<Complex64>, Box<dyn error::Error>> {
        use Expr::*;
        match *self {
            Number(num) => Ok(Array1::from_elem(
                params.x_axis_values.len(),
                Complex64::from(num),
            )),
            Op(ref l, op, ref r) => Ok(op.reduce(l.evaluate(params)?, r.evaluate(params)?)),
            Constant(c) => Ok(Array1::from_elem(params.x_axis_values.len(), c.get())),
            Func(func, ref expr) => Ok(func.evaluate(expr.evaluate(params)?)),
            Var(key) => match key {
                x if x == params.x_axis_name => {
                    Ok(params.x_axis_values.mapv(|elem| Complex64::from(elem)))
                }
                _ => match params.single_params.get(key) {
                    Some(val) => Ok(Array1::from_elem(
                        params.x_axis_values.len(),
                        Complex64::new(*val, 0.),
                    )),
                    None => Err(MissingParameter::new(key).into()),
                },
            },
            // RepeatedVar(key) => match key {
            //     x_axis if x_axis == params.x_axis_name => {
            //         Ok(params.x_axis_values.mapv(|elem| Complex64::from(elem)))
            //     }
            //     single_param if params.single_params.contains_key(single_param) => {
            //         Ok(Array1::from_elem(
            //             params.x_axis_values.len(),
            //             Complex64::from(params.single_params.get(single_param).unwrap()),
            //         ))
            //     }
            // },
            Dielectric(ref expr) => expr.evaluate(params),
            Index(ref expr) => Ok(expr.evaluate(params)?.map(|x| x.powi(2))),
            // Sum(expr) => {
            //     for param in params.rep_params.keys() {
            //         for elem in params.rep_params.get(param).unwrap() {}
            //     }
            //     Ok()
            // }
            // Sum =>
            // Build [{key, key2, key3}, {key, key2, key3}] structure
            // HashMap<str, Vec<f64>> -> Vec<HashMap<String, f64>>
            // Iterate over the array
            // Missing:
            // KramersKronig(Box<Expr<'input>>),
            // Sum(Box<Expr<'input>>),
            // RepeatedVar(&'input str),
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
    pub fn reduce(&self, left: Array1<Complex64>, right: Array1<Complex64>) -> Array1<Complex64> {
        use Opcode::*;
        match *self {
            Mul => left * right,
            Div => left / right,
            Add => left + right,
            Sub => left - right,
            Pow => Zip::from(&left)
                .and(&right)
                .map_collect(|base, &exp| (*base).powc(exp)),
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

type ArrTuple = (Array1<Complex64>, Array1<Complex64>);
type ComplexTuple = (Complex64, Complex64);
type ArrComplexTuple = (Array1<Complex64>, Complex64);
type ComplexArrTuple = (Complex64, Array1<Complex64>);

impl Evaluate<ArrTuple, Array1<Complex64>> for Opcode {
    fn evaluate(&self, expr: ArrTuple) -> Array1<Complex64> {
        expr.0
    }
}

impl Evaluate<ComplexTuple, Complex64> for Opcode {
    fn evaluate(&self, expr: ComplexTuple) -> Complex64 {
        expr.0
    }
}

impl Evaluate<ArrComplexTuple, Array1<Complex64>> for Opcode {
    fn evaluate(&self, expr: ArrComplexTuple) -> Array1<Complex64> {
        expr.0
    }
}

impl Evaluate<ComplexArrTuple, Array1<Complex64>> for Opcode {
    fn evaluate(&self, expr: ComplexArrTuple) -> Array1<Complex64> {
        expr.1
    }
}

impl Evaluate<Array1<Complex64>, Array1<Complex64>> for Func {
    fn evaluate(&self, expr: Array1<Complex64>) -> Array1<Complex64> {
        expr.map(|x| self.evaluate(*x))
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

impl Evaluate<Option<Complex64>, Complex64> for Constant {
    fn evaluate(&self, _: Option<Complex64>) -> Complex64 {
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
