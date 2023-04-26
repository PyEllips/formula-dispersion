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
pub struct MissingSingleParameter;

impl Display for MissingSingleParameter {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "A single parameter is missing")
    }
}

impl error::Error for MissingSingleParameter {}

impl Expr<'_> {
    #[allow(unused)]
    pub fn evaluate(
        &self,
        x_axis_name: &str,
        x_axis_values: &ArrayView1<'_, f64>,
        single_params: &HashMap<&str, f64>,
        rep_params: &HashMap<&str, Vec<f64>>,
    ) -> Result<Array1<Complex64>, Box<dyn error::Error>> {
        use Expr::*;
        return match *self {
            Number(num) => {
                Ok(Complex64::new(num, 0.) + Array1::<Complex64>::zeros(x_axis_values.len()))
            }
            Op(ref l, op, ref r) => Ok(op.reduce(
                l.evaluate(x_axis_name, x_axis_values, single_params, rep_params)?,
                r.evaluate(x_axis_name, x_axis_values, single_params, rep_params)?,
            )),
            Constant(c) => Ok(c.get() + Array1::<Complex64>::zeros(x_axis_values.len())),
            Func(func, ref expr) => Ok(func.apply(expr.evaluate(
                x_axis_name,
                x_axis_values,
                single_params,
                rep_params,
            )?)),
            Var(key) => {
                match key {
                    x if x == x_axis_name => {
                        Ok(Array1::<Complex64>::zeros(x_axis_values.len()) + x_axis_values)
                    }
                    _ => match single_params.get(key) {
                        Some(val) => Ok(Complex64::new(*val, 0.)
                            + Array1::<Complex64>::zeros(x_axis_values.len())),
                        None => Err(MissingSingleParameter.into()),
                    },
                }
            }
            Dielectric(ref expr) => {
                Ok(expr.evaluate(x_axis_name, x_axis_values, single_params, rep_params)?)
            }
            Index(ref expr) => Ok(expr
                .evaluate(x_axis_name, x_axis_values, single_params, rep_params)?
                .map(|x| x.powi(2))),

            // Missing:
            // KramersKronig(Box<Expr<'input>>),
            // Sum(Box<Expr<'input>>),
            // RepeatedVar(&'input str),
            _ => Err(NotImplementedError.into()),
        };
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
            Complex64::new(1., 0.)
        } else if self.re == 0. {
            Complex64::new(zero_val, 0.)
        } else {
            Complex64::new(0., 0.)
        }
    }
}

impl Func {
    pub fn apply(&self, expr: Array1<Complex64>) -> Array1<Complex64> {
        use Func::*;
        match *self {
            Sin => expr.map(|x| x.sin()),
            Cos => expr.map(|x| x.cos()),
            Tan => expr.map(|x| x.tan()),
            Sqrt => expr.map(|x| x.sqrt()),
            Ln => expr.map(|x| x.ln()),
            Log => expr.map(|x| x.log(10.)),
            Dawsn => expr.map(|x| x.dawson()),
            Heaviside => expr.map(|x| x.heaviside(0.)),
        }
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
            Pi => Complex64::new(PI, 0.),
            Eps0 => Complex64::new(physical_constants::VACUUM_ELECTRIC_PERMITTIVITY, 0.),
            PlanckConstBar => Complex64::new(physical_constants::PLANCK_CONSTANT / 2. / PI, 0.),
            PlanckConst => Complex64::new(physical_constants::PLANCK_CONSTANT, 0.),
            SpeedOfLight => Complex64::new(physical_constants::SPEED_OF_LIGHT_IN_VACUUM, 0.),
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
