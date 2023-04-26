use num_complex::Complex64;
use numpy::ndarray::{Array1, ArrayView1, Zip};
use std::collections::HashMap;
use std::fmt::{Debug, Error, Formatter};
use std::str::FromStr;

#[derive(PartialEq)]
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

impl Expr<'_> {
    #[allow(unused)]
    pub fn evaluate(
        &self,
        x_axis_name: &str,
        x_axis_values: &ArrayView1<'_, f64>,
        single_params: &HashMap<String, f64>,
        rep_params: &HashMap<String, Array1<f64>>,
    ) -> Array1<Complex64> {
        use Expr::*;
        return match *self {
            Number(num) => {
                Complex64::new(num, 0.) + Array1::<Complex64>::zeros(x_axis_values.len())
            }
            Op(ref l, op, ref r) => op.apply(
                l.evaluate(x_axis_name, x_axis_values, single_params, rep_params),
                r.evaluate(x_axis_name, x_axis_values, single_params, rep_params),
            ),
            _ => Array1::<Complex64>::zeros(x_axis_values.len()),
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
    pub fn apply(&self, left: Array1<Complex64>, right: Array1<Complex64>) -> Array1<Complex64> {
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

#[derive(Copy, Clone, PartialEq)]
pub enum Constant {
    I,
    Pi,
    Eps0,
    PlanckConstBar,
    PlanckConst,
    SpeedOfLight,
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
