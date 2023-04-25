use std::fmt::{Debug, Error, Formatter};

pub enum Expr<'input> {
    Number(i32),
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

#[derive(Copy, Clone)]
pub enum Opcode {
    Mul,
    Div,
    Add,
    Sub,
    Pow,
}

#[derive(Copy, Clone)]
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

#[derive(Copy, Clone)]
pub enum Constant {
    I,
    Pi,
    Eps0,
    PlanckConstBar,
    PlanckConst,
    SpeedOfLight,
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
