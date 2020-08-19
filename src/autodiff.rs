use crate::num::{One, ScalarFloat, Zero};
use num::{Float, Num, NumCast, ToPrimitive};
use std::cmp::Ordering;

static mut DEFAULT_TAPE: Tape = Tape::new();
fn tape_mut() -> &'static mut Tape { unsafe { &mut DEFAULT_TAPE } }
fn tape() -> &'static Tape { unsafe { &DEFAULT_TAPE } }

// TODO think about some way to optimize this for constants such that they don't need to be
// added to the graph(possibly have an invalid value or make the usize optional?)
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WeightedEdge(ScalarFloat, NodeIdx);

#[derive(Debug, PartialEq, Clone, Copy)]
enum Parents {
  None,
  One(WeightedEdge),
  Two(WeightedEdge, WeightedEdge),
}

/// A tape for tracking differentiated variables. There is one global singleton being tracked,
/// and can be replaced by calling reset-tape.
pub struct Tape {
  nodes: Vec<Node>,
  grads: Vec<ScalarFloat>,
}

/// Marker for indices into a tape. u32 to save space.
pub type NodeIdx = u32;

impl Tape {
  pub const fn new() -> Self {
    Self {
      nodes: vec![],
      grads: vec![],
    }
  }
  fn push_node(&mut self, n: Node) -> NodeIdx {
    let idx = self.nodes.len();
    self.nodes.push(n);
    self.grads.push(0.0);
    idx as u32
  }
  /// Computes the gradient going backwards from this variable to all previous variables
  fn backwards(&mut self, start: NodeIdx) {
    assert!(start < self.nodes.len() as u32);
    self.grads[start as usize] = 1.0;
    // because we only add items in order, we can propogate backwards
    // This might be slower than using a stack if there are a lot of irrelevant
    // calculations.
    for i in (0u32..=start).rev() {
      let i = i as usize;
      match self.nodes[i].parents {
        Parents::None => (),
        Parents::One(WeightedEdge(w, n_i)) => self.grads[n_i as usize] += w * self.grads[i],
        Parents::Two(WeightedEdge(w0, i0), WeightedEdge(w1, i1)) => {
          self.grads[i0 as usize] += w0 * self.grads[i];
          self.grads[i1 as usize] += w1 * self.grads[i];
        },
      }
    }
  }
  /// Clear the gradients of the tape.
  pub fn clear_grads(&mut self) { self.grads.iter_mut().for_each(|v| *v = 0.0) }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Node {
  parents: Parents,
}

impl Node {
  pub const fn new_root() -> Self {
    Self {
      parents: Parents::None,
    }
  }
  pub const fn new_unary(w_e: WeightedEdge) -> Self {
    Self {
      parents: Parents::One(w_e),
    }
  }
  pub const fn new_binary(e_0: WeightedEdge, e_1: WeightedEdge) -> Self {
    Self {
      parents: Parents::Two(e_0, e_1),
    }
  }
}

/// A tracked variable which can be differentiated.
#[derive(Debug, Copy, Clone)]
pub struct Var {
  v: ScalarFloat,
  idx: u32,
}

impl Var {
  /// Creates an instance of a variable
  // allocates a node on the tape and stores the idx
  pub fn new(v: ScalarFloat) -> Self { Self::new_w_idx(v, tape_mut().push_node(Node::new_root())) }
  /// Creates a new node with a given value and index
  const fn new_w_idx(v: ScalarFloat, idx: u32) -> Self { Self { v, idx } }
  fn create_unary(&self, w: ScalarFloat) -> u32 {
    tape_mut().push_node(Node::new_unary(WeightedEdge(w, self.idx)))
  }
  fn create_binary(&self, w: ScalarFloat, w_o: ScalarFloat, o_idx: u32) -> u32 {
    tape_mut().push_node(Node::new_binary(
      WeightedEdge(w, self.idx),
      WeightedEdge(w_o, o_idx),
    ))
  }
  /// Computes the relationship of all variables used to this variable.
  pub fn backwards(&self) { tape_mut().backwards(self.idx) }

  /// Returns the gradient of this variable w.r.t the variable that had backward most recently
  /// called on it.
  pub fn grad_wrt(&self) -> ScalarFloat { tape().grads[self.idx as usize] }

  /// Updates this variable with the gradient
  pub fn update(&mut self, update: impl Fn(ScalarFloat, ScalarFloat) -> ScalarFloat) {
    self.v = update(self.v, self.grad_wrt());
  }
}

/// Implements a unary operation with a gradient
macro_rules! impl_unary_grad {
  ($func: ident, grad($v: ident) = $v_grad: expr) => {
    fn $func($v) -> Self { Self::new_w_idx($v.v.$func(), $v.create_unary($v_grad)) }
  };
  ($func: ident, undefined grad($v: ident)) => {
    fn $func($v) -> Self { Self::new($v.v.$func()) }
  };
}

macro_rules! impl_bool {
  ($($func: ident;)*) => {
    $(fn $func(self) -> bool { self.v.$func() })+
  }
}

macro_rules! impl_value {
  ($( $func: ident$(,)* )*) => {
    $( fn $func() -> Self { Self::new(ScalarFloat::$func()) } )*
  };
}

macro_rules! impl_binary_grad {
  ($func: ident, grad($v: ident, $o: ident) = $grads: expr) => {
    fn $func($v, $o: Self) -> Self {
      let (self_grad, o_grad) = $grads;
      Self::new_w_idx($v.v.$func($o.v), $v.create_binary(self_grad, o_grad, $o.idx))
    }
  };
}

macro_rules! sqr {
  ($v: expr) => {
    $v * $v
  };
}
impl Num for Var {
  type FromStrRadixErr = <ScalarFloat as Num>::FromStrRadixErr;
  fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
    let v = ScalarFloat::from_str_radix(s, radix)?;
    Ok(Self::new(v))
  }
}

impl Float for Var {
  impl_value!(
    nan,
    infinity,
    neg_infinity,
    neg_zero,
    min_value,
    max_value,
    min_positive_value
  );

  impl_unary_grad!(ln, grad(self) = self.v.recip());
  impl_unary_grad!(log2, grad(self) = (self.v * 2f32.ln()).recip());
  impl_unary_grad!(log10, grad(self) = (self.v * 10f32.ln()).recip());

  fn classify(self) -> std::num::FpCategory { self.v.classify() }
  fn integer_decode(self) -> (u64, i16, i8) { self.v.integer_decode() }
  // I could fuse this, but would require adding more complicated edges(3 parents)
  fn mul_add(self, mul: Self, add: Self) -> Self { (self * mul) + add }

  /// Taking the log w.r.t. an arbitrary base is currently unsupported
  /// So it assumes that base is fixed.
  fn log(self, base: Self) -> Self {
    Self::new_w_idx(
      self.v.log(base.v),
      self.create_unary((base.v.ln() * self.v).recip()),
    )
  }
  impl_unary_grad!(ln_1p, grad(self) = self.v.recip() + 1.0);

  impl_unary_grad!(exp, grad(self) = self.v.exp());
  impl_unary_grad!(exp2, grad(self) = 2f32.ln() * self.v.exp2());
  impl_unary_grad!(exp_m1, grad(self) = self.v.exp());
  impl_unary_grad!(sqrt, grad(self) = 0.5 * self.v.powf(-0.5));
  impl_unary_grad!(cbrt, grad(self) = self.v.powf(-2.0 / 3.0) / 3.0);
  impl_unary_grad!(abs, grad(self) = self.v.signum());

  fn abs_sub(self, o: Self) -> Self {
    if self.v <= o.v {
      o - self
    } else {
      self - o
    }
  }

  impl_unary_grad!(sin, grad(self) = self.v.cos());
  impl_unary_grad!(asin, grad(self) = (1.0 - sqr!(self.v)).recip());
  impl_unary_grad!(cos, grad(self) = -self.v.sin());
  impl_unary_grad!(acos, grad(self) = -(1.0 - sqr!(self.v)).recip());
  impl_unary_grad!(tan, grad(self) = sqr!(self.v.cos()).recip());
  impl_unary_grad!(atan, grad(self) = (1.0 + sqr!(self.v)).recip());

  fn sin_cos(self) -> (Self, Self) {
    let (sin, cos) = self.v.sin_cos();
    (
      Self::new_w_idx(sin, self.create_unary(cos)),
      Self::new_w_idx(cos, self.create_unary(-sin)),
    )
  }
  impl_unary_grad!(tanh, grad(self) = sqr!(self.v.cosh()).recip());
  impl_unary_grad!(sinh, grad(self) = self.v.cosh());
  impl_unary_grad!(cosh, grad(self) = self.v.sinh());
  // TODO compress this to not make intermediates
  fn hypot(self, o: Self) -> Self { (self.powi(2) + o.powi(2)).sqrt() }

  impl_binary_grad!(
    atan2,
    grad(self, o) = {
      let i = (sqr!(self.v) + sqr!(o.v)).recip();
      (i * o.v, -i * self.v)
    }
  );

  impl_unary_grad!(asinh, grad(self) = (sqr!(self.v) + 1.0).sqrt().recip());
  impl_unary_grad!(acosh, grad(self) = (sqr!(self.v) - 1.0).sqrt().recip());
  impl_unary_grad!(atanh, grad(self) = (1.0 - sqr!(self.v)).recip());

  impl_unary_grad!(recip, grad(self) = -sqr!(self.v).recip());

  impl_unary_grad!(floor, undefined grad(self));
  impl_unary_grad!(ceil, undefined grad(self));
  impl_unary_grad!(round, undefined grad(self));
  impl_unary_grad!(signum, undefined grad(self));
  impl_unary_grad!(trunc, undefined grad(self));
  impl_unary_grad!(fract, grad(self) = 1.0);

  // boolean check functions
  impl_bool!(
    is_finite;
    is_infinite;
    is_nan;
    is_sign_positive;
    is_sign_negative;
    is_normal;
  );

  // manually implemented to ensure efficiency
  /// Treats exp as a constant
  fn powf(self, exp: Self) -> Self {
    // compute this value for efficiency rather than 2 separate computations
    let val = self.v.powf(exp.v - 1.0);
    Self::new_w_idx(val * self.v, self.create_unary(val * exp.v))
  }

  fn powi(self, exp: i32) -> Self {
    let val = self.v.powi(exp - 1);
    Self::new_w_idx(val * self.v, self.create_unary(val * exp as ScalarFloat))
  }

  impl_binary_grad!(
    max,
    grad(self, o) = {
      assert!(!self.v.is_nan(), "NOT IMPLEMENTED NaN max");
      assert!(!o.v.is_nan(), "NOT IMPLEMENTED NaN max");
      match self.v.partial_cmp(&o.v) {
        None => todo!(),
        // technically this should (1.0, 1.0), but halving it keeps the amount of grads eql
        Some(Ordering::Equal) => (0.5, 0.5),
        Some(Ordering::Greater) => (1.0, 0.0),
        Some(Ordering::Less) => (0.0, 1.0),
      }
    }
  );

  impl_binary_grad!(
    min,
    grad(self, o) = {
      assert!(!self.v.is_nan(), "NOT IMPLEMENTED NaN max");
      assert!(!o.v.is_nan(), "NOT IMPLEMENTED NaN max");
      match self.v.partial_cmp(&o.v) {
        None => todo!(),
        // technically this should (1.0, 1.0), but halving it keeps the amount of grads eql
        Some(Ordering::Equal) => (0.5, 0.5),
        Some(Ordering::Greater) => (0.0, 1.0),
        Some(Ordering::Less) => (1.0, 0.0),
      }
    }
  );
}

macro_rules! impl_bin_op_grad {
  ($type: ty, $func: ident, $op: tt, grad($l: ident) = $l_grad: expr,
                                     grad($r: ident) = $r_grad: expr) => {
    impl $type for Var {
      type Output = Var;
      fn $func($l, $r: Var) -> Self::Output {
        let new_idx = $l.create_binary($l_grad, $r_grad, $r.idx);
        Self::new_w_idx($l.v $op $r.v, new_idx)
      }
    }
  }
}

use std::ops::{Add, Div, Mul, Rem, Sub};
impl_bin_op_grad!(Add, add, +,
                  grad(self) = 1.0,
                  grad(rhs) = 1.0);
impl_bin_op_grad!(Sub, sub, -,
                  grad(self) = 1.0,
                  grad(rhs) = -1.0);
impl_bin_op_grad!(Mul, mul, *,
                  grad(self) = rhs.v,
                  grad(rhs) = self.v);
impl_bin_op_grad!(Div, div, /,
                  grad(self) = rhs.v.recip(),
                  grad(rhs) = -self.v * sqr!(rhs.v.recip()));
impl_bin_op_grad!(Rem, rem, %,
                  // just treat the derivative here as disconnecting by sticking a 0 in
                  grad(self) = 0.0,
                  grad(rhs) = 0.0);

// Specialized macro for constants, we don't need to track them in the graph.
macro_rules! impl_bin_const_op_grad {
  ($type: ty, $func: ident, $op: tt, grad($l: ident, $r: ident) = $l_grad: expr) => {
    impl $type for Var {
      type Output = Var;
      fn $func($l, $r: ScalarFloat) -> Self::Output {
        Self::new_w_idx($l.v $op $r, $l.create_unary($l_grad))
      }
    }
  }
}

impl_bin_const_op_grad!(Add<ScalarFloat>, add, +, grad(self, rhs) = 1.0);
impl_bin_const_op_grad!(Sub<ScalarFloat>, sub, -, grad(self, rhs) = 1.0);
impl_bin_const_op_grad!(Mul<ScalarFloat>, mul, *, grad(self, rhs) = rhs);
impl_bin_const_op_grad!(Div<ScalarFloat>, div, /, grad(self, rhs) = rhs.recip());
// TODO should fix this here maybe
impl_bin_const_op_grad!(Rem<ScalarFloat>, rem, %, grad(self, rhs) = 0.0);

impl std::ops::Neg for Var {
  type Output = Self;
  impl_unary_grad!(neg, grad(self) = -1.0);
}

impl PartialEq for Var {
  fn eq(&self, o: &Self) -> bool { self.v == o.v }
}

impl PartialOrd for Var {
  fn partial_cmp(&self, o: &Self) -> Option<Ordering> { self.v.partial_cmp(&o.v) }
}

impl Zero for Var {
  fn zero() -> Self { Self::new(ScalarFloat::zero()) }
  fn is_zero(&self) -> bool { self.v.is_zero() }
}

impl One for Var {
  fn one() -> Self { Self::new(ScalarFloat::one()) }
  fn is_one(&self) -> bool { self.v.is_one() }
}

impl NumCast for Var {
  fn from<T: ToPrimitive>(n: T) -> Option<Self> { Some(Self::new(NumCast::from(n)?)) }
}

impl ToPrimitive for Var {
  fn to_i64(&self) -> Option<i64> { self.v.to_i64() }
  fn to_u64(&self) -> Option<u64> { self.v.to_u64() }
}
