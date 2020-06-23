#![allow(clippy::many_single_char_names)]
use crate::num::DefaultFloat;
use num::{Float, One, Zero};
use std::{
  array::LengthAtMost32,
  borrow::{Borrow, BorrowMut},
  mem::{forget, MaybeUninit},
  ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
  },
};

/// Vector over floats and a const-size.
/// Often used through Vec2, Vec3, and Vec4 instead of the raw struct.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub struct Vector<T = DefaultFloat, const N: usize>(pub [T; N])
where
  [T; N]: LengthAtMost32;

/// 2D vector with default float type (f32).
pub type Vec2<T = DefaultFloat> = Vector<T, 2>;
/// 3D vector with default float type (f32).
pub type Vec3<T = DefaultFloat> = Vector<T, 3>;
/// 4D vector with default float type (f32).
/// Often implicitly created by Vec3::homogeneous.
pub type Vec4<T = DefaultFloat> = Vector<T, 4>;

impl<T: Copy, const N: usize> Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  /// Creates a vector of the value v (every element = v).
  pub fn of(v: T) -> Self { Vector([v; N]) }

  /// Applies this function to every vector value.
  #[inline]
  pub fn apply_fn<F, S>(self, mut f: F) -> Vector<S, N>
  where
    F: FnMut(T) -> S,
    [S; N]: LengthAtMost32, {
    let mut out: [MaybeUninit<S>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
      out[i] = MaybeUninit::new(f(self[i]));
    }
    let ptr = &mut out as *mut _ as *mut [S; N];
    let res = unsafe { ptr.read() };
    forget(out);
    Vector(res)
  }
}
impl<T, const N: usize> Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  pub fn with<F>(mut f: F) -> Self
  where
    F: FnMut(usize) -> T, {
    let mut out: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
      out[i] = MaybeUninit::new(f(i));
    }
    let ptr = &mut out as *mut _ as *mut [T; N];
    let res = unsafe { ptr.read() };
    forget(out);
    Vector(res)
  }
}

impl<T: Float + Zero, const N: usize> Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  /// Takes the dot product of two vectors
  pub fn dot(&self, o: &Self) -> T { (0..N).fold(T::zero(), |acc, n| acc + self.0[n] * o.0[n]) }
  /// Computes the sqr_magnitude of the vector
  pub fn sqr_magn(&self) -> T { self.dot(self) }
  /// Takes the magnitude of the vector
  pub fn magn(&self) -> T { self.sqr_magn().sqrt() }
  /// Returns a unit vector in the same direction as self.
  /// Consider division instead of calling this method if you need efficiency.
  pub fn norm(&self) -> Self { *self / self.magn() }
  pub fn cos_similarity(&self, o: &Self) -> T { self.dot(o) / (self.magn() * o.magn()) }
  pub fn reflect(&self, across: &Self) -> Self {
    *self - (*across) * self.dot(across) * (T::one() + T::one())
  }
  pub fn refract(&self, norm: &Self, eta: T) -> Option<Self> {
    let cos_l = self.dot(norm);
    let discrim = T::one() - eta * eta * (T::one() - cos_l * cos_l);
    if discrim.is_sign_negative() {
      return None;
    }
    let cos_r = discrim.sqrt();
    Some(*self * eta - *norm * (eta * cos_l + cos_r))
  }
  /// Computes a vector from a list of strings.
  pub fn from_str_radix(strs: [&str; N], radix: u32) -> Result<Self, T::FromStrRadixErr> {
    let mut out = Self::zero();
    for i in 0..N {
      out[i] = T::from_str_radix(strs[i], radix)?;
    }
    Ok(out)
  }
  /// Linearly interpolates from self to v according to alpha, where 0 => self, and 1 => v.
  pub fn lerp(&self, v: &Self, alpha: T) -> Self { *self * (T::one() - alpha) + *v * alpha }
  /// Computes the max component of this vector
  pub fn max_component(&self) -> usize {
    let mut max_pos = 0;
    for i in 1..N {
      if self[i] > self[max_pos] {
        max_pos = i;
      }
    }
    max_pos
  }
  /// Computes the minimum component of this vector
  pub fn min_component(&self) -> usize {
    let mut min_pos = 0;
    for i in 1..N {
      if self[i] < self[min_pos] {
        min_pos = i;
      }
    }
    min_pos
  }
  /// Clamps self between min and max
  pub fn clamp(&mut self, min: T, max: T) {
    for i in 0..N {
      self[i] = self[i].min(max).max(min);
    }
  }
  pub fn dist(&self, o: &Self) -> T { (*self - *o).magn() }
  /// Zero-extend this vector to a larger vector.
  /// Must increase the size of the vector or keep it the same size.
  pub fn zxtend<const M: usize>(&self) -> Vector<T, M>
  where
    [T; M]: LengthAtMost32, {
    assert!(M >= N);
    let mut out: Vector<T, M> = Vector::zero();
    for i in 0..N {
      out[i] = self[i];
    }
    out
  }
  /// Shrink this vector to a lower dimension
  /// Must lower or keep the same size.
  pub fn reduce<const M: usize>(&self) -> Vector<T, M>
  where
    [T; M]: LengthAtMost32, {
    assert!(M <= N);
    let mut out: Vector<T, M> = Vector::zero();
    for i in 0..M {
      out[i] = self[i];
    }
    out
  }

  pub fn sift(&self, o: &Self) -> (Self, Self) {
    let mut min = *self;
    let mut max = *o;
    for i in 0..N {
      if self[i] > o[i] {
        max[i] = self[i];
        min[i] = o[i];
      }
    }
    (min, max)
  }
}

impl<const N: usize> Vector<bool, N>
where
  [bool; N]: LengthAtMost32,
{
  pub fn any(&self) -> bool {
    for i in 0..N {
      if self[i] {
        return true;
      }
    }
    false
  }
  pub fn all(&self) -> bool {
    for i in 0..N {
      if !self[i] {
        return false;
      }
    }
    true
  }
}

impl<T: Float> Vec3<T> {
  pub fn new(a: T, b: T, c: T) -> Self { Vector([a, b, c]) }
  /// Takes the cross product of self with other
  pub fn cross(&self, o: &Self) -> Self {
    let [a, b, c] = self.0;
    let [x, y, z] = o.0;
    Vector([b * z - c * y, c * x - a * z, a * y - b * z])
  }
  pub fn sided(&self, o: &Self, normal: &Self) -> bool {
    self.cross(o).dot(normal).is_sign_positive()
  }

  /// Returns the homogeneous form of this vector.
  pub fn homogeneous(&self) -> Vec4<T> {
    let &Vector([x, y, z]) = self;
    Vec4::new(x, y, z, T::one())
  }

  pub fn homogenize(&self) -> Vec2<T> {
    let &Vector([x, y, z]) = self;
    Vec2::new(x / z, y / z)
  }

  /// X component of this vector
  pub fn x(&self) -> T { self[0] }
  /// Y component of this vector
  pub fn y(&self) -> T { self[1] }
  /// Z component of this vector
  pub fn z(&self) -> T { self[2] }
}

impl<T: Copy> Vec2<T> {
  pub fn new(a: T, b: T) -> Self { Vector([a, b]) }
  pub fn x(&self) -> T { self[0] }
  pub fn y(&self) -> T { self[1] }
  pub fn flip(&self) -> Self {
    let [i, j] = self.0;
    Vec2::new(j, i)
  }
}

impl<T: Float> Vec2<T> {
  pub fn signed_angle(&self, dst: &Self) -> T {
    let [i, j] = self.0;
    let [x, y] = dst.0;
    (i * y - j * x).atan2(self.dot(dst))
  }
  pub fn perp(&self) -> Self {
    let [i, j] = self.0;
    Vec2::new(j, -i)
  }
  pub fn homogeneous(&self) -> Vec3<T> {
    let [i, j] = self.0;
    Vec3::new(i, j, T::one())
  }
}

impl<T: Float> Vec4<T> {
  pub fn new(a: T, b: T, c: T, w: T) -> Self { Vector([a, b, c, w]) }
  pub fn x(&self) -> T { self[0] }
  pub fn y(&self) -> T { self[1] }
  pub fn z(&self) -> T { self[2] }
  pub fn w(&self) -> T { self[3] }
  pub fn homogenize(&self) -> Vec3<T> {
    let &Vector([x, y, z, w]) = self;
    Vec3::new(x / w, y / w, z / w)
  }
}

// Trait implementations for convenience

impl<T, const N: usize> AsRef<[T]> for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  fn as_ref(&self) -> &[T] { &self.0 }
}

impl<T, const N: usize> Borrow<[T]> for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  fn borrow(&self) -> &[T] { &self.0 }
}

impl<T, const N: usize> BorrowMut<[T]> for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  fn borrow_mut(&mut self) -> &mut [T] { &mut self.0 }
}

// Op implementations

impl<T: Float, const N: usize> One for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  fn one() -> Self { Vector([T::one(); N]) }
  fn is_one(&self) -> bool { self.0.iter().all(T::is_one) }
}

impl<T: Float, const N: usize> Zero for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  fn zero() -> Self { Vector([T::zero(); N]) }
  fn is_zero(&self) -> bool { self.0.iter().all(T::is_zero) }
}

impl<T, const N: usize> Index<usize> for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  type Output = T;
  fn index(&self, i: usize) -> &Self::Output { &self.0[i] }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut self.0[i] }
}

impl<T: Neg<Output = T> + Copy, const N: usize> Neg for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  type Output = Self;
  fn neg(mut self) -> Self::Output {
    for i in 0..N {
      self.0[i] = -self.0[i];
    }
    self
  }
}

impl<T: Not<Output = T> + Copy, const N: usize> Not for Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  type Output = Self;
  fn not(mut self) -> Self::Output {
    for i in 0..N {
      self.0[i] = !self.0[i];
    }
    self
  }
}

macro_rules! vec_op {
  ($t: ident, $func: ident, $op: tt) => {
    impl<T: $t + Copy, const N: usize> $t for Vector<T, N>
    where
      [T; N]: LengthAtMost32,
      [<T as $t>::Output; N]: LengthAtMost32,
    {
      type Output = Vector<T::Output, N>;
      fn $func(self, o: Self) -> Self::Output {
        let mut out: [MaybeUninit<T::Output>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..N {
          out[i] = MaybeUninit::new(self[i] $op o[i]);
        }
        let ptr = &mut out as *mut _ as *mut [T::Output; N];
        let res = unsafe { ptr.read() };
        forget(out);
        Vector(res)
      }
    }
  };
}

vec_op!(Add, add, +);
vec_op!(Mul, mul, *);
vec_op!(Sub, sub, -);
vec_op!(Div, div, /);
vec_op!(Rem, rem, %);

// Boolean operations
vec_op!(BitAnd, bitand, &);
vec_op!(BitOr, bitor, |);
vec_op!(BitXor, bitxor, ^);

macro_rules! scalar_op {
  ($t: ident, $func: ident, $op: tt) => {
    impl<T: $t + Copy, const N: usize> $t<T> for Vector<T, N>
    where
      [T; N]: LengthAtMost32,
      [<T as $t>::Output; N]: LengthAtMost32,
    {
      type Output = Vector<T::Output, N>;
      fn $func(self, o: T) -> Self::Output {
        let mut out: [MaybeUninit<T::Output>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..N {
          out[i] = MaybeUninit::new(self[i] $op o);
        }
        let ptr = &mut out as *mut _ as *mut [T::Output; N];
        let res = unsafe { ptr.read() };
        forget(out);
        Vector(res)
      }
    }
  };
}

scalar_op!(Add, add, +);
scalar_op!(Mul, mul, *);
scalar_op!(Sub, sub, -);
scalar_op!(Div, div, /);
scalar_op!(Rem, rem, %);

// Boolean operations
scalar_op!(BitAnd, bitand, &);
scalar_op!(BitOr, bitor, |);
scalar_op!(BitXor, bitxor, ^);

macro_rules! assign_op {
  ($t: ident, $func: ident, $op: tt) => {
    impl<T: $t + Copy, const N: usize> $t<T> for Vector<T, N>
    where
      [T; N]: LengthAtMost32,
    {
      fn $func(&mut self, o: T) {
        for i in 0..N {
          self.0[i] $op o;
        }
      }
    }
    impl<T: $t + Copy, const N: usize> $t for Vector<T, N>
    where
      [T; N]: LengthAtMost32,
    {
      fn $func(&mut self, o: Self) {
        for i in 0..N {
          self.0[i] $op o.0[i];
        }
      }
    }
  };
}

assign_op!(AddAssign, add_assign, +=);
assign_op!(SubAssign, sub_assign, -=);
assign_op!(MulAssign, mul_assign, *=);
assign_op!(DivAssign, div_assign, /=);
assign_op!(RemAssign, rem_assign, %=);

// Boolean operations
assign_op!(BitAndAssign, bitand_assign, &=);
assign_op!(BitOrAssign, bitor_assign, |=);
assign_op!(BitXorAssign, bitxor_assign, ^=);

macro_rules! elemwise_impl {
  ($func: ident, $call: path, $name: expr) => {
    #[doc="Element-wise "]
    #[doc=$name]
    #[doc="."]
    pub fn $func(&self) -> Self { self.apply_fn($call) }
  };
  ($func: ident, $call: path) => {
    elemwise_impl!($func, $call, stringify!($func));
  };
}

macro_rules! curried_elemwise_impl {
  ($func: ident, $call: path, $name: expr) => {
    #[doc="Element-wise "]
    #[doc=$name]
    #[doc="."]
    pub fn $func(&self, v: T) -> Self { self.apply_fn(|u| $call(u, v)) }
  };
  ($func: ident, $call: path) => {
    curried_elemwise_impl!($func, $call, stringify!($func));
  };
}
impl<T: Float, const N: usize> Vector<T, N>
where
  [T; N]: LengthAtMost32,
{
  // Trigonometric stuff
  elemwise_impl!(cos, T::cos);
  elemwise_impl!(sin, T::sin);
  elemwise_impl!(tan, T::tan);

  elemwise_impl!(acos, T::acos);
  elemwise_impl!(asin, T::asin);
  elemwise_impl!(atan, T::atan);

  elemwise_impl!(acosh, T::acosh);
  elemwise_impl!(asinh, T::asinh);
  elemwise_impl!(atanh, T::atanh);
  curried_elemwise_impl!(atan2, T::atan2);
  curried_elemwise_impl!(hypot, T::hypot);

  // Rounding stuff
  elemwise_impl!(ceil, T::ceil);
  elemwise_impl!(floor, T::floor);
  elemwise_impl!(round, T::round);

  // Decomposition stuff
  elemwise_impl!(fract, T::fract);
  elemwise_impl!(trunc, T::trunc);

  // Sign value stuff
  elemwise_impl!(abs, T::abs);
  curried_elemwise_impl!(abs_sub, T::abs_sub);
  elemwise_impl!(signum, T::signum);

  pub fn is_sign_positive(&self) -> Vector<bool, N>
  where
    [bool; N]: LengthAtMost32, {
    self.apply_fn(T::is_sign_positive)
  }

  pub fn is_sign_negative(&self) -> Vector<bool, N>
  where
    [bool; N]: LengthAtMost32, {
    self.apply_fn(T::is_sign_negative)
  }

  // Reciprocal
  elemwise_impl!(recip, T::recip);

  // Logarithmic stuff
  elemwise_impl!(log2, T::log2);
  elemwise_impl!(log10, T::log10);
  elemwise_impl!(ln, T::ln);
  elemwise_impl!(ln_1p, T::ln_1p);
  elemwise_impl!(exp, T::exp);
  elemwise_impl!(exp2, T::exp2);
  elemwise_impl!(exp_m1, T::exp_m1);
  elemwise_impl!(sqrt, T::sqrt);
  elemwise_impl!(cbrt, T::cbrt);
  curried_elemwise_impl!(powf, T::powf);
  pub fn powi(&self, v: i32) -> Self { self.apply_fn(|u| u.powi(v)) }
  curried_elemwise_impl!(log, T::log);

  // Min/max stuff
  curried_elemwise_impl!(max, T::max);
  curried_elemwise_impl!(min, T::min);

  // Degree related stuff
  elemwise_impl!(to_degrees, T::to_degrees);
  elemwise_impl!(to_radians, T::to_radians);
}

#[test]
fn example() {
  let a = Vec3::of(0.0);
  let b = a + 1.0;
  let c = b.sin();
  let _dot = b.dot(&c);

  let x = c.x();
  let y = c.y();
  let z = c.z();

  let Vector([i, j, k]) = c;
  assert_eq!(x, i);
  assert_eq!(y, j);
  assert_eq!(z, k);
}
