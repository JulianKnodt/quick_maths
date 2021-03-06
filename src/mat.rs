#![allow(clippy::many_single_char_names)]
use crate::{
  num::DefaultFloat,
  quat::Quat,
  vec::{Vec2, Vec3, Vec4, Vector},
};
use num::{Float, One, Zero};
use std::ops::{Add, Div, Index, IndexMut, Mul, Range, Sub};

// TODO decide if matrix should be a type alias or a struct
/// A matrix, where each vector represents a column
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Matrix<const M: usize, const N: usize, T = DefaultFloat>(pub Vector<N, Vector<M, T>>);

/// 4x4 Matrix
pub type Mat4<T = DefaultFloat> = Matrix<4, 4, T>;
/// 3x3 Matrix
pub type Mat3<T = DefaultFloat> = Matrix<3, 3, T>;
/// 2x2 Matrix
pub type Mat2<T = DefaultFloat> = Matrix<2, 2, T>;

impl<T, const M: usize, const N: usize> Matrix<M, N, T> {
  /// Top-Bottom, Left-Right iterator.
  pub fn y_x_iter(&self) -> impl Iterator<Item = &T> + '_ {
    (self.0).0.iter().flat_map(|col| col.0.iter())
    // (0..N).flat_map(move |x| (0..M).map(move |y| &self[x][y]))
  }
  /// Left-Right, Top-Bottom iterator.
  pub fn x_y_iter(&self) -> impl Iterator<Item = &T> + '_ {
    (0..M).flat_map(move |y| (0..N).map(move |x| &self[x][y]))
  }
  pub fn col_iter(&self) -> impl Iterator<Item = &Vector<M, T>> { self.0.iter() }
}

impl<T: Float, const M: usize, const N: usize> Matrix<M, N, T> {
  pub fn dot(&self, vec: &Vector<N, T>) -> Vector<M, T> {
    let mut out: Vector<M, T> = Vector::zero();
    for i in 0..N {
      out = out + self[i] * vec[i];
    }
    out
  }
  pub(crate) fn qdot<const Q: usize>(&self, vec: &Vector<Q, T>) -> Vector<M, T> {
    // Check that Q is less than or equal to N.
    // We allow smaller values so we can multiply smaller vectors efficiently
    assert!(Q <= N);
    let mut out: Vector<M, T> = Vector::zero();
    for i in 0..Q {
      out = out + self.0[i] * vec[i];
    }
    out
  }
  pub fn t(&self) -> Matrix<N, M, T> {
    let mut empty: Matrix<N, M, T> = Matrix::zero();
    for y in 0..N {
      for x in 0..M {
        empty.0[y][x] = self.0[y][x];
      }
    }
    empty
  }
  /// Performs naive matrix multiplication
  pub fn matmul<const P: usize>(&self, o: &Matrix<N, P, T>) -> Matrix<M, P, T> {
    let mut empty: Matrix<M, P, T> = Matrix::zero();
    for i in 0..P {
      empty[i] = self.dot(&o[i]);
    }
    empty
  }
  pub fn swap_rows(&mut self, cols: Range<usize>, a: usize, b: usize) {
    assert!(a < M);
    assert!(b < M);
    for i in cols {
      self[i].0.swap(a, b);
    }
  }
  pub fn swap_cols(&mut self, a: usize, b: usize) {
    assert!(a < N);
    assert!(b < N);
    (self.0).0.swap(a, b);
  }
  pub fn apply_fn<F, S>(&self, f: F) -> Matrix<M, N, S>
  where
    F: FnMut(T) -> S + Copy,
    S: Float, {
    let mut empty: Matrix<M, N, S> = Matrix::zero();
    for i in 0..N {
      empty[i] = self[i].apply_fn(f);
    }
    empty
  }
  /// Zero extend this matrix to a larger size
  pub fn zxtend<const I: usize, const J: usize>(&self) -> Matrix<I, J, T> {
    assert!(I >= M);
    assert!(J >= N);
    let mut out: Matrix<I, J, T> = Matrix::zero();
    for i in 0..N {
      out[i] = self[i].extend(T::zero());
    }
    out
  }
  /// Take some subset of this matrix(only takes from the topt left)
  pub fn reduce<const I: usize, const J: usize>(&self) -> Matrix<I, J, T> {
    assert!(I <= M);
    assert!(J <= N);
    let mut out: Matrix<I, J, T> = Matrix::zero();
    for i in 0..J {
      out[i] = self[i].reduce();
    }
    out
  }
  pub fn frobenius(&self) -> T { (self.0).0.iter().fold(T::zero(), |acc, n| acc + n.dot(n)) }
}

impl<T: Float, const M: usize> Matrix<M, M, T> {
  /// Creates a square matrix from a diagonal
  pub fn from_diag(v: Vector<M, T>) -> Self {
    let mut out = Self::zero();
    for i in 0..M {
      out[i][i] = v[i];
    }
    out
  }
  /// Returns elements on the diagonal from top left to bottom right
  pub fn diag(&self) -> impl Iterator<Item = T> + '_ { (0..M).map(move |i| self[i][i]) }
  /// Returns elements not on the diagonal in no specific order
  pub fn off_diag(&self) -> impl Iterator<Item = T> + '_ {
    (0..M).flat_map(move |i| (0..M).filter(move |&j| j != i).map(move |j| self[i][j]))
  }
  /// Identity extend this matrix to a larger size(ones along diagonal)
  pub fn ixtend<const I: usize>(&self) -> Matrix<I, I, T> {
    let mut out: Matrix<I, I, T> = self.zxtend();
    for i in M..I {
      out[i][i] = T::one();
    }
    out
  }
  pub fn trace(&self) -> T { self.diag().fold(T::zero(), |acc, n| acc + n) }
  /// LUP decomposes self into lower triangular, upper triangular and pivot matrix
  pub fn lup(&self) -> (Self, Self, Self) {
    let mut l = Self::one();
    let mut u = *self;
    let mut p = Self::one();
    for k in 0..(M - 1) {
      let i = k + argmax(&u.0[k].apply_fn(T::abs).0[k..]);
      u.swap_rows(k..M, i, k);
      l.swap_rows(0..k, i, k);
      p.swap_rows(0..M, i, k);
      for j in (k + 1)..M {
        l.0[k][j] = u.0[k][j] / u.0[k][k];
        for i in k..M {
          u.0[i][j] = u.0[i][j] - l.0[k][j] * u.0[i][k];
        }
      }
    }
    (l, u, p)
  }
  /// Given an upper triangular matrix and a vector, compute the solution to the system of
  /// equations
  pub fn usolve(&self, b: &Vector<M, T>) -> Vector<M, T> {
    let mut out: Vector<M, T> = Vector::zero();
    for y in (0..M).rev() {
      let mut acc = b[y];
      for x in y + 1..M {
        acc = acc - out[x] * self[x][y];
      }
      out[y] = acc / self[y][y];
    }
    out
  }
  /// Given a lower triangular matrix and a vector, compute the solution to the system of
  /// equations
  pub fn lsolve(&self, b: &Vector<M, T>) -> Vector<M, T> {
    let mut out: Vector<M, T> = Vector::zero();
    for y in 0..M {
      let mut acc = b[y];
      for x in 0..y.saturating_sub(1) {
        acc = acc - out[x] * self[x][y];
      }
      out[y] = acc / self[y][y];
    }
    out
  }
  /// Solves for x in the linear system Ax = b;
  pub fn solve((l, u, p): &(Self, Self, Self), b: &Vector<M, T>) -> Vector<M, T> {
    u.usolve(&l.lsolve(&p.dot(b)))
  }
}

impl<T: Float, const M: usize, const N: usize> Zero for Matrix<M, N, T> {
  fn zero() -> Self { Matrix(Vector::of(Vector::zero())) }
  fn is_zero(&self) -> bool { (self.0).0.iter().all(|c| c.is_zero()) }
}

impl<T: Float> Mat3<T> {
  pub fn new(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Self { Matrix(Vec3::new(c0, c1, c2)) }
  /// Computes the determinant of this matrix
  pub fn det(&self) -> T {
    let &Matrix(Vector(
      [Vector([e00, e01, e02]), Vector([e10, e11, e12]), Vector([e20, e21, e22])],
    )) = self;
    e00 * e11 * e22 +
    e01 * e12 * e20 +
    e02 * e10 * e21 -
    // subtraction component
    e02 * e11 * e20 -
    e01 * e10 * e22 -
    e00 * e12 * e21
  }
  /// Inverts this matrix, does not handle non-invertible matrices
  pub fn inv(&self) -> Self { self.t() / self.det() }
  pub fn rot(around: &Vec3<T>, cos_t: T) -> Self {
    let &Vector([i, j, k]) = around;
    let l = T::one();
    let sin_t = l - cos_t * cos_t;
    Self(Vec3::new(
      Vector([
        i * i * (l - cos_t) + cos_t,
        i * j * (l - cos_t) + k * sin_t,
        i * k * (l - cos_t) - j * sin_t,
      ]),
      Vector([
        i * j * (l - cos_t) - k * sin_t,
        j * j * (l - cos_t) + cos_t,
        j * k * (l - cos_t) - i * sin_t,
      ]),
      Vector([
        i * k * (l - cos_t) + k * sin_t,
        j * k * (l - cos_t) - i * sin_t,
        k * k * (l - cos_t) + cos_t,
      ]),
    ))
  }
  pub fn scale(by: &Vec3<T>) -> Self {
    let &Vector([i, j, k]) = by;
    let o = T::zero();
    Self(Vec3::new(
      Vector([i, o, o]),
      Vector([o, j, o]),
      Vector([o, o, k]),
    ))
  }
  /// Translation operator for 2 space
  pub fn translate(by: &Vec2<T>) -> Self {
    let &Vector([i, j]) = by;
    let o = T::zero();
    let l = T::one();
    Self(Vec3::new(
      Vector([l, o, i]),
      Vector([o, l, j]),
      Vector([o, o, l]),
    ))
  }
  pub fn project(normal: &Vec3<T>) -> Self {
    let normal = normal.norm();
    let Vector([i, j, k]) = normal;
    let l = T::one();
    let o = T::zero();
    let (x, y, z) = match (i.is_zero(), j.is_zero(), k.is_zero()) {
      (true, true, true) => return Self::zero(),
      (true, true, false) | (true, false, true) => (l, o, o),
      (false, true, true) => (o, o, l),
      (false, false, true) => (-j, i, k),
      (false, true, false) => (-k, j, i),
      (true, false, false) => (i, -k, j),
      (false, false, false) => (i, k, -j),
    };
    let b_0 = Vector([x, y, z]);
    let b_1 = normal.cross(&b_0);
    Self::new(b_0, b_1, Vec3::zero())
  }
  /// https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
  /// Converts a quaternion into an equivalent matrix
  pub fn from_quat(q: Quat<T>) -> Self {
    let Vector([x, y, z, w]) = q;
    let t = T::one() + T::one();
    Matrix(Vec3::new(
      Vec3::new(
        w * w + x * x - y * y - z * z,
        t * x * y + t * w * z,
        t * x * z - t * w * y,
      ),
      Vec3::new(
        t * x * y - t * w * z,
        w * w - x * x + y * y - z * z,
        t * y * z + t * w * x,
      ),
      Vec3::new(
        t * x * z + t * w * y,
        t * y * z - t * w * x,
        w * w - x * x - y * y + z * z,
      ),
    ))
  }
}

impl<T: Float> Mat2<T> {
  /// Computes the determinant of this matrix
  pub fn det(&self) -> T {
    let &Matrix(Vector([Vector([e00, e01]), Vector([e10, e11])])) = self;
    e00 * e11 - e01 * e10
  }
  /// Inverts this matrix, does not handle non-invertible matrices
  pub fn inv(&self) -> Self {
    let det = self.det();
    let &Matrix(Vector([Vector([e00, e01]), Vector([e10, e11])])) = self;
    Matrix(Vec2::new(Vector([e11, -e01]), Vector([-e10, e00]))) / det
  }
  /// Returns the rotation matrix given a theta in the counterclockwise direction
  pub fn rot(theta: T) -> Self {
    let (sin_t, cos_t) = theta.sin_cos();
    Matrix(Vec2::new(Vector([cos_t, sin_t]), Vector([-sin_t, cos_t])))
  }
  /// Returns the scale matrix given scale in each direction
  pub fn scale(sx: T, sy: T) -> Self {
    let o = T::zero();
    Matrix(Vec2::new(Vec2::new(sx, o), Vec2::new(o, sy)))
  }
}

/// Multiplicative identity
impl<T: Float, const M: usize> One for Matrix<M, M, T> {
  fn one() -> Self {
    let mut empty = Self::zero();
    for i in 0..M {
      empty[i][i] = T::zero();
    }
    empty
  }
  fn is_one(&self) -> bool {
    self.diag().all(|v| v.is_one()) && self.off_diag().all(|v| v.is_zero())
  }
}

/// Computes the argmax over a slice of floats assuming it is non-empty
fn argmax<T: Float>(v: &[T]) -> usize {
  assert!(!v.is_empty());
  v.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap()
    .0
}

impl<T: Float> Mat4<T> {
  pub fn new(c0: Vec4<T>, c1: Vec4<T>, c2: Vec4<T>, c3: Vec4<T>) -> Self {
    Matrix(Vector([c0, c1, c2, c3]))
  }
  /// Returns a translation matrix by t
  pub fn translate(t: Vec3<T>) -> Self {
    let l = T::one();
    let o = T::zero();
    let Vector([x, y, z]) = t;
    Matrix(Vector([
      Vector([l, o, o, o]),
      Vector([o, l, o, o]),
      Vector([o, o, l, o]),
      Vector([x, y, z, l]),
    ]))
  }
  /// Returns the translation encoded in this matrix
  pub fn translation(&self) -> Vec3<T> { self[3].homogenize() }
  // Computes the determinant of this matrix
  // fn det(&self) -> T { todo!() }

  /// Computes the inverse of this matrix if it exists
  pub fn inv(&self) -> Self {
    // twas a b'
    // taken from https://github.com/mrdoob/three.js/blob/master/src/math/Matrix4.js
    // which took it from http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.htm
    let &Matrix(Vector(
      [Vector([e11, e21, e31, e41]), Vector([e12, e22, e32, e42]), Vector([e13, e23, e33, e43]), Vector([e14, e24, e34, e44])],
    )) = self;
    let t11 =
      e23 * e34 * e42 - e24 * e33 * e42 + e24 * e32 * e43 - e22 * e34 * e43 - e23 * e32 * e44
        + e22 * e33 * e44;
    let t12 =
      e14 * e33 * e42 - e13 * e34 * e42 - e14 * e32 * e43 + e12 * e34 * e43 + e13 * e32 * e44
        - e12 * e33 * e44;
    let t13 =
      e13 * e24 * e42 - e14 * e23 * e42 + e14 * e22 * e43 - e12 * e24 * e43 - e13 * e22 * e44
        + e12 * e23 * e44;
    let t14 =
      e14 * e23 * e32 - e13 * e24 * e32 - e14 * e22 * e33 + e12 * e24 * e33 + e13 * e22 * e34
        - e12 * e23 * e34;
    let det = e11 * t11 + e21 * t12 + e31 * t13 + e41 * t14;
    // Don't check if det is zero here just assume it's invertible
    let det_i = det.recip();
    let o11 = t11 * det_i;
    let o21 =
      (e24 * e33 * e41 - e23 * e34 * e41 - e24 * e31 * e43 + e21 * e34 * e43 + e23 * e31 * e44
        - e21 * e33 * e44)
        * det_i;
    let o31 =
      (e22 * e34 * e41 - e24 * e32 * e41 + e24 * e31 * e42 - e21 * e34 * e42 - e22 * e31 * e44
        + e21 * e32 * e44)
        * det_i;
    let o41 =
      (e23 * e32 * e41 - e22 * e33 * e41 - e23 * e31 * e42 + e21 * e33 * e42 + e22 * e31 * e43
        - e21 * e32 * e43)
        * det_i;

    let o12 = t12 * det_i;
    let o22 =
      (e13 * e34 * e41 - e14 * e33 * e41 + e14 * e31 * e43 - e11 * e34 * e43 - e13 * e31 * e44
        + e11 * e33 * e44)
        * det_i;
    let o32 =
      (e14 * e32 * e41 - e12 * e34 * e41 - e14 * e31 * e42 + e11 * e34 * e42 + e12 * e31 * e44
        - e11 * e32 * e44)
        * det_i;
    let o42 =
      (e12 * e33 * e41 - e13 * e32 * e41 + e13 * e31 * e42 - e11 * e33 * e42 - e12 * e31 * e43
        + e11 * e32 * e43)
        * det_i;

    let o13 = t13 * det_i;
    let o23 =
      (e14 * e23 * e41 - e13 * e24 * e41 - e14 * e21 * e43 + e11 * e24 * e43 + e13 * e21 * e44
        - e11 * e23 * e44)
        * det_i;
    let o33 =
      (e12 * e24 * e41 - e14 * e22 * e41 + e14 * e21 * e42 - e11 * e24 * e42 - e12 * e21 * e44
        + e11 * e22 * e44)
        * det_i;
    let o43 =
      (e13 * e22 * e41 - e12 * e23 * e41 - e13 * e21 * e42 + e11 * e23 * e42 + e12 * e21 * e43
        - e11 * e22 * e43)
        * det_i;

    let o14 = t14 * det_i;
    let o24 =
      (e13 * e24 * e31 - e14 * e23 * e31 + e14 * e21 * e33 - e11 * e24 * e33 - e13 * e21 * e34
        + e11 * e23 * e34)
        * det_i;
    let o34 =
      (e14 * e22 * e31 - e12 * e24 * e31 - e14 * e21 * e32 + e11 * e24 * e32 + e12 * e21 * e34
        - e11 * e22 * e34)
        * det_i;
    let o44 =
      (e12 * e23 * e31 - e13 * e22 * e31 + e13 * e21 * e32 - e11 * e23 * e32 - e12 * e21 * e33
        + e11 * e22 * e33)
        * det_i;
    Matrix(Vector([
      Vector([o11, o21, o31, o41]),
      Vector([o12, o22, o32, o42]),
      Vector([o13, o23, o33, o43]),
      Vector([o14, o24, o34, o44]),
    ]))
  }
  /// Returns the pseudo-inverse of this matrix
  pub fn p_inv(&self) -> Self {
    let t = self.t();
    t.matmul(self).inv().matmul(&t)
  }
}

impl<T, const M: usize> Matrix<M, 1, T> {
  pub fn squeeze(self) -> Vector<M, T> {
    let Matrix(Vector([v])) = self;
    v
  }
}

impl<T, const M: usize, const N: usize> Index<usize> for Matrix<M, N, T> {
  type Output = Vector<M, T>;
  fn index(&self, i: usize) -> &Self::Output { &self.0[i] }
}

impl<T, const M: usize, const N: usize> IndexMut<usize> for Matrix<M, N, T> {
  fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut self.0[i] }
}

macro_rules! def_op {
  ($ty: ty, $func: ident, $op: tt) => {
    impl<T: Float, const M: usize, const N: usize> $ty for Matrix<M, N, T>
    {
      type Output = Self;
      fn $func(mut self, o: Self) -> Self {
        for x in 0..N {
          self.0[x] = self.0[x] $op o.0[x];
        }
        self
      }
    }
  };
}

def_op!(Add, add, +);
def_op!(Sub, sub, -);
def_op!(Mul, mul, *);
def_op!(Div, div, /);

macro_rules! def_scalar_op {
  ($ty: ty, $func: ident, $op: tt) => {
    impl<T: Float, const M: usize, const N: usize> $ty for Matrix<M, N, T> {
      type Output = Self;
      fn $func(mut self, o: T) -> Self {
        for x in 0..N {
          self.0[x] = self.0[x] $op o;
        }
        self
      }
    }
  };
}

def_scalar_op!(Add<T>, add, +);
def_scalar_op!(Sub<T>, sub, -);
def_scalar_op!(Mul<T>, mul, *);
def_scalar_op!(Div<T>, div, /);

/*
// XXX Commented out not because it's broken but because I shifted it from the old matrix code
#[test]
fn test_lu_decomp() {
  let a: Matrix4<f32> = Matrix4([
    Vec4([3., -3., 6., -9.]),
    Vec4([-7., 5., 1., 0.]),
    Vec4([6., -4., 0., -5.]),
    Vec4([-9., 5., -5., 12.]),
  ]);
  let lup = a.lup();
  let (l, u, p) = lup;
  let out = p.t().matmul(&l.matmul(&u));
  for i in 0..Matrix4::<f32>::N {
    for j in 0..Matrix4::<f32>::N {
      assert!((a.0[i][j] - out.0[i][j]).abs() < f32::epsilon());
    }
  }
  let x = Vec4([3.0, 0.0, 1.2, 4.5]);
  let b = a.vecmul(&x);
  let x_p = Matrix4::solve(&lup, &b);
  assert!((x_p - x).sqr_magn() < 0.00001);
}
*/
macro_rules! elemwise_impl {
  ($func: ident, $call: path, $name: expr) => {
    #[doc="Element-wise "]
    #[doc=$name]
    #[doc="."]
    pub fn $func(&self) -> Self { self.apply_fn($call) }
  };
  ($($func: ident, $call: path;)*) => {
    $(elemwise_impl!($func, $call, stringify!($func));)*
  };
}

macro_rules! curried_elemwise_impl {
  ($func: ident, $call: path, $name: expr) => {
    #[doc="Element-wise "]
    #[doc=$name]
    #[doc="."]
    #[inline]
    pub fn $func(&self, v: T) -> Self { self.apply_fn(|u| $call(u, v)) }
  };
  ($func: ident, $call: path) => {
    curried_elemwise_impl!($func, $call, stringify!($func));
  };
}
impl<T: Float, const M: usize, const N: usize> Matrix<M, N, T> {
  // Trigonometric stuff
  elemwise_impl!(
    cos, T::cos;
    sin, T::sin;
    tan, T::tan;

    acos, T::acos;
    asin, T::asin;
    atan, T::atan;

    acosh, T::acosh;
    asinh, T::asinh;
    atanh, T::atanh;
  );

  curried_elemwise_impl!(atan2, T::atan2);
  curried_elemwise_impl!(hypot, T::hypot);

  // Rounding stuff
  elemwise_impl!(
    ceil, T::ceil;
    floor, T::floor;
    round, T::round;
  );

  // Decomposition stuff
  elemwise_impl!(
    fract, T::fract;
    trunc, T::trunc;
  );

  // Sign value stuff
  curried_elemwise_impl!(abs_sub, T::abs_sub);
  elemwise_impl!(
    abs, T::abs;
    signum, T::signum;
  );

  // Reciprocal
  elemwise_impl!(
    recip, T::recip;
  );

  // Logarithmic stuff
  elemwise_impl!(
    log2, T::log2;
    log10, T::log10;
    ln, T::ln;
    ln_1p, T::ln_1p;
    exp, T::exp;
    exp2, T::exp2;
    exp_m1, T::exp_m1;
    sqrt, T::sqrt;
    cbrt, T::cbrt;
  );

  curried_elemwise_impl!(powf, T::powf);
  pub fn powi(&self, v: i32) -> Self { self.apply_fn(|u| u.powi(v)) }
  curried_elemwise_impl!(log, T::log);

  // Min/max stuff
  curried_elemwise_impl!(max, T::max);
  curried_elemwise_impl!(min, T::min);

  // Degree related stuff
  elemwise_impl!(
    to_degrees, T::to_degrees;
    to_radians, T::to_radians;
  );
}
