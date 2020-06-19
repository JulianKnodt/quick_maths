use crate::vec::{Quat, Vec2, Vec3, Vec4, Vector};
use num::{Float, One, Zero};
use std::ops::Range;

pub trait Matrix {
  type Field: Float;
  type Vector: Vector<Field = Self::Field>;
  fn det(&self) -> Self::Field;
  fn inv(&self) -> Self;
  fn t(&self) -> Self;
  fn vecmul(&self, v: &Self::Vector) -> Self::Vector;
  fn matmul(&self, o: &Self) -> Self;
}

/// A 3x3 Matrix type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix3<T>(pub [Vec3<T>; 3]);

impl<T: Float> Matrix for Matrix3<T> {
  type Field = T;
  type Vector = Vec3<Self::Field>;
  /// Computes the determinant of this matrix
  fn det(&self) -> T {
    let &Matrix3([Vec3(e00, e01, e02), Vec3(e10, e11, e12), Vec3(e20, e21, e22)]) = self;
    e00 * e11 * e22 +
    e01 * e12 * e20 +
    e02 * e10 * e21 -
    // subtraction
    e02 * e11 * e20 -
    e01 * e10 * e22 -
    e00 * e12 * e21
  }
  /// Basic vector multiplication by a vector
  fn vecmul(&self, o: &Vec3<T>) -> Vec3<T> {
    let &Matrix3([c0, c1, c2]) = self;
    let &Vec3(x, y, z) = o;
    c0 * x + c1 * y + c2 * z
  }
  /// Inverts this matrix, does not handle non-invertible matrices
  fn inv(&self) -> Self { self.t() / self.det() }
  /// returns the matrix multiplication of this matrix and another one
  fn matmul(&self, o: &Self) -> Self {
    let &Matrix3([c0, c1, c2]) = o;
    Matrix3([self.vecmul(&c0), self.vecmul(&c1), self.vecmul(&c2)])
  }
  /// transposes the matrix
  fn t(&self) -> Self {
    let &Matrix3([Vec3(e00, e01, e02), Vec3(e10, e11, e12), Vec3(e20, e21, e22)]) = self;
    Matrix3([
      Vec3(e00, e10, e20),
      Vec3(e01, e11, e21),
      Vec3(e02, e12, e22),
    ])
  }
}

impl<T: Clone> From<T> for Matrix3<T> {
  fn from(t: T) -> Self { Matrix3([Vec3::from(t.clone()), Vec3::from(t.clone()), Vec3::from(t)]) }
}

impl<T: Zero> Zero for Matrix3<T> {
  fn zero() -> Self { Self([Vec3::zero(), Vec3::zero(), Vec3::zero()]) }
  fn is_zero(&self) -> bool { self.0.iter().all(|col| col.is_zero()) }
}

/// Multiplicative identity
impl<T: One + Zero + Copy + PartialEq> One for Matrix3<T> {
  fn one() -> Self {
    let o = T::zero();
    let l = T::one();
    Self([Vec3(l, o, o), Vec3(o, l, o), Vec3(o, o, l)])
  }
  fn is_one(&self) -> bool {
    let o = T::zero();
    let l = T::one();
    self == &Matrix3([Vec3(l, o, o), Vec3(o, l, o), Vec3(o, o, l)])
  }
}

impl<T: Float> From<Quat<T>> for Matrix3<T> {
  // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
  /// Converts a quaternion into an equivalent matrix
  fn from(q: Quat<T>) -> Self {
    let Quat(Vec3(x, y, z), w) = q;
    let t = T::from(2.0).unwrap();
    Matrix3([
      Vec3(
        w * w + x * x - y * y - z * z,
        t * x * y + t * w * z,
        t * x * z - t * w * y,
      ),
      Vec3(
        t * x * y - t * w * z,
        w * w - x * x + y * y - z * z,
        t * y * z + t * w * x,
      ),
      Vec3(
        t * x * z + t * w * y,
        t * y * z - t * w * x,
        w * w - x * x - y * y + z * z,
      ),
    ])
  }
}

impl<T: Float> Matrix3<T> {
  /// Returns the main diagonal along this matrix
  pub fn diag(&self) -> Vec3<T> {
    let &Matrix3([Vec3(e00, _, _), Vec3(_, e11, _), Vec3(_, _, e22)]) = self;
    Vec3(e00, e11, e22)
  }
  /// returns the trace of this matrix
  pub fn trace(&self) -> T {
    let Vec3(e00, e11, e22) = self.diag();
    e00 + e11 + e22
  }
  pub fn rot(around: &Vec3<T>, cos_t: T) -> Self {
    let &Vec3(i, j, k) = around;
    let l = T::one();
    let sin_t = l - cos_t * cos_t;
    Self([
      Vec3(
        i * i * (l - cos_t) + cos_t,
        i * j * (l - cos_t) + k * sin_t,
        i * k * (l - cos_t) - j * sin_t,
      ),
      Vec3(
        i * j * (l - cos_t) - k * sin_t,
        j * j * (l - cos_t) + cos_t,
        j * k * (l - cos_t) - i * sin_t,
      ),
      Vec3(
        i * k * (l - cos_t) + k * sin_t,
        j * k * (l - cos_t) - i * sin_t,
        k * k * (l - cos_t) + cos_t,
      ),
    ])
  }
  pub fn scale(by: &Vec3<T>) -> Self {
    let &Vec3(i, j, k) = by;
    let o = T::zero();
    Self([Vec3(i, o, o), Vec3(o, j, o), Vec3(o, o, k)])
  }
  /// translates x and y
  pub fn trans(by: &Vec2<T>) -> Self {
    let &Vec2(i, j) = by;
    let o = T::zero();
    let l = T::one();
    Self([Vec3(l, o, i), Vec3(o, l, j), Vec3(o, o, l)])
  }
  /// Projects onto the plane defined by normal
  pub fn project(normal: &Vec3<T>) -> Self {
    let normal = normal.norm();
    let Vec3(i, j, k) = normal;
    let l = T::one();
    let o = T::zero();
    let b_0 = match (i.is_zero(), j.is_zero(), k.is_zero()) {
      (true, true, true) => return Self::zero(),
      (true, true, false) | (true, false, true) => Vec3(l, o, o),
      (false, true, true) => Vec3(o, o, l),
      (false, false, true) => Vec3(-j, i, k),
      (false, true, false) => Vec3(-k, j, i),
      (true, false, false) => Vec3(i, -k, j),
      (false, false, false) => Vec3(i, k, -j),
    };
    let b_1 = normal.cross(&b_0);
    Self([b_0, b_1, Vec3::zero()])
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix2<T>([Vec2<T>; 2]);

impl<T: Float> Matrix for Matrix2<T> {
  type Field = T;
  type Vector = Vec2<Self::Field>;
  /// Computes the determinant of this matrix
  fn det(&self) -> Self::Field {
    let &Matrix2([Vec2(e00, e01), Vec2(e10, e11)]) = self;
    e00 * e11 - e01 * e10
  }
  /// Inverts this matrix, does not handle non-invertible matrices
  fn inv(&self) -> Self {
    let det = self.det();
    let &Matrix2([Vec2(e00, e01), Vec2(e10, e11)]) = self;
    Matrix2([Vec2(e11, -e01), Vec2(-e10, e00)]) / det
  }
  /// transposes the matrix
  fn t(&self) -> Self {
    let &Matrix2([Vec2(e00, e01), Vec2(e10, e11)]) = self;
    Matrix2([Vec2(e00, e10), Vec2(e01, e11)])
  }
  fn vecmul(&self, v: &Vec2<T>) -> Vec2<T> {
    let &Matrix2([c0, c1]) = self;
    c0 * v.0 + c1 * v.1
  }
  fn matmul(&self, o: &Self) -> Self {
    let &Matrix2([c0, c1]) = o;
    Matrix2([self.vecmul(&c0), self.vecmul(&c1)])
  }
}

impl<T: Float> Matrix2<T> {
  /// Returns the rotation matrix given a theta in the counterclockwise direction
  pub fn rot(theta: T) -> Self {
    let (sin_t, cos_t) = theta.sin_cos();
    Matrix2([Vec2(cos_t, sin_t), Vec2(-sin_t, cos_t)])
  }
  /// Returns the scale matrix given scale in each direction
  pub fn scale(sx: T, sy: T) -> Self {
    let o = T::zero();
    Matrix2([Vec2(sx, o), Vec2(o, sy)])
  }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Matrix4<T>(pub [Vec4<T>; 4]);

impl<T: Zero + Copy> Zero for Matrix4<T> {
  fn zero() -> Self { Self([Vec4::zero(); 4]) }
  fn is_zero(&self) -> bool { self.0.iter().all(|col| col.is_zero()) }
}

/// Multiplicative identity
impl<T: One + Zero + Copy + PartialEq> One for Matrix4<T> {
  fn one() -> Self {
    let o = T::zero();
    let l = T::one();
    Self([
      Vec4([l, o, o, o]),
      Vec4([o, l, o, o]),
      Vec4([o, o, l, o]),
      Vec4([o, o, o, l]),
    ])
  }
  fn is_one(&self) -> bool {
    let l = T::one();
    let o = T::zero();
    &Matrix4([
      Vec4([l, o, o, o]),
      Vec4([o, l, o, o]),
      Vec4([o, o, l, o]),
      Vec4([o, o, o, l]),
    ]) == self
  }
}

impl<T> Matrix4<T> {
  /// Swaps rows within a set of columns for this matrix
  pub fn swap_rows(&mut self, cols: Range<usize>, a: usize, b: usize) {
    self.0[cols].iter_mut().for_each(|c| c.0.swap(a, b));
  }
}

impl<T: Copy> Matrix4<T> {
  pub fn apply_fn<F, S>(&self, f: F) -> Matrix4<S>
  where
    F: Fn(T) -> S + Copy, {
    let &Matrix4([c0, c1, c2, c3]) = self;
    Matrix4([
      c0.apply_fn(f),
      c1.apply_fn(f),
      c2.apply_fn(f),
      c3.apply_fn(f),
    ])
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

impl<T: Float> Matrix4<T> {
  const N: usize = 4;
  /// Returns a translation matrix by t
  pub fn translate(t: Vec3<T>) -> Self {
    let l = T::one();
    let o = T::zero();
    let Vec3(x, y, z) = t;
    Matrix4([
      Vec4([l, o, o, o]),
      Vec4([o, l, o, o]),
      Vec4([o, o, l, o]),
      Vec4([x, y, z, l]),
    ])
  }
  pub fn frobenius(&self) -> T {
    let &Matrix4([c0, c1, c2, c3]) = self;
    (c0.dot(&c0) + c1.dot(&c1) + c2.dot(&c2) + c3.dot(&c3)).sqrt()
  }
  /// LUP decomposes self into lower triangular, upper triangular and pivot matrix
  // TODO this actually is pretty easily made generic so eventually need to do that
  pub fn lup(&self) -> (Self, Self, Self) {
    let mut l = Self::one();
    let mut u = *self;
    let mut p = Self::one();
    for k in 0..(Self::N - 1) {
      let i = k + argmax(&u.0[k].apply_fn(T::abs).0[k..]);
      u.swap_rows(k..Self::N, i, k);
      l.swap_rows(0..k, i, k);
      p.swap_rows(0..Self::N, i, k);
      for j in (k + 1)..Self::N {
        l.0[k][j] = u.0[k][j] / u.0[k][k];
        for i in k..Self::N {
          u.0[i][j] = u.0[i][j] - l.0[k][j] * u.0[i][k];
        }
      }
    }
    (l, u, p)
  }
  /// Given an upper triangular matrix and a vector, compute the solution to the system of
  /// equations
  pub fn usolve(&self, b: &Vec4<T>) -> Vec4<T> {
    let &Matrix4(
      [Vec4([e00, _, _, _]), Vec4([e10, e11, _, _]), Vec4([e20, e21, e22, _]), Vec4([e30, e31, e32, e33])],
    ) = self;
    let &Vec4([b0, b1, b2, b3]) = b;
    let l = b3 / e33;
    let k = (b2 - l * e32) / e22;
    let j = (b1 - k * e21 - l * e31) / e11;
    let i = (b0 - j * e10 - k * e20 - l * e30) / e00;
    Vec4([i, j, k, l])
  }
  /// Given a lower triangular matrix and a vector, compute the solution to the system of
  /// equations
  pub fn lsolve(&self, b: &Vec4<T>) -> Vec4<T> {
    let &Matrix4(
      [Vec4([e00, e10, e20, e30]), Vec4([_, e11, e21, e31]), Vec4([_, _, e22, e32]), Vec4([_, _, _, e33])],
    ) = self;
    let &Vec4([b0, b1, b2, b3]) = b;
    let i = b0 / e00;
    let j = (b1 - i * e10) / e11;
    let k = (b2 - i * e20 - j * e21) / e22;
    let l = (b3 - i * e30 - j * e31 - k * e32) / e33;
    Vec4([i, j, k, l])
  }
  /// Solves for x in the linear system Ax = b;
  pub fn solve(lup: &(Self, Self, Self), b: &Vec4<T>) -> Vec4<T> {
    let (l, u, p) = lup;
    let b = p.vecmul(b);
    u.usolve(&l.lsolve(&b))
  }
  pub fn vec3mul(&self, v: &Vec3<T>) -> Vec3<T> {
    let &Matrix4([Vec4([e00, e01, e02, _]), Vec4([e10, e11, e12, _]), Vec4([e20, e21, e22, _]), _]) =
      self;
    let &Vec3(x, y, z) = v;
    Vec3(
      e00 * x + e01 * y + e02 * z,
      e10 * x + e11 * y + e12 * z,
      e20 * x + e21 * y + e22 * z,
    )
  }
}

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

/// Extends a 3x3 matrix such that it has a 4th column with last element 1.
impl<T: One + Zero + Copy> From<Matrix3<T>> for Matrix4<T> {
  fn from(v: Matrix3<T>) -> Self {
    let Matrix3([a, b, c]) = v;
    let o = T::zero();
    let l = T::one();
    Matrix4([a.extend(o), b.extend(o), c.extend(o), Vec4([o, o, o, l])])
  }
}

impl<T: Float> Matrix for Matrix4<T> {
  type Field = T;
  type Vector = Vec4<Self::Field>;
  /// Computes the determinant of this matrix
  fn det(&self) -> T { todo!() }
  /// transposes the matrix
  fn t(&self) -> Self {
    let &Matrix4(
      [Vec4([e00, e01, e02, e03]), Vec4([e10, e11, e12, e13]), Vec4([e20, e21, e22, e23]), Vec4([e30, e31, e32, e33])],
    ) = self;
    Matrix4([
      Vec4([e00, e10, e20, e30]),
      Vec4([e01, e11, e21, e31]),
      Vec4([e02, e12, e22, e32]),
      Vec4([e03, e13, e23, e33]),
    ])
  }
  fn vecmul(&self, o: &Vec4<T>) -> Vec4<T> {
    let &Matrix4([c1, c2, c3, c4]) = self;
    let &Vec4([a, b, c, d]) = o;
    c1 * a + c2 * b + c3 * c + c4 * d
  }
  /// Implement matrix multiplication
  fn matmul(&self, o: &Self) -> Self {
    let Matrix4([a, b, c, d]) = o;
    Matrix4([
      self.vecmul(a),
      self.vecmul(b),
      self.vecmul(c),
      self.vecmul(d),
    ])
  }
  /// Computes the inverse of this matrix if it exists
  fn inv(&self) -> Self {
    // taken from https://github.com/mrdoob/three.js/blob/master/src/math/Matrix4.js
    // which took it from http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.htm
    let &Matrix4(
      [Vec4([e11, e21, e31, e41]), Vec4([e12, e22, e32, e42]), Vec4([e13, e23, e33, e43]), Vec4([e14, e24, e34, e44])],
    ) = self;
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
    Matrix4([
      Vec4([o11, o21, o31, o41]),
      Vec4([o12, o22, o32, o42]),
      Vec4([o13, o23, o33, o43]),
      Vec4([o14, o24, o34, o44]),
    ])
  }
}

/*
// TODO look into using constant generics because this could replace all the code from above
// more easily maybe.
pub struct GenMatrix<T, const R: usize, const C: usize> {
  data: [[T; R]; C]
}

impl<T, const R: usize, const C: usize> GenMatrix {

}
*/

// Defines all element-wise operations between matrices of the same size
macro_rules! def_op {
  ($name: ident, $fn_name: ident, $op: tt) => {
    impl<T: $name> $name for Matrix2<T> {
      type Output = Matrix2<<T as $name>::Output>;
      fn $fn_name(self, o: Self) -> Self::Output {
        let Matrix2([a, b]) = self;
        let Matrix2([i, j]) = o;
        Matrix2([a $op i, b $op j])
      }
    }
    impl<T: $name> $name for Matrix3<T> {
      type Output = Matrix3<<T as $name>::Output>;
      fn $fn_name(self, o: Self) -> Self::Output {
        let Matrix3([a, b, c]) = self;
        let Matrix3([i, j, k]) = o;
        Matrix3([a $op i, b $op j, c $op k])
      }
    }
    impl<T: $name> $name for Matrix4<T> {
      type Output = Matrix4<<T as $name>::Output>;
      fn $fn_name(self, o: Self) -> Self::Output {
        let Matrix4([a, b, c, d]) = self;
        let Matrix4([i, j, k, l]) = o;
        Matrix4([a $op i, b $op j, c $op k, d $op l])
      }
    }
  }
}

// defines scalar operations between matrices
macro_rules! def_scalar_op {
  ($name: ident, $fn_name: ident, $op: tt) => {
    impl<T>$name<T> for Matrix2<T> where T: $name + Copy {
      type Output = Matrix2<<T as $name>::Output>;
      fn $fn_name(self, o: T) -> Self::Output {
        let Matrix2([a, b]) = self;
        Matrix2([a $op o, b $op o])
      }
    }
    impl<T>$name<T> for Matrix3<T> where T: $name + Copy {
      type Output = Matrix3<<T as $name>::Output>;
      fn $fn_name(self, o: T) -> Self::Output {
        let Matrix3([a, b, c]) = self;
        Matrix3([a $op o, b $op o, c $op o])
      }
    }
    impl<T>$name<T> for Matrix4<T> where T: $name + Copy {
      type Output = Matrix4<<T as $name>::Output>;
      fn $fn_name(self, o: T) -> Self::Output {
        let Matrix4([a, b, c, d]) = self;
        Matrix4([a $op o, b $op o, c $op o, d $op o])
      }
    }
  };
}

use std::ops::{Add, Div, Mul, Sub};
def_op!(Add, add, +);
def_op!(Sub, sub, -);
def_op!(Mul, mul, *);
def_op!(Div, div, /);

def_scalar_op!(Add, add, +);
def_scalar_op!(Sub, sub, -);
def_scalar_op!(Mul, mul, *);
def_scalar_op!(Div, div, /);
