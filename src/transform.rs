use crate::{num::DefaultFloat, Float, Mat2, Mat3, Mat4, Matrix, Ray, Vec2, Vec3, Vec4, Vector};
use num::One;
use std::ops::Mul;

/// Transform type which represents an easily invertible operator.
/// i.e. rotation in 3D, translation, etc.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform<T = DefaultFloat, const N: usize> {
  /// Forward transformation
  pub fwd: Matrix<T, N, N>,
  /// Inverted transformation
  pub bkwd: Matrix<T, N, N>,
}

/// 3D transformation, where 4 represents the dimension of the matrix used.
pub type Transform4<T = DefaultFloat> = Transform<T, 4>;
/// 2D transformation, where 3 represents the dimension of the matrix used.
pub type Transform3<T = DefaultFloat> = Transform<T, 3>;

impl<T: Float, const N: usize> Transform<T, N> {
  pub fn identity() -> Self {
    Self {
      fwd: Matrix::one(),
      bkwd: Matrix::one(),
    }
  }
  pub fn inv(&self) -> Self {
    let &Transform { fwd, bkwd } = self;
    Self {
      fwd: bkwd,
      bkwd: fwd,
    }
  }
}

impl<T: Float, const N: usize> Mul for Transform<T, N> {
  type Output = Transform<T, N>;
  fn mul(self, o: Self) -> Self::Output {
    Self::Output {
      fwd: self.fwd.matmul(&o.fwd),
      bkwd: o.bkwd.matmul(&self.bkwd),
    }
  }
}

impl<T: Float> Transform4<T> {
  pub fn new(m: Mat4<T>) -> Self {
    Self {
      bkwd: m.inv(),
      fwd: m,
    }
  }
  pub fn scale(by: Vec3<T>) -> Self {
    Self {
      fwd: Mat3::scale(&by).ixtend(),
      bkwd: Mat3::scale(&by.recip()).ixtend(),
    }
  }
  pub fn rot(axis: Vec3<T>, theta: T) -> Self {
    let fwd = Mat3::rot(&axis, theta.cos()).ixtend();
    Self { bkwd: fwd.t(), fwd }
  }
  pub fn translate(by: Vec3<T>) -> Self {
    Self {
      fwd: Mat4::translate(by),
      bkwd: Mat4::translate(-by),
    }
  }
  pub fn orthographic(z_near: T, z_far: T) -> Self {
    assert!(
      z_far > z_near,
      "Expected z_far > z_near, got far: {:?}, near: {:?}",
      z_far,
      z_near
    );
    let l = T::one();
    let o = T::zero();
    Self::scale(Vec3::new(l, l, l / (z_far - z_near))) * Self::translate(Vec3::new(o, o, -z_near))
  }
  /// Perspective transformation
  pub fn perspective(fov: T, near: T, far: T) -> Self {
    let s1 = far / (far - near);
    let s2 = -near * s1;
    let l = T::one();
    let o = T::zero();
    let m = Mat4::new(
      Vec4::new(l, o, o, o),
      Vec4::new(o, l, o, o),
      Vec4::new(o, o, s1, l),
      Vec4::new(o, o, s2, o),
    );
    let inv_tan_angle = (fov.to_radians() / (l + l)).tan().recip();
    Self::scale(Vec3::new(inv_tan_angle, inv_tan_angle, l)) * Self::new(m)
  }
  /// A transformation that orients something as if it's looking towards "at" from "pos".
  pub fn look_at(pos: Vec3<T>, at: Vec3<T>, up: Vec3<T>) -> Self {
    let dir = (at - pos).norm();
    // do a couple of extra norm calls here, maybe can remove them but nice to be safe
    let right = dir.cross(&up.norm()).norm();
    let up = dir.cross(&right).norm();
    let cam_to_world = Mat4::new(right.zxtend(), up.zxtend(), dir.zxtend(), pos.homogeneous());
    Self {
      bkwd: cam_to_world.inv(),
      fwd: cam_to_world,
    }
  }
  /// Applies this transformation as if to a point.
  pub fn apply_point(&self, pt: &Vec3<T>) -> Vec3<T> {
    self.fwd.dot(&pt.homogeneous()).homogenize()
  }
  /// Applies this transformation as if applying it to a vector
  /// What this means operationally is no transformations.
  pub fn apply_vec(&self, vec: &Vec3<T>) -> Vec3<T> { self.fwd.qdot(&vec).reduce() }
  /// Applies this transformation as if applying it to a (surface) normal vector
  pub fn apply_normal(&self, n: &Vec3<T>) -> Vec3<T> {
    let Matrix(Vector(
      [Vector([e00, e10, e20, _]), Vector([e01, e11, e21, _]), Vector([e02, e12, e22, _]), _],
    )) = self.bkwd;
    let &Vector([x, y, z]) = n;
    Vec3::new(
      e00 * x + e10 * y + e20 * z,
      e01 * x + e11 * y + e21 * z,
      e02 * x + e12 * y + e22 * z,
    )
  }
  /// Applies this transform to a ray (convenience for applying point to origin and vector to
  /// direction)
  pub fn apply_ray(&self, r: &Ray<T>) -> Ray<T> {
    Ray::new(self.apply_point(&r.pos), self.apply_vec(&r.dir))
  }
}

impl<T: Float> Transform3<T> {
  pub fn scale(by: Vec2<T>) -> Self {
    Self {
      fwd: Mat3::scale(&by.zxtend()),
      bkwd: Mat3::scale(&by.recip().zxtend()),
    }
  }
  pub fn rot(theta: T) -> Self {
    let fwd = Mat2::rot(theta).ixtend();
    Self { bkwd: fwd.t(), fwd }
  }
  pub fn translate(by: Vec2<T>) -> Self {
    Self {
      fwd: Mat3::translate(&by),
      bkwd: Mat3::translate(&-by),
    }
  }
  pub fn apply_point(&self, v: Vec2<T>) -> Vec2<T> { self.fwd.dot(&v.homogeneous()).homogenize() }
}
