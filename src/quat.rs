use crate::{
  num::Float,
  vec::{Vec3, Vec4, Vector},
  DefaultFloat,
};

/// Quats are Vec4's with implicit imaginary first three terms.
pub type Quat<T = DefaultFloat> = Vec4<T>;

impl<T: Float> Quat<T> {
  pub fn conj(mut self) -> Self {
    self[0] = -self[0];
    self[1] = -self[1];
    self[2] = -self[2];
    self
  }
  pub fn is_unit(&self) -> bool { self.sqr_magn().is_one() }
  /// Encodes a scaling factor into the quaternion
  pub fn scale(&self, factor: T) -> Self { *self * factor.sqrt() }
  /// Returns a quaternion which is a rotation in the 3 dimensions given
  pub fn rot(along: &Vec3<T>) -> Self {
    let two = T::one() + T::one();
    let (Vector([sx, sy, sz]), Vector([cx, cy, cz])) = (*along / two).sin_cos();
    Quat::new(
      sx * cy * cz - cx * sy * sz,
      cx * sy * cz + sx * cy * sz,
      cx * cy * sz - sx * sy * cz,
      cx * cy * cz + sx * sy * sz,
    )
  }
  // TODO investigate slerping
  pub fn quat_mul(self, o: Self) -> Self {
    let Vector([x, y, z, w]) = self;
    let Vector([i, j, k, l]) = o;
    Quat::new(
      w * i + l * x + y * k - z * j,
      w * j + l * y + z * i - k * x,
      w * k + l * z + j * x - i * y,
      w * l - x * i - y * j - z * k,
    )
  }
}

impl<T: Float> Vec3<T> {
  pub fn to_quat(&self) -> Quat<T> {
    let [x, y, z] = self.0;
    Quat::new(x, y, z, T::zero())
  }
  pub fn apply_quat(&self, q: &Quat<T>) -> Self {
    let p = self.to_quat();
    let [x, y, z, _] = ((*q * p) * q.conj()).0;
    Vec3::new(x, y, z)
  }
}
