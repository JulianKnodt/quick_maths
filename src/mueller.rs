use crate::{DefaultFloat, Float, Mat4, Matrix, One, Vec4};

/// Operator of an optical element defined as a function of
/// direction of propogation & wavelength.
pub type Mueller<T = DefaultFloat> = Mat4<T>;

impl<T: Float> Mueller<T> {
  /// A non-polarizing mueller matrix which does not affect incoming light.
  pub fn non_polarizing() -> Self { Self::one() }
  /// Returns a depolarizing matrix that scales intensity by the scale.
  pub fn depolarizer(intensity_scale: T) -> Self {
    let o = T::zero();
    Self::from_diag(Vec4::new(intensity_scale, o, o, o))
  }
  /// Corresponds to a scaling of polarization
  pub fn absorber(by: T) -> Self { Self::from_diag(Vec4::of(by)) }
  pub fn linear_polarizer(alpha: T) -> Self {
    let v = alpha / (T::one() + T::one());
    let o = T::zero();
    Matrix(Vec4::new(
      Vec4::new(v, v, o, o),
      Vec4::new(v, v, o, o),
      Vec4::new(o, o, o, o),
      Vec4::new(o, o, o, o),
    ))
  }
  fn circular_retarder_precomputed(sin: T, cos: T) -> Self {
    let o = T::zero();
    let l = T::one();
    Matrix(Vec4::new(
      Vec4::new(l, o, o, o),
      Vec4::new(o, cos, sin, o),
      Vec4::new(o, -sin, cos, o),
      Vec4::new(o, o, o, l),
    ))
  }
  pub fn circular_retarder(delta: T) -> Self {
    let (sin, cos) = delta.sin_cos();
    Self::circular_retarder_precomputed(sin, cos)
  }
  pub fn rotated_element(&self, theta: T) -> Self {
    let rot = Self::circular_retarder(theta + theta);
    rot.matmul(&self.matmul(&rot.t()))
  }
}
