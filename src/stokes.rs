use crate::{DefaultFloat, Float, Vec4, Vector};

pub type Stokes<T = DefaultFloat> = Vec4<T>;

impl<T: Float> Stokes<T> {
  pub fn unpolarized(intensity: T) -> Self { Vec4::new(intensity, T::zero(), T::zero(), T::zero()) }
  pub fn from_flux_measures(p_hori: T, p_vert: T, p_45: T, p_135: T, p_r: T, p_l: T) -> Self {
    Vec4::new(p_hori + p_vert, p_hori - p_vert, p_45 - p_135, p_r - p_l)
  }
  pub fn degree_of_polarization(&self) -> T {
    let &Vector([s0, s1, s2, s3]) = self;
    (s1 * s1 + s2 * s2 + s3 * s3).sqrt() / s0
  }
  pub fn degree_of_linear_polarization(&self) -> T {
    let &Vector([s0, s1, s2, _]) = self;
    (s1 * s1 + s2 * s2).sqrt() / s0
  }
  pub fn degree_of_circular_polarization(&self) -> T {
    let &Vector([s0, _, _, s3]) = self;
    s3 / s0
  }
  /// Decomposes this stokes vector into a polarized and unpolarized component respectively.
  pub fn decompose(&self) -> (Self, Self) {
    let dop = self.degree_of_polarization();
    let &Vector([s0, s1, s2, s3]) = self;
    let l = T::one();
    (
      Vec4::new(s0 * dop, s1, s2, s3),
      Self::unpolarized((l - dop) * s0),
    )
  }

  /// Ratio between length of axes of polarization
  pub fn eccentricity(&self) -> T { (T::one() - self.ellipticity()).sqrt() }
  /// Aspect ratio of the ellipse (minor to major axis)
  pub fn ellipticity(&self) -> T {
    let &Vector([s0, s1, s2, s3]) = self;
    s3 / (s0 + (s1 * s1 + s2 * s2).sqrt())
  }
  /// Orientation of the longer axis upwards from the x-axis
  pub fn azimuth(&self) -> T {
    let &Vector([_, s1, s2, _]) = self;
    s2.atan2(s1) / (T::one() + T::one())
  }
}
