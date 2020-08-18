use crate::{
  num::{DefaultFloat, Float},
  vec::Vector,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ray<const N: usize, T = DefaultFloat> {
  pub pos: Vector<N, T>,
  pub dir: Vector<N, T>,
}

/// 3D ray with default float type.
pub type Ray3<T = DefaultFloat> = Ray<3, T>;

impl<const N: usize, T> Ray<N, T> {
  /// Returns a new ray with the given position and direction
  pub fn new(pos: Vector<N, T>, dir: Vector<N, T>) -> Self { Ray { pos, dir } }
}
impl<const N: usize, T: Float> Ray<N, T> {
  /// Returns the position along a ray that corresponds to some parameter T
  pub fn at(&self, t: T) -> Vector<N, T> { self.pos + self.dir * t }
  pub fn step(&mut self, t: T) { self.pos = self.pos + (self.dir * t) }
  /// Flips the direction of this ray
  pub fn flip(&self) -> Self { Ray::new(self.pos, -self.dir) }
  /// Sets the length of this ray to the given amount
  pub fn set_length(&mut self, t: T) { self.dir = self.dir.norm() * t; }
}
