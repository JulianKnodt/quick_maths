use crate::{
  num::Float,
  revec::{Vec2, Vec3},
};
use std::{
  marker::PhantomData,
  ops::{Add, Mul, Neg},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Ray<T, V = Vec3<T>> {
  pub pos: V,
  pub dir: V,
  // It makes it much cleaner to declare structs using this
  phantom: PhantomData<T>,
}

impl<T, V> Ray<T, V> {
  /// Returns a new ray with the given position and direction
  pub fn new(pos: V, dir: V) -> Self {
    Ray {
      pos,
      dir,
      phantom: PhantomData,
    }
  }
}
impl<T: Float, V> Ray<T, V>
where
  V: Add<Output = V> + Mul<T, Output = V> + Copy,
{
  /// Returns the position along a ray that corresponds to some parameter T
  pub fn at(&self, t: T) -> V { self.pos + self.dir * t }
  pub fn step(&mut self, t: T) { self.pos = self.pos + (self.dir * t) }
}

impl<T: Float> Ray<T, Vec2<T>> {
  /// Sets the length of this ray to the given amount
  pub fn set_length(&mut self, t: T) { self.dir = self.dir.norm() * t; }
}

impl<T, V> Ray<T, V>
where
  V: Neg<Output = V> + Copy,
{
  /// Flips the direction of this ray
  pub fn flip(&self) -> Self { Ray::new(self.pos, -self.dir) }
}
