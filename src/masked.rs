use crate::Vector;
use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct MaskedVector<'a, T, const N: usize> {
  vec: &'a mut Vector<T, N>,
  mask: Vector<bool, N>,
}

impl<T, F, const N: usize> Index<F> for MaskedVector<'_, T, N>
where
  F: FnMut(&T) -> bool,
{
  type Output = Self;
  fn index(&self, _: F) -> &Self::Output { &self }
}

impl<T, F, const N: usize> Index<F> for &'_ mut MaskedVector<'_, T, N>
where
  F: FnMut(&T) -> bool,
{
  type Output = Self;
  fn index(&self, _: F) -> &Self::Output { &self }
}

impl<T, F, const N: usize> IndexMut<F> for MaskedVector<'_, T, N>
where
  F: FnMut(&T) -> bool,
{
  fn index_mut(&mut self, mut f: F) -> &mut Self::Output {
    for i in 0..N {
      self.mask[i] &= f(&self.vec[i]);
    }
    self
  }
}

impl<T, F, const N: usize> IndexMut<F> for &'_ mut MaskedVector<'_, T, N>
where
  F: FnMut(&T) -> bool,
{
  fn index_mut(&mut self, mut f: F) -> &mut Self::Output {
    for i in 0..N {
      self.mask[i] &= f(&self.vec[i]);
    }
    self
  }
}

impl<T: Copy, const N: usize> Vector<T, N> {
  pub fn mask(&mut self) -> MaskedVector<'_, T, N> { MaskedVector::new(self, Vector::of(true)) }
}

impl<'a, T, const N: usize> MaskedVector<'a, T, N> {
  fn new(vec: &'a mut Vector<T, N>, mask: Vector<bool, N>) -> Self { Self { vec, mask } }
}
