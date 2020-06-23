use crate::{Ray, Vector};
use quickcheck::{Arbitrary, Gen};
use std::array::LengthAtMost32;

impl<A: Arbitrary + Copy, const N: usize> Arbitrary for Vector<A, N>
where
  [A; N]: LengthAtMost32,
{
  fn arbitrary<G: Gen>(g: &mut G) -> Self { Vector::with(|_| A::arbitrary(g)) }
}

impl Arbitrary for Ray {
  fn arbitrary<G: Gen>(g: &mut G) -> Self {
    Ray::new(Vector::arbitrary(g), Vector::arbitrary(g).norm())
  }
}
