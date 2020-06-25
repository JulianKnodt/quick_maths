use crate::{Ray, Vector};
use quickcheck::{Arbitrary, Gen};

impl<A: Arbitrary, const N: usize> Arbitrary for Vector<A, N> {
  fn arbitrary<G: Gen>(g: &mut G) -> Self { Vector::with(|_| A::arbitrary(g)) }
}

impl Arbitrary for Ray {
  fn arbitrary<G: Gen>(g: &mut G) -> Self {
    Ray::new(Vector::arbitrary(g), Vector::arbitrary(g).norm())
  }
}
