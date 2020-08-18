use crate::{Ray, Vector};
use quickcheck::{Arbitrary, Gen};

impl<A: Arbitrary, const N: usize> Arbitrary for Vector<N, A> {
  fn arbitrary<G: Gen>(g: &mut G) -> Self { Vector::with(|_| A::arbitrary(g)) }
}

impl<const N: usize> Arbitrary for Ray<N> {
  fn arbitrary<G: Gen>(g: &mut G) -> Self {
    Ray::new(Vector::arbitrary(g), Vector::arbitrary(g).norm())
  }
}
