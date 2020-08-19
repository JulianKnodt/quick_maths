use crate::{Ray, ScalarFloat, Var, Vector};
use quickcheck::{Arbitrary, Gen};

impl<A: Arbitrary, const N: usize> Arbitrary for Vector<N, A> {
  fn arbitrary<G: Gen>(g: &mut G) -> Self { Vector::with(|_| A::arbitrary(g)) }
}

impl<A: Arbitrary, const N: usize> Arbitrary for Ray<A, N> {
  fn arbitrary<G: Gen>(g: &mut G) -> Self { Ray::new(Vector::arbitrary(g), Vector::arbitrary(g)) }
}

impl Arbitrary for Var {
  fn arbitrary<G: Gen>(g: &mut G) -> Self { Var::new(ScalarFloat::arbitrary(g)) }
}
