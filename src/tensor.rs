pub const fn product<const D: usize>(v: [usize; D]) -> usize {
  /*
  let mut out = 1;
  for i in 0..D {
    out *= v[i];
  }
  out
  */
  v.into_iter().fold(1, |acc, n| acc * n)
}

// This would be the structure of a tensor if it didn't ICE
pub struct Tensor<T, const D: usize, const DIMS: [usize; D]>([T; product(DIMS)]);

