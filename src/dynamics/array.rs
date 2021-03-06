use crate::Zero;
use num::Float;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Analog to numpy arrays, fully dynamic dimensions
#[derive(Debug, Clone)]
pub struct Array<T = f32> {
  // TODO add an offset to represent offsets from base?
  shape: Box<[u32]>,
  data: Box<[T]>,
}

impl<T: Zero> Array<T> {
  pub fn zeros(shape: impl Into<Box<[u32]>>) -> Self {
    let shape = shape.into();
    let data = (0..shape.iter().product())
      .map(|_| T::zero())
      .collect::<Vec<_>>()
      .into_boxed_slice();
    Self { shape, data }
  }
}

impl<T, const N: usize> From<[T; N]> for Array<T> {
  fn from(data: [T; N]) -> Self {
    Array {
      shape: Box::new([N as u32]),
      data: Box::new(data),
    }
  }
}

impl<T> From<T> for Array<T> {
  fn from(v: T) -> Self { Self::just(v) }
}

/*
pub struct ShapedIter<'a, T> {
  idx: Box<[u32]>,
  array: &'a Array<T>,
}

impl<'a, T> Iterator for ShapedIter<'a, T> {
  type Item = (Box<[u32], &'a T);
  fn next(&mut self) -> Option<Self::Item> {
    let next = self.array.get(&self.idx)?;
    for i in 0..self.array.shape.len() {
      self.idx[i] += 1;
      if self.idx[i] >= self.array.shape[i] {
        self.idx[i] = 0;
      } else {
        break
      }
    }
    Some((&self.idx, next))
  }
}

// This is a good idea but requires stacking
// impl<N: Into<Array<T>>, T, const N: usize> From<[N; N]>
*/

impl<T> Array<T> {
  pub fn just(v: T) -> Self {
    Self {
      shape: Box::new([1]),
      data: Box::new([v]),
    }
  }
  pub fn new(data: impl Into<Self>) -> Self { data.into() }
  pub fn len(&self) -> u32 { self.shape.iter().product() }
  pub fn is_empty(&self) -> bool { self.shape.iter().any(|&v| v == 0) }
  pub fn reshape(&mut self, i: impl Into<Box<[u32]>>) {
    let shape = i.into();
    assert!(shape.len() > 0, "Cannot reshape into empty array");
    assert_eq!(
      shape.iter().product::<u32>(),
      self.shape.iter().product(),
      "Cannot reshape {:?} into {:?}",
      self.shape,
      shape
    );
    self.shape = shape;
  }
  pub fn from_iter(shape: Box<[u32]>, data: impl IntoIterator<Item = T>) -> Self {
    assert!(shape.len() > 0, "Cannot reshape into empty array");
    let data = data.into_iter().collect::<Vec<_>>().into_boxed_slice();
    assert_eq!(
      shape.iter().product::<u32>() as usize,
      data.len(),
      "Shape and data have different dimensions"
    );
    Self { shape, data }
  }
  pub fn t(&mut self) { self.shape.reverse(); }
  pub fn iter(&self) -> impl Iterator<Item = &T> { self.data.iter() }
  fn index(&self, i: impl AsRef<[u32]>) -> u32 {
    let idx = i.as_ref();
    assert_eq!(idx.len(), self.shape.len());
    let mut out = 0;
    let mut curr_offset = 1;
    for (i, v) in idx.iter().enumerate() {
      out += v * curr_offset;
      curr_offset *= self.shape[i as usize];
    }
    out
  }
  pub fn get(&self, i: impl AsRef<[u32]>) -> Option<&T> { self.data.get(self.index(i) as usize) }
}

impl<T: Float> Array<T> {
  pub fn matmul(&self, o: &Self) -> Self {
    assert_eq!(self.shape.len(), 2);
    assert_eq!(o.shape.len(), 2);
    assert_eq!(self.shape[1], o.shape[0]);
    let mut out = Self::zeros([self.shape[0], o.shape[1]]);
    for i in 0..self.shape[0] {
      for j in 0..self.shape[1] {
        for k in 0..o.shape[1] {
          out[[i, k]] = out[[i, k]] + self[[i, j]] * self[[j, k]];
        }
      }
    }
    out
  }
  pub fn dot(&self, o: &Self) -> Self {
    match (self.shape.len(), o.shape.len()) {
      (1, 1) => {
        assert_eq!(self.shape, o.shape, "Invalid dimensions for inner product");
        Self::just(
          self
            .iter()
            .zip(o.iter())
            .fold(T::zero(), |acc, (&l, &r)| acc + l * r),
        )
      },
      (2, 1) => {
        assert_eq!(
          self.shape[1], o.shape[0],
          "Invalid dimensions for matrix-vector product",
        );
        let mut out = Self::zeros([self.shape[0]]);
        for i in 0..self.shape[0] {
          for j in 0..self.shape[1] {
            out[[i]] = out[[i]] + self[[i, j]] * o[[j]];
          }
        }
        out
      },
      (1, 2) => todo!(),
      (0, _) | (_, 0) => panic!("Invalid arrays passed to dot"),
      (_, _) => todo!(),
    }
  }
  // TODO implement einstein summation notation (einsum)
  // because it's a convenient way to represent things
}

impl<T, I: AsRef<[u32]>> Index<I> for Array<T> {
  type Output = T;
  fn index(&self, i: I) -> &Self::Output { &self.data[self.index(i) as usize] }
}

impl<T, I: AsRef<[u32]>> IndexMut<I> for Array<T> {
  fn index_mut(&mut self, i: I) -> &mut Self::Output { &mut self.data[self.index(i) as usize] }
}

macro_rules! impl_bin_op {
  ($(impl $typ: ident, $func: ident, $op: tt;)*) => {
    $(
      impl<T: $typ + Copy> $typ for Array<T> {
        type Output = Array<T::Output>;
        fn $func(self, o: Self) -> Self::Output {
          assert_eq!(self.shape, o.shape, "Currently only supports inputs of same size");
          let iter = self.iter().zip(o.iter()).map(|(&l, &r)| l $op r);
          Array::from_iter(self.shape.clone(), iter)
        }
      }
    )*
  };
}

// Implement binary operations
impl_bin_op!(
  impl Mul, mul, *;
  impl Add, add, +;
  impl Sub, sub, -;
  impl Div, div, /;
);
