use crate::{Matrix, Vector};
use num::Float;
use std::mem::MaybeUninit;

#[derive(Debug, Clone)]
/// A Tensor which only specifies the number of dimensions
pub struct DynTensor<T = f32, const D: usize> {
  shape: Vector<usize, D>,
  // strides: Vector<usize, D>,
  data: Box<[T]>,
}

impl<T, const D: usize> DynTensor<T, D> {
  pub fn len(&self) -> usize { self.shape.0.iter().product() }
  pub fn is_empty(&self) -> bool { self.shape.0.iter().any(|&v| v == 0) }
  pub fn shape(&self) -> &[usize; D] { &self.shape.0 }
  pub fn iter(&self) -> impl Iterator<Item = &T> { self.data.iter() }
  pub fn reshape<const D2: usize>(self, new_shape: Vector<usize, D2>) -> DynTensor<T, D2> {
    assert_eq!(
      new_shape.0.iter().product::<usize>(),
      self.shape.0.iter().product::<usize>(),
      "Cannot reshape {:?} into {:?}: Unequal number of elements",
      self.shape,
      new_shape
    );
    DynTensor {
      shape: new_shape,
      data: self.data,
    }
  }
  pub fn from_iter(shape: [usize; D], iter: impl Iterator<Item = T>) -> Self {
    let size = shape.iter().product();
    let mut data = Box::new_uninit_slice(size);
    let mut count = 0;
    for (i, v) in iter.enumerate() {
      data[i] = MaybeUninit::new(v);
      count += 1;
    }
    assert_eq!(count, size, "Iterator passed did not have enough elements");
    let data = unsafe { data.assume_init() };
    let shape = Vector(shape);
    Self { shape, data }
  }
}

trait TensorMul<T, const I: usize> {
  type Output;
  fn dot(&self, rh: DynTensor<T, I>) -> Self::Output;
}

impl<T: Float> TensorMul<T, 1> for DynTensor<T, 1> {
  type Output = T;
  fn dot(&self, rhs: DynTensor<T, 1>) -> Self::Output {
    assert_eq!(
      self.shape, rhs.shape,
      "Length Mismatch between dot product of dyn vectors"
    );
    self
      .data
      .iter()
      .zip(rhs.data.iter())
      .map(|(&l, &r)| l * r)
      .fold(T::zero(), |acc, n| acc + n)
  }
}

impl<T: Float> TensorMul<T, 1> for DynTensor<T, 2> {
  type Output = T;
  fn dot(&self, rhs: DynTensor<T, 1>) -> Self::Output {
    assert_eq!(
      self.shape[1], rhs.shape[0],
      "Length Mismatch between {:?} {:?}",
      self.shape, rhs.shape
    );
    todo!();
  }
}

impl<T, const N: usize> Vector<T, N> {
  pub fn dyn_tensor(self) -> DynTensor<T, 1> {
    let shape = Vector([N]);
    let data = Box::new(self.0);
    DynTensor { shape, data }
  }
}

impl<T: Copy, const N: usize, const M: usize> Matrix<T, M, N> {
  // currently this still allocates even though we're bringing in exactly enough memory in the
  // correct format. I'm not sure how to fix this.
  pub fn dyn_tensor(self) -> DynTensor<T, 2> {
    let shape = Vector([M, N]);
    let mut data = Box::new_uninit_slice(M * N);
    let data = unsafe {
      for i in 0..N {
        for j in 0..M {
          data[i * M + j] = MaybeUninit::new(self[i][j]);
        }
      }
      data.assume_init()
    };
    DynTensor { shape, data }
  }
}

macro_rules! impl_bin_op {
  ($(impl $typ: ident, $func: ident, $op: tt;)*) => {
    $(
      impl<T: $typ + Copy, const D: usize> $typ for DynTensor<T, D> {
        type Output = DynTensor<T::Output, D>;
        fn $func(self, o: Self) -> Self::Output {
          assert_eq!(self.shape, o.shape, "Currently only supports inputs of same size");
          let iter = self.iter().zip(o.iter()).map(|(&l, &r)| l $op r);
          DynTensor::from_iter(self.shape.0, iter)
        }
      }
    )*
  };
}

use std::ops::{Add, Div, Mul, Sub};
// Implement binary operations
impl_bin_op!(
  impl Mul, mul, *;
  impl Add, add, +;
  impl Sub, sub, -;
  impl Div, div, /;
);

/*
pub struct SingleStridedSlice<'a, T> {
  slice: &'a [T],
  strides: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StridedSlice<'a, T> {
  slice: &'a [T],
  strides: &'a [usize],
}

impl<'a, T> StridedSlice<'a, T> {
  pub fn iter(&'a self) -> Box<dyn Iterator<Item = &T> + 'a> {
    self
      .strides
      .iter()
      .fold(Box::new(self.slice.iter()), |acc, &n| {
        Box::new(acc.step_by(n))
      })
  }
}

fn strides<const D: usize>(shape: Vector<usize, D>) -> Vector<usize, D> {
  let mut curr = 1;
  shape.apply_fn(|v| {
    let prev = curr;
    curr = curr * v;
    prev
  })
}
*/

#[test]
fn example() {
  use crate::{Mat3, Vec3};
  let v = Vec3::of(3.0);
  let m = Mat3::scale(&Vec3::of(3.0));
  let t = m.dyn_tensor();
  // check that allocation doesn't break everything
}
