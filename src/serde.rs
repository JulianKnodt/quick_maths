use crate::{Matrix, Vector};
use serde::{
  de::{Error, SeqAccess, Visitor},
  ser::SerializeTuple,
  Deserialize, Deserializer, Serialize, Serializer,
};
use std::{
  fmt,
  marker::PhantomData,
  mem::{self, MaybeUninit},
};

impl<T, const N: usize> Serialize for Vector<N, T>
where
  T: Serialize,
{
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_tuple(N)?;
    for i in 0..N {
      seq.serialize_element(&self[i])?;
    }
    seq.end()
  }
}

impl<T, const M: usize, const N: usize> Serialize for Matrix<M, N, T>
where
  T: Serialize,
{
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_tuple(N)?;
    for i in 0..N {
      seq.serialize_element(&self[i])?;
    }
    seq.end()
  }
}

struct ArrayVisitor<A>(PhantomData<A>);

impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<[T; N]>
where
  T: Deserialize<'de>,
{
  type Value = [T; N];
  fn expecting(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
    fmt.write_fmt(format_args!("Array of length {}", N))
  }
  fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
  where
    A: SeqAccess<'de>, {
    let mut to = MaybeUninit::<[T; N]>::uninit();
    let top: *mut T = unsafe { mem::transmute(&mut to) };
    for i in 0..N {
      if let Some(element) = seq.next_element()? {
        unsafe {
          top.add(i).write(element);
        }
      } else {
        return Err(A::Error::invalid_length(i, &self));
      }
    }
    unsafe { Ok(to.assume_init()) }
  }
}

impl<'de, T, const N: usize> Deserialize<'de> for Vector<N, T>
where
  T: Deserialize<'de>,
{
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>, {
    deserializer
      .deserialize_tuple(N, ArrayVisitor::<[T; N]>(PhantomData))
      .map(Vector)
  }
}

impl<'de, T, const M: usize, const N: usize> Deserialize<'de> for Matrix<M, N, T>
where
  T: Deserialize<'de>,
{
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>, {
    deserializer
      .deserialize_tuple(N, ArrayVisitor::<_>(PhantomData))
      .map(Vector)
      .map(Matrix)
  }
}
