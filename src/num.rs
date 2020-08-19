use num::Float as NumFloat;
use std::fmt::Debug;

/// A trait which is both a float and debug.
pub trait Float: NumFloat + Debug {}
impl<T> Float for T where T: NumFloat + Debug {}

pub use num::{One, Zero};

// These are the default scalars used throughout the library.
cfg_if::cfg_if! {
  if #[cfg(feature="f64_default")] {
    /// Default Scalar float = f64
    pub type ScalarFloat = f64;
  } else {
    /// Default Scalar float is f32
    pub type ScalarFloat = f32;
  }
}

cfg_if::cfg_if! {
  if #[cfg(feature="autodiff")] {
    /// Enable auto differentiation in all structs
    pub type DefaultFloat = crate::autodiff::Var;
  } else {
    /// No autodifferentiation is tracked.
    pub type DefaultFloat = ScalarFloat;
  }
}
