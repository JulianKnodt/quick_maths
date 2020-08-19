use num::Float as NumFloat;
use std::fmt::Debug;

/// A trait which is both a float and debug.
pub trait Float: NumFloat + Debug {}
impl<T> Float for T where T: NumFloat + Debug {}

pub use num::{One, Zero};

// These are the default scalars used throughout the library.
cfg_if::cfg_if! {
  if #[cfg(feature="f64_default")] {
    pub type DefaultFloat=f64;
  } else {
    /// The default float used by all structures.
    /// When constructing a vector and omitting the type, this will be used.
    pub type DefaultFloat=f32;
  }
}

// TODO wrap scalars in some higher order type with compile flags.
