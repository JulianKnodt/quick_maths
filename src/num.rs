use num::Float as NumFloat;
use std::fmt::Debug;

/// A trait which is both a float and debug.
pub trait Float: NumFloat + Debug {}
impl<T> Float for T where T: NumFloat + Debug {}

pub use num::{One, Zero};

cfg_if::cfg_if! {
  if #[cfg(feature="f64_default")] {
    pub type DefaultFloat=f64;
  } else {
    /// Define the default float as f32
    pub type DefaultFloat=f32;
  }
}
