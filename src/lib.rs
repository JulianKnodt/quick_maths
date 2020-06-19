#![allow(incomplete_features)]
#![feature(const_generics, const_generic_impls_guard, maybe_uninit_uninit_array)]

// Linear algebra modules
/// Matrix definitions
pub mod mat;
pub use mat::*;

/// Quaternion definitions
pub mod quat;

/// Simple Ray definitions
pub mod ray;
pub use ray::*;

/// Vector definitions
pub mod vec;
pub use vec::*;

/// Convenience trait to make all floats also debug
pub mod num;
pub use crate::num::*;
