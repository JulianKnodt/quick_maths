#![allow(incomplete_features)]
#![feature(const_generics, const_generic_impls_guard, maybe_uninit_uninit_array)]

// Linear algebra modules
/// Matrix definitions
pub mod mat;
#[doc(inline)]
pub use mat::*;

/// Quaternion definitions
pub mod quat;

/// Simple Ray definitions
pub mod ray;
#[doc(inline)]
pub use ray::*;

/// Vector definitions
pub mod vec;
#[doc(inline)]
pub use vec::*;

/// Invertible transforms based on
/// http://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Transformations.html
pub mod transform;
#[doc(inline)]
pub use transform::*;

/// Convenience re-export of num-traits
pub mod num;
#[doc(inline)]
pub use crate::num::*;

#[cfg(feature = "serde")]
pub mod serde;
