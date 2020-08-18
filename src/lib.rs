#![allow(incomplete_features)]
#![feature(const_generics, maybe_uninit_uninit_array, new_uninit, array_map)]

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

#[cfg(feature = "quickcheck")]
pub mod quickcheck;

/*
/// Convenience functions for Mueller matrices
#[cfg(feature = "mueller")]
pub mod mueller;

/// Convenience functions for stokes vectors
#[cfg(feature = "stokes")]
pub mod stokes;

#[cfg(feature = "masked")]
pub mod masked;
*/

/// Dynamic vectors and matrices.
// TODO move this to the dynamics folder
mod dynamics;
pub use dynamics::{Array, DynTensor};
