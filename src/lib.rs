#![allow(incomplete_features)]
#![feature(const_generics, const_generic_impls_guard)]

// Linear algebra modules
pub mod mat;
pub mod quat;
pub mod ray;
pub mod vec;

// Convenience trait to make all floats also debug
pub mod num;
