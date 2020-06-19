#![allow(incomplete_features)]
#![feature(const_generics, const_generic_impls_guard)]

// Linear algebra modules
/*
pub mod map;
pub mod mat;
pub mod vec;
*/
pub mod quat;
pub mod ray;
pub mod remat;
pub mod revec;

// Convenience trait to make all floats also debug
pub mod num;
