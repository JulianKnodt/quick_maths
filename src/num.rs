use num::{Float as NumFloat, One as NumOne, Zero as NumZero};
use std::fmt::Debug;

/// A trait which is both a float and debug.
pub trait Float: NumFloat + Debug {}
impl<T> Float for T where T: NumFloat + Debug {}

pub trait Zero: NumZero {}
impl<T> Zero for T where T: NumZero {}

pub trait One: NumOne {}
impl<T> One for T where T: NumOne {}
