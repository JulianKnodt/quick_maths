use num::Float;

/// Analog to numpy arrays, fully dynamic dimensions
#[derive(Debug)]
pub struct Array<T = f32> {
  dims: u32,
  shape: Box<[u32]>,
  data: Box<[T]>,
}

impl<T: Float> Array<T> {
  pub fn len(&self) -> u32 { self.shape.iter().product() }
  pub fn is_empty(&self) -> bool { self.shape.iter().any(|&v| v == 0) }
}
