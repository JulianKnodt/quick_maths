use crate::{ScalarFloat, Var};
use std::fmt::Debug;

pub trait Optimizer<'params>: Debug {
  type Parameters;
  fn new(params: Self::Parameters) -> Self;
  // Add to the list of parameters for this optimizer
  fn extend(&mut self, params: impl Iterator<Item = &'params mut Var>);
  // Apply the gradients being tracked to those on the tape.
  fn step(&'params self);
}

#[derive(Debug)]
struct Naive {
  learning_rate: ScalarFloat,
  parameters: Vec<*mut Var>,
}

impl<'params> Optimizer<'params> for Naive {
  type Parameters = ScalarFloat;
  fn new(learning_rate: Self::Parameters) -> Self {
    Naive {
      learning_rate,
      parameters: vec![],
    }
  }
  fn extend(&mut self, params: impl Iterator<Item = &'params mut Var>) {
    self.parameters.extend(params.map(|p| p as *mut Var));
  }
  fn step(&'params self) {
    for &i in self.parameters.iter() {
      let mut i = unsafe { *i };
      i.update(|v, grad| v + self.learning_rate * grad);
    }
  }
}
