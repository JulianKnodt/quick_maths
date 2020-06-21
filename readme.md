# Quick Maths

- Important note: this package is designed for nightly with const-generics.

A basic package for vector and matrix arithmetic which uses rust's unstable const generic
feature. I just need something that does math conveniently and don't want to have to deal with
NDArray or the like...

This package is defined for ergonomic usage and efficiency.

Example usage:

```rust
use quick_maths::{Vec3, Vector};

let a = Vec3::of(0.0);
let b = a + 1.0;
let c = b.sin();
let _dot = b.dot(&c);

// Taking elements out of the vector
let x = c.x();
let y = c.y();
let z = c.z();

// Or all at once
let Vector([i, j, k]) = c;
assert_eq!(x, i);
assert_eq!(y, j);
assert_eq!(z, k);
```

## Some things that make usage smoother

#### Default types

Unless you explicitly need to change it, the type is set to a reasonable default(f32). If you
want to change the type of the default, this can be changed by setting a compilation flag. This
allows for easy usage of the library without worrying unless you need to.

#### TODOs

- Add serde support
- Add benchmarks
