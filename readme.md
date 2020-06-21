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
allows for easy usage of the library without worrying unless you need to. Thus, if you need to
store a vector or matrix in a struct you don't need to specify the type usually:

```rust
struct Circle {
  // implicitly f32 below
  center: Vec2,
  radius: f32,
}
```
If it's ever needed to switch all defaults from f32 to f64, then you can do that by specifying
the dependency with the `default_f64` feature.

#### One struct for Vectors & Matrices API

One ergonomic pain-point with other APIs is that they tend to use multiple structs because of a
lack of const-generics. Even though they often have traits unifying these structs, it turns the
code into trait spaghetti. Thus, it is much easier in terms of both usage and authoring to have
a single struct. The only issue with this is the reliance on an unstable nightly feature, but it
leads to much easier to read code.

#### Looks like a Float

Element-wise functions are all implemented. That means that vectors essentially look identical
to floats. This is crucial for ergonomic usage. Numpy works hard to create a unifying API over
every level of tensor, and this is similar in that regard.


## What is missing

It's still necessary to have different types of multiplication for different dimensions, but it
might be convenient to add in general purpose tensor products.

Also convolutions but who knows if that's necessary.

#### TODOs

- Add benchmarks
