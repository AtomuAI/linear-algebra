// Copyright 2024 Bewusstsein Labs

/*!
# Bewusstsein - Linear Algebra

A linear algebra library for Rust.

## Using the library

You will need the latest nightly build of Rust to use this library.
and the cargo package manager.

You will need to use the following feature flags in your code:

```rust
#![allow(incomplete_features)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
```

To use the library, add the following to your `Cargo.toml`:

```toml
[dependencies]
bewusstsein = "*" # or the latest version
```
*/

#![allow(incomplete_features)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

pub mod traits;
pub mod ops;
pub mod shape;
pub mod tensor;
pub mod slice;
//pub mod slice;
pub mod vector;
pub mod matrix;
pub mod polynomial;
pub mod generalized_polynomial;
pub mod rotation;
//pub mod quaternion;
