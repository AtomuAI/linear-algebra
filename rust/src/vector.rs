// Copyright 2024 Bewusstsein Labs

mod test;

use bewusstsein::memory::stack::Stack;

use crate::tensor::Tensor;

struct Vector<T: Default + Clone + Copy + PartialEq, const N: usize>( Tensor<T, Stack<usize, 1>, Stack<T, N>> );
pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector4<T> = Vector<T, 4>;
pub type Vector5<T> = Vector<T, 5>;
pub type Vector6<T> = Vector<T, 6>;
pub type Vector7<T> = Vector<T, 7>;
pub type Vector8<T> = Vector<T, 8>;
pub type Vector9<T> = Vector<T, 9>;