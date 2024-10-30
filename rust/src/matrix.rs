// Copyright 2024 Bewusstsein Labs

mod test;

use bewusstsein::memory::stack::Stack;

use crate::tensor::Tensor;

struct Matrix<T: Default + Clone + Copy + PartialEq, const N: usize>( Tensor<T, Stack<usize, 2>, Stack<T, N>> );
pub type Matrix2x2<T> = Matrix<T, 4>;
pub type Matrix3x3<T> = Matrix<T, 9>;
pub type Matrix4x4<T> = Matrix<T, 16>;
pub type Matrix5x5<T> = Matrix<T, 25>;
pub type Matrix6x6<T> = Matrix<T, 36>;
pub type Matrix7x7<T> = Matrix<T, 49>;
pub type Matrix8x8<T> = Matrix<T, 64>;
pub type Matrix9x9<T> = Matrix<T, 81>;