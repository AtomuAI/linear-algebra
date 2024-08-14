// Copyright 2024 Bewusstsein Labs

use crate::tensor::Tensor;

pub type Matrix<T> = Tensor<T, 2>;
pub type Vector<T> = Tensor<T, 1>;