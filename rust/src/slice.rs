// Copyright 2024 Bewusstsein Labs

use std::ops::{ Index, IndexMut };

use crate::tensor::{ Tensor, Error };

pub struct Slice<'a, T, const N: usize> {
    tensor: &'a mut Tensor<T, N>,
    start: [usize; N],
    end: [usize; N],
    strides: [usize; N],
}

impl<'a, T, const N: usize> Slice<'a, T, N>
where
    T: Default + Clone + PartialEq,
{
    pub fn new(tensor: &'a mut Tensor<T, N>, start: [usize; N], end: [usize; N], strides: [usize; N]) -> Self {
        Slice {
            tensor,
            start,
            end,
            strides,
        }
    }

    pub fn dim( &self ) -> usize {
        self.tensor.dim()
    }

    pub fn size( &self ) -> usize {
        let mut size = 1;
        for i in 0..N {
            size *= (self.end[i] - self.start[i]) / self.strides[i];
        }
        size
    }

    pub fn shape( &self ) -> [ usize; N ] {
        let mut shape = [0; N];
        for i in 0..N {
            shape[i] = (self.end[i] - self.start[i]) / self.strides[i];
        }
        shape
    }

    pub fn tensor(&self) -> Result<Tensor<T, N>, Error> {
        let shape = self.shape();
        let mut tensor = Tensor::new( shape );
        for i in 0..self.size() {
            let mut index = [0; N];
            let mut stride = 1;
            for j in 0..N {
                index[j] = (i / stride) % shape[j];
                stride *= shape[j];
            }
            tensor[index] = self[index].clone();
        }
        Ok( tensor )
    }

    pub fn fill(&mut self, value: T) {
        let shape = self.shape();
        for i in 0..self.size() {
            let mut index = [0; N];
            let mut stride = 1;
            for j in 0..N {
                index[j] = (i / stride) % shape[j];
                stride *= shape[j];
            }
            self[index] = value.clone();
        }
    }

    pub fn zero(&mut self) {
        self.fill(T::default());
    }

    fn calculate_flat_index(&self, index: &[usize; N]) -> usize {
        let mut flat_index = 0;
        let mut stride = 1;
        for (i, &dim_index) in index.iter().rev().enumerate() {
            let tensor_index = self.start[N - 1 - i] + dim_index * self.strides[N - 1 - i];
            flat_index += tensor_index * stride;
            stride *= self.tensor.shape()[N - 1 - i];
        }
        flat_index
    }
}

// Dimensional Indexing
impl<'a, T, const N: usize> Index<[usize; N]> for Slice<'a, T, N>
where
    T: Default + Clone + PartialEq,
{
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        let flat_index = self.calculate_flat_index(&index);
        &self.tensor[flat_index]
    }
}

impl<'a, T, const N: usize> IndexMut<[usize; N]> for Slice<'a, T, N>
where
    T: Default + Clone + PartialEq,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let flat_index = self.calculate_flat_index(&index);
        &mut self.tensor[flat_index]
    }
}

// Flat Indexing
impl<'a, T, const N: usize> Index<usize> for Slice<'a, T, N>
where
    T: Default + Clone + PartialEq,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.tensor[index]
    }
}

impl<'a, T, const N: usize> IndexMut<usize> for Slice<'a, T, N>
where
    T: Default + Clone + PartialEq,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.tensor[index]
    }
}