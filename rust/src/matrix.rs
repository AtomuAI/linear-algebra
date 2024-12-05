// Copyright 2024 Bewusstsein Labs

//mod test;

use std::ops::{ Deref, DerefMut };

use memory::{
    stack::Stack,
    MemoryTraits,
    MemoryType,
    Memory
};

use crate::{
    shape::Shape,
    tensor::{ Tensor, TensorTraits }
};

#[derive( Clone, Default )]
pub struct Matrix<T, const N: usize, const M: usize>( Tensor<T, 2, Stack<{ N * M }>> ) where T: 'static + Copy + Default, [(); N * M]:;

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    T: 'static + Copy + Default,
    [(); N * M]:
{
    pub fn new() -> Self {
        Self( Tensor::<T, 2, Stack<{ N * M }>>::new( Shape::<2>::from( [N, M] ) ) )
    }

    pub fn take( src: [T; N * M] ) -> Self {
        Self( Tensor::<T, 2, Stack<{ N * M }>>::take( src ) )
    }
}

impl<T, const N: usize, const M: usize> Deref for Matrix<T, N, M>
where
    T: 'static + Copy + Default,
    [(); N * M]:
{
    type Target = Tensor<T, 2, Stack<{ N * M }>>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize, const M: usize> DerefMut for Matrix<T, N, M>
where
    T: 'static + Copy + Default,
    [(); N * M]:
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

pub type Matrix2x2<T> = Matrix<T, 2, 2>;
pub type Matrix2x3<T> = Matrix<T, 2, 3>;
pub type Matrix2x4<T> = Matrix<T, 2, 4>;
pub type Matrix2x5<T> = Matrix<T, 2, 5>;
pub type Matrix2x6<T> = Matrix<T, 2, 6>;
pub type Matrix2x7<T> = Matrix<T, 2, 7>;
pub type Matrix2x8<T> = Matrix<T, 2, 8>;
pub type Matrix2x9<T> = Matrix<T, 2, 9>;

pub type Matrix3x2<T> = Matrix<T, 3, 2>;
pub type Matrix3x3<T> = Matrix<T, 3, 3>;
pub type Matrix3x4<T> = Matrix<T, 3, 4>;
pub type Matrix3x5<T> = Matrix<T, 3, 5>;
pub type Matrix3x6<T> = Matrix<T, 3, 6>;
pub type Matrix3x7<T> = Matrix<T, 3, 7>;
pub type Matrix3x8<T> = Matrix<T, 3, 8>;
pub type Matrix3x9<T> = Matrix<T, 3, 9>;

pub type Matrix4x2<T> = Matrix<T, 4, 2>;
pub type Matrix4x3<T> = Matrix<T, 4, 3>;
pub type Matrix4x4<T> = Matrix<T, 4, 4>;
pub type Matrix4x5<T> = Matrix<T, 4, 5>;
pub type Matrix4x6<T> = Matrix<T, 4, 6>;
pub type Matrix4x7<T> = Matrix<T, 4, 7>;
pub type Matrix4x8<T> = Matrix<T, 4, 8>;
pub type Matrix4x9<T> = Matrix<T, 4, 9>;

pub type Matrix5x2<T> = Matrix<T, 5, 2>;
pub type Matrix5x3<T> = Matrix<T, 5, 3>;
pub type Matrix5x4<T> = Matrix<T, 5, 4>;
pub type Matrix5x5<T> = Matrix<T, 5, 5>;
pub type Matrix5x6<T> = Matrix<T, 5, 6>;
pub type Matrix5x7<T> = Matrix<T, 5, 7>;
pub type Matrix5x8<T> = Matrix<T, 5, 8>;
pub type Matrix5x9<T> = Matrix<T, 5, 9>;

pub type Matrix6x2<T> = Matrix<T, 6, 2>;
pub type Matrix6x3<T> = Matrix<T, 6, 3>;
pub type Matrix6x4<T> = Matrix<T, 6, 4>;
pub type Matrix6x5<T> = Matrix<T, 6, 5>;
pub type Matrix6x6<T> = Matrix<T, 6, 6>;
pub type Matrix6x7<T> = Matrix<T, 6, 7>;
pub type Matrix6x8<T> = Matrix<T, 6, 8>;
pub type Matrix6x9<T> = Matrix<T, 6, 9>;

pub type Matrix7x2<T> = Matrix<T, 7, 2>;
pub type Matrix7x3<T> = Matrix<T, 7, 3>;
pub type Matrix7x4<T> = Matrix<T, 7, 4>;
pub type Matrix7x5<T> = Matrix<T, 7, 5>;
pub type Matrix7x6<T> = Matrix<T, 7, 6>;
pub type Matrix7x7<T> = Matrix<T, 7, 7>;
pub type Matrix7x8<T> = Matrix<T, 7, 8>;
pub type Matrix7x9<T> = Matrix<T, 7, 9>;

pub type Matrix8x2<T> = Matrix<T, 8, 2>;
pub type Matrix8x3<T> = Matrix<T, 8, 3>;
pub type Matrix8x4<T> = Matrix<T, 8, 4>;
pub type Matrix8x5<T> = Matrix<T, 8, 5>;
pub type Matrix8x6<T> = Matrix<T, 8, 6>;
pub type Matrix8x7<T> = Matrix<T, 8, 7>;
pub type Matrix8x8<T> = Matrix<T, 8, 8>;
pub type Matrix8x9<T> = Matrix<T, 8, 9>;

pub type Matrix9x2<T> = Matrix<T, 9, 2>;
pub type Matrix9x3<T> = Matrix<T, 9, 3>;
pub type Matrix9x4<T> = Matrix<T, 9, 4>;
pub type Matrix9x5<T> = Matrix<T, 9, 5>;
pub type Matrix9x6<T> = Matrix<T, 9, 6>;
pub type Matrix9x7<T> = Matrix<T, 9, 7>;
pub type Matrix9x8<T> = Matrix<T, 9, 8>;
pub type Matrix9x9<T> = Matrix<T, 9, 9>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let matrix = Matrix2x2::<f32>::new();
        assert_eq!( matrix[ 0 ], 0.0 );
    }

    #[test]
    fn default_test() {
        let matrix = Matrix2x2::<f32>::default();
        assert_eq!( matrix[ 0 ], 0.0 );
    }

    #[test]
    fn iter_test() {
        let src = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ];
        let matrix = Matrix3x3::<u32>::take( src );
        for ( i, value ) in matrix.deref().iter().enumerate() {
            assert_eq!( value, &src[ i ] );
        }
    }
}