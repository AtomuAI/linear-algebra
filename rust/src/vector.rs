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
pub struct Vector<T: 'static + Default + Copy, const N: usize>( Tensor<T, 1, Stack<N>> );

impl<T, const N: usize> Vector<T, N>
where
    T: 'static + Copy + Default
{
    pub fn new() -> Self {
        Self( Tensor::<T, 1, Stack<N>>::new( Shape::<1>::from( [N] ) ) )
    }

    pub fn take( src: [T; N] ) -> Self {
        Self( Tensor::<T, 1, Stack<N>>::take( src ) )
    }
}

impl<T, const N: usize> Deref for Vector<T, N>
where
    T: 'static + Copy + Default
{
    type Target = Tensor<T, 1, Stack<N>>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Vector<T, N>
where
    T: 'static + Copy + Default
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector4<T> = Vector<T, 4>;
pub type Vector5<T> = Vector<T, 5>;
pub type Vector6<T> = Vector<T, 6>;
pub type Vector7<T> = Vector<T, 7>;
pub type Vector8<T> = Vector<T, 8>;
pub type Vector9<T> = Vector<T, 9>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let vector = Vector2::<f32>::new();
        assert_eq!( vector[ 0 ], 0.0 );
    }

    #[test]
    fn default_test() {
        let vector = Vector2::<f32>::default();
        assert_eq!( vector[ 0 ], 0.0 );
    }

    #[test]
    fn iter_test() {
        let src = [ 1, 2, 3, 4, 5 ];
        let vector = Vector::<u32, 5>::take( src );
        for ( i, value ) in vector.iter().enumerate() {
            assert_eq!( value, &src[ i ] );
        }
    }
}