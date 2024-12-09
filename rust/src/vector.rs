// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut, Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign }
};

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

#[derive( Clone, Copy, Default, Debug )]
pub struct Vector<T: 'static + Default + Copy + Debug, const COL: usize>( Tensor<T, 1, Stack<COL>> );

impl<T, const COL: usize> Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    pub fn new() -> Self {
        Self( Tensor::<T, 1, Stack<COL>>::new( Shape::<1>::from( [COL] ) ) )
    }

    pub fn take( src: [T; COL] ) -> Self {
        Self( Tensor::<T, 1, Stack<COL>>::take( Shape::<1>::from( [COL] ), src ) )
    }

    pub fn cols() -> usize {
        COL
    }
}

impl<T, const COL: usize> PartialEq for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug + PartialEq
{
    fn eq( &self, other: &Self ) -> bool {
        self.0 == other.0
    }
}

impl<T, const COL: usize> Deref for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    type Target = Tensor<T, 1, Stack<COL>>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T, const COL: usize> DerefMut for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const COL: usize> Add for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( self, other: Self ) -> Self::Output {
        Self( self.0 + other.0 )
    }
}

impl<T, const COL: usize> Sub for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( self, other: Self ) -> Self::Output {
        Self( self.0 - other.0 )
    }
}

impl<T, const COL: usize> Mul for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( self, other: Self ) -> Self::Output {
        Self( self.0 * other.0 )
    }
}

impl<T, const COL: usize> Div for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( self, other: Self ) -> Self::Output {
        Self( self.0 / other.0 )
    }
}

impl<T, const COL: usize> Add<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( self, scalar: T ) -> Self::Output {
        Self( self.0 + scalar )
    }
}

impl<T, const COL: usize> Sub<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( self, scalar: T ) -> Self::Output {
        Self( self.0 - scalar )
    }
}

impl<T, const COL: usize> Mul<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( self, scalar: T ) -> Self::Output {
        Self( self.0 * scalar )
    }
}

impl<T, const COL: usize> Div<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( self, scalar: T ) -> Self::Output {
        Self( self.0 / scalar )
    }
}

impl<T, const COL: usize> AddAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, other: Self ) {
        self.0 += other.0;
    }
}

impl<T, const COL: usize> SubAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, other: Self ) {
        self.0 -= other.0;
    }
}

impl<T, const COL: usize> MulAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, other: Self ) {
        self.0 *= other.0;
    }
}

impl<T, const COL: usize> DivAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, other: Self ) {
        self.0 /= other.0;
    }
}

impl<T, const COL: usize> AddAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, scalar: T ) {
        self.0 += scalar;
    }
}

impl<T, const COL: usize> SubAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, scalar: T ) {
        self.0 -= scalar;
    }
}

impl<T, const COL: usize> MulAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, scalar: T ) {
        self.0 *= scalar;
    }
}

impl<T, const COL: usize> DivAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, scalar: T ) {
        self.0 /= scalar;
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

    #[test]
    fn mat_mul_test() {
        use crate::matrix::Matrix2x2;
        use crate::tensor::contract;

        let a = Vector2::<f32>::take([
            1.0, 2.0
        ]);

        let b = Matrix2x2::<f32>::take([
            1.0, 2.0,
            3.0, 4.0
        ]);

        let mut c = Vector2::<f32>::take([
            0.0, 0.0
        ]);

        println!( "Before:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        contract( &a, &b, &mut c, &[0], &[0] );

        println!( "After:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        assert_eq!( c[0], 7.0 );
        assert_eq!( c[1], 10.0 );
    }
}
