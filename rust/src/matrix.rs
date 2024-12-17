// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut, Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign }
};
use num::traits::Num;

use memory::stack::Stack;

use crate::{
    vector::Vector,
    shape::Shape,
    tensor::{ Tensor, TensorAccess }
};

#[derive( Clone, Debug )]
pub struct Matrix<T, const COL: usize, const ROW: usize>( [ T; COL * ROW ] ) where T: 'static + Copy + Default + Debug, [(); COL * ROW]:;

impl<T, const COL: usize, const ROW: usize> Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    pub const fn new_const( src: [T; COL * ROW] ) -> Self {
        Self( src )
    }

    pub fn new() -> Self
    where
        T: Num
    {
        Self( [ T::zero(); COL * ROW ] )
    }

    #[inline(always)]
    pub const fn dim( &self ) -> usize {
        2
    }

    #[inline(always)]
    pub const fn cols() -> usize {
        COL
    }

    #[inline(always)]
    pub const fn rows() -> usize {
        ROW
    }

    #[inline(always)]
    pub const fn shape( &self ) -> Shape<2> {
        Shape::new_const( [COL, ROW] )
    }

    #[inline(always)]
    pub fn idx( col: usize, row: usize ) -> usize {
        row * ROW + col
    }

    #[inline(always)]
    pub const fn idx_const( col: usize, row: usize ) -> usize {
        row * ROW + col
    }

    pub fn reshape<const NEW_COL: usize, const NEW_ROW: usize>(&self) -> &Matrix<T, NEW_COL, NEW_ROW>
    where
        T: Default + Copy + Debug,
        [(); NEW_COL * NEW_ROW]:,
    {
        assert_eq!( COL * ROW, NEW_COL * NEW_ROW, "Total number of elements must remain the same for reshape." );
        unsafe { // SAFETY: This is safe because we have asserted that the total number of elements is the same
            &*( self as *const Matrix<T, COL, ROW> as *const Matrix<T, NEW_COL, NEW_ROW> )
        }
    }

    pub fn resize<const NEW_COL: usize, const NEW_ROW: usize>( &self ) -> Matrix<T, NEW_COL, NEW_ROW>
    where
        T: Default + Copy + Debug,
        [(); NEW_COL * NEW_ROW]:
    {
        let mut result = Matrix::default();
        let min_cols = COL.min( NEW_COL );
        let min_rows = ROW.min( NEW_ROW );

        self.0.chunks( COL ).take( min_rows ).enumerate().for_each( |( row, chunk )| {
            chunk.iter().take( min_cols ).enumerate().for_each( |( col, &value )| {
                result.0[ Self::idx( row, col ) ] = value;
            });
        });

        result
    }
}

impl<T, const COL: usize, const ROW: usize> Deref for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    type Target = [ T; COL * ROW ];

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T, const COL: usize, const ROW: usize> DerefMut for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const COL: usize, const ROW: usize> Index<usize> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.0[ index ]
    }
}

impl<T, const COL: usize, const ROW: usize> IndexMut<usize> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.0[ index ]
    }
}

impl<T, const COL: usize, const ROW: usize> Index<[usize; 2]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    type Output = T;

    fn index( &self, index: [usize; 2] ) -> &Self::Output {
        &self.0[ Self::idx( index[ 1 ], index[ 0 ] ) ]

    }
}

impl<T, const COL: usize, const ROW: usize> IndexMut<[usize; 2]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    fn index_mut( &mut self, index: [usize; 2] ) -> &mut Self::Output {
        &mut self.0[ Self::idx( index[ 1 ], index[ 0 ] ) ]
    }
}

impl<T, const COL: usize, const ROW: usize> Default for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:
{
    fn default() -> Self {
        Self( [ T::default(); COL * ROW ] )
    }
}

impl<T, const COL: usize, const ROW: usize> From<[T; COL * ROW]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: [T; COL * ROW] ) -> Self {
        Self( src )
    }
}

impl<T, const COL: usize, const ROW: usize> From<[[T; COL]; ROW]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * ROW]:,
{
    fn from(src: [[T; COL]; ROW]) -> Self {
        let mut data = [T::default(); COL * ROW];

        src.iter().enumerate().for_each(
            |( i, row )| { row.iter().enumerate().for_each(
                |( j, &item )| data[ Self::idx( i, j ) ] = item
            )}
        );

        Self(data)
    }
}

#[allow(clippy::identity_op)]
impl<T, const COL: usize> From<Vector<T, COL>> for Matrix<T, COL, 1>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * 1]:
{
    fn from( src: Vector<T, COL> ) -> Self {
        unsafe{ ::core::ptr::read( &src as *const Vector<T, COL> as *const Matrix<T, COL, 1> ) }
    }
}

impl<T, const COL: usize, const ROW: usize> From<Tensor<T, 2, Stack<{COL * ROW}>>> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: Tensor<T, 2, Stack<{COL * ROW}>> ) -> Self {
        Self( ***src.memory() )
    }
}

impl<T, const COL: usize, const ROW: usize> PartialEq for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug + PartialEq,
    [(); COL * ROW]:
{
    fn eq( &self, other: &Self ) -> bool {
        self.0 == other.0
    }
}


impl<T, const COL: usize, const ROW: usize> Add for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn add( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Sub for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn sub( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Mul for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn mul( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Div for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn div( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Add<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn add( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Sub<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn sub( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Mul<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn mul( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Div<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone,
    [(); COL * ROW]:
{
    type Output = Self;

    fn div( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> AddAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + AddAssign,
    [(); COL * ROW]:
{
    fn add_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a += b );
    }
}

impl<T, const COL: usize, const ROW: usize> SubAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + SubAssign,
    [(); COL * ROW]:
{
    fn sub_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a -= b );
    }
}

impl<T, const COL: usize, const ROW: usize> MulAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + MulAssign,
    [(); COL * ROW]:
{
    fn mul_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a *= b );
    }
}

impl<T, const COL: usize, const ROW: usize> DivAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + DivAssign,
    [(); COL * ROW]:
{
    fn div_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a /= b );
    }
}

impl<T, const COL: usize, const ROW: usize> AddAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + AddAssign,
    [(); COL * ROW]:
{
    fn add_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a += scalar );
    }
}

impl<T, const COL: usize, const ROW: usize> SubAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + SubAssign,
    [(); COL * ROW]:
{
    fn sub_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a -= scalar );
    }
}

impl<T, const COL: usize, const ROW: usize> MulAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + MulAssign,
    [(); COL * ROW]:
{
    fn mul_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a *= scalar );
    }
}

impl<T, const COL: usize, const ROW: usize> DivAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + DivAssign,
    [(); COL * ROW]:
{
    fn div_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a /= scalar );
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
        let matrix = Matrix3x3::<u32>::from( src );
        for ( i, value ) in matrix.deref().iter().enumerate() {
            assert_eq!( value, &src[ i ] );
        }
    }

    #[test]
    fn mat_mul_test() {
        use crate::tensor::contract;

        let a = Matrix2x2::<f32>::from([
            1.0, 2.0,
            3.0, 4.0
        ]);

        let b = Matrix2x2::<f32>::from([
            1.0, 2.0,
            3.0, 4.0
        ]);

        let c = Matrix2x2::<f32>::from([
            0.0, 0.0,
            0.0, 0.0
        ]);

        println!( "Before:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        let a: Tensor<f32, 2, Stack<4>> = a.into();
        let b: Tensor<f32, 2, Stack<4>> = b.into();
        let mut c: Tensor<f32, 2, Stack<4>> = c.into();
        contract( &a, &b, &mut c, &[1], &[0] );

        println!( "After:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        assert_eq!( c[[0, 0]], 7.0 );
        assert_eq!( c[[0, 1]], 10.0 );
        assert_eq!( c[[1, 0]], 15.0 );
        assert_eq!( c[[1, 1]], 22.0 );
    }
}
