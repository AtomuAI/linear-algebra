// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut, Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign }
};
use num::traits::Num;

use memory::stack::Stack;
use arithmetic::{ AddAssignTo, SubAssignTo, MulAssignTo, DivAssignTo };

use crate::{
    matrix::Matrix,
    shape::Shape,
    tensor::{ Tensor, TensorAccess }
};

#[derive( Clone, Debug )]
pub struct Vector<T: 'static + Default + Copy + Debug, const COL: usize>( [ T; COL ] );

impl<T, const COL: usize> Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    pub const fn new_const( src: [T; COL] ) -> Self {
        Self( src )
    }

    pub fn new( src: [T; COL] ) -> Self {
        Self( src )
    }

    pub fn zero() -> Self
    where
        T: Num
    {
        Self( [T::zero(); COL] )
    }

    #[inline(always)]
    pub const fn dim( &self ) -> usize {
        1
    }

    #[inline(always)]
    pub const fn cols() -> usize {
        COL
    }

    #[inline(always)]
    pub const fn shape( &self ) -> Shape<1> {
        Shape::new_const( [COL] )
    }

    pub fn resize<const NEW_COL: usize>( &self ) -> Vector<T, NEW_COL>
    where
        T: Default + Copy + Debug,
    {
        let mut result = Vector::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &value, result_value )| *result_value = value );
        result
    }

    pub fn dot( &self, other: &Self ) -> T
    where
        T: Default + Copy + Debug + Add<Output = T> + Mul<Output = T>
    {
        self.iter().zip( other.iter() )
            .fold( T::default(), |acc, ( &a, &b )| acc + a * b )
    }
}

impl<T, const COL: usize> Deref for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    type Target = [T; COL];

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

impl<T, const COL: usize> Index<usize> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.0[ index ]
    }
}

impl<T, const COL: usize> IndexMut<usize> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.0[ index ]
    }
}

impl<T, const COL: usize> Default for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn default() -> Self {
        Self( [T::default(); COL] )
    }
}

impl<T, const COL: usize> From<[T; COL]> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: [T; COL] ) -> Self {
        Self( src )
    }
}

#[allow(clippy::identity_op)]
impl<T, const COL: usize> From<Matrix<T, COL, 1>> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug,
    [(); COL * 1]:
{
    fn from( src: Matrix<T, COL, 1> ) -> Self {
        unsafe{ ::core::ptr::read( &src as *const Matrix<T, COL, 1> as *const Vector<T, COL> ) }
    }
}

impl<T, const COL: usize> From<Tensor<T, 1, Stack<COL>>> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: Tensor<T, 1, Stack<COL>> ) -> Self {
        Self( ***src.memory() )
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

impl<T, const COL: usize> Add for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( mut self, other: Self ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a + b );
        self
    }
}

impl<T, const COL: usize> Sub for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( mut self, other: Self ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a - b );
        self
    }
}

impl<T, const COL: usize> Mul for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( mut self, other: Self ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a * b );
        self
    }
}

impl<T, const COL: usize> Div for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( mut self, other: Self ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a / b );
        self
    }
}

impl<T, const COL: usize> Add<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a + scalar );
        self
    }
}

impl<T, const COL: usize> Sub<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a - scalar );
        self
    }
}

impl<T, const COL: usize> Mul<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a * scalar );
        self
    }
}

impl<T, const COL: usize> Div<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a / scalar );
        self
    }
}

impl<T, const COL: usize> AddAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a += b );
    }
}

impl<T, const COL: usize> SubAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a -= b );
    }
}

impl<T, const COL: usize> MulAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a *= b );
    }
}

impl<T, const COL: usize> DivAssign for Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a /= b );
    }
}

impl<T, const COL: usize> AddAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a += scalar );
    }
}

impl<T, const COL: usize> SubAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a -= scalar );
    }
}

impl<T, const COL: usize> MulAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a *= scalar );
    }
}

impl<T, const COL: usize> DivAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a /= scalar );
    }
}

impl<T, const COL: usize> AddAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Self;

    fn add_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
    }
}

impl<T, const COL: usize> SubAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Self;

    fn sub_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
    }
}

impl<T, const COL: usize> MulAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Self;

    fn mul_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
    }
}

impl<T, const COL: usize> DivAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Self;

    fn div_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
    }
}

impl<T, const COL: usize> AddAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Self;

    fn add_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
    }
}

impl<T, const COL: usize> SubAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Self;

    fn sub_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
    }
}

impl<T, const COL: usize> MulAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Self;

    fn mul_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
    }
}

impl<T, const COL: usize> DivAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Self;

    fn div_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
    }
}

//

impl<T, const COL: usize> Add for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn add( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
        result
    }
}

impl<T, const COL: usize> Sub for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn sub( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
        result
    }
}

impl<T, const COL: usize> Mul for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn mul( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
        result
    }
}

impl<T, const COL: usize> Div for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn div( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
        result
    }
}

impl<T, const COL: usize> Add<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn add( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
        result
    }
}

impl<T, const COL: usize> Sub<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn sub( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
        result
    }
}

impl<T, const COL: usize> Mul<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn mul( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
        result
    }
}

impl<T, const COL: usize> Div<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn div( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
        result
    }
}

impl<T, const COL: usize> AddAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a += b );
    }
}

impl<T, const COL: usize> SubAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a -= b );
    }
}

impl<T, const COL: usize> MulAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a *= b );
    }
}

impl<T, const COL: usize> DivAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a /= b );
    }
}

impl<T, const COL: usize> AddAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a += scalar );
    }
}

impl<T, const COL: usize> SubAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a -= scalar );
    }
}

impl<T, const COL: usize> MulAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a *= scalar );
    }
}

impl<T, const COL: usize> DivAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a /= scalar );
    }
}

impl<T, const COL: usize> AddAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Vector<T, COL>;

    fn add_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
    }
}

impl<T, const COL: usize> SubAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Vector<T, COL>;

    fn sub_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
    }
}

impl<T, const COL: usize> MulAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Vector<T, COL>;

    fn mul_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
    }
}

impl<T, const COL: usize> DivAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Vector<T, COL>;

    fn div_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
    }
}

impl<T, const COL: usize> AddAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Vector<T, COL>;

    fn add_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
    }
}

impl<T, const COL: usize> SubAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Vector<T, COL>;

    fn sub_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
    }
}

impl<T, const COL: usize> MulAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Vector<T, COL>;

    fn mul_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
    }
}

impl<T, const COL: usize> DivAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Vector<T, COL>;

    fn div_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
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
        let vector = Vector2::<f32>::zero();
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
        let vector = Vector::<u32, 5>::from( src );
        for ( i, value ) in vector.iter().enumerate() {
            assert_eq!( value, &src[ i ] );
        }
    }

    #[test]
    fn mat_mul_test() {
        use crate::matrix::Matrix2x2;
        use crate::tensor::contract;

        let a = Vector2::<f32>::from([
            1.0, 2.0
        ]);

        let b = Matrix2x2::<f32>::from([
            1.0, 2.0,
            3.0, 4.0
        ]);

        let c = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        /*
        let d = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        let e = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        let mut f = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        <&Vector2<f32>>::add_assign_to( &d, &e, &mut f );
        */

        println!( "Before:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        let a: Tensor<f32, 1, Stack<2>> = a.into();
        let b: Tensor<f32, 2, Stack<4>> = b.into();
        let mut c: Tensor<f32, 1, Stack<2>> = c.into();
        contract( &a, &b, &mut c, &[0], &[0] );

        println!( "After:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        assert_eq!( c[0], 7.0 );
        assert_eq!( c[1], 10.0 );
    }

    #[test]
    fn add_test() {
        let a = Vector2::<f32>::from([
            1.0, 2.0
        ]);

        let b = Vector2::<f32>::from([
            3.0, 4.0
        ]);

        let c = &a + &b;

        println!( "{:?} = <{:?}, {:?}>", c, a, b );
    }

    #[test]
    fn dot_test() {
        let a = Vector2::<f32>::from([
            1.0, 2.0
        ]);

        let b = Vector2::<f32>::from([
            3.0, 4.0
        ]);

        let c = a.dot( &b );

        println!( "{:?} = <{:?}, {:?}>", c, a, b );
        assert_eq!( c, 11.0 );
    }
}
