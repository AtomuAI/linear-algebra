// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Deref, Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg }
};
use num::{ complex::Complex, Num, Float, traits::{NumAssign, FloatConst} };

use crate::tensor::TensorTraits;
use crate::matrix::Matrix3x3;

#[derive( Clone, Default, Debug )]
pub struct Rotation<T>( Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Quaternion<T>
where
    T: 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    pub fn new( w: T, x: T, y: T, z: T ) -> Self {
        Self (
            Matrix3x3::take([
                Complex::new( w, x ), Complex::new( y, z ),
                Complex::new( -y, z ), Complex::new( w, -x )
            ])
        )
    }

    pub fn from_yaw( yaw: T ) -> Self
    where
        T: Div<Output = T>
    {
        let w = ( yaw / T::from( 2 ).unwrap() ).cos();
        let x = ( yaw / T::from( 2 ).unwrap() ).sin();
        let y = T::zero();
        let z = T::zero();
        Self::new( w, x, y, z )
    }

    pub fn from_pitch( pitch: T ) -> Self
    where
        T: Div<Output = T>
    {
        let w = ( pitch / T::from( 2 ).unwrap() ).cos();
        let x = T::zero();
        let y = ( pitch / T::from( 2 ).unwrap() ).sin();
        let z = T::zero();

        Self::new( w, x, y, z )
    }

    pub fn from_roll( roll: T ) -> Self
    where
        T: Div<Output = T>
    {
        let w = ( roll / T::from( 2 ).unwrap() ).cos();
        let x = T::zero();
        let y = T::zero();
        let z = ( roll / T::from( 2 ).unwrap() ).cos();

        Self::new( w, x, y, z )
    }

    pub fn from_euler( yaw: T, pitch: T, roll: T ) -> Self
    where
        T: Div<Output = T> + Float,
    {
        let ( cy, sy ) = ( ( yaw / T::from( 2 ).unwrap() ).cos(), ( yaw / T::from( 2 ).unwrap() ).sin() );
        let ( cp, sp ) = ( ( pitch / T::from( 2 ).unwrap() ).cos(), ( pitch / T::from( 2 ).unwrap() ).sin() );
        let ( cr, sr ) = ( ( roll / T::from( 2 ).unwrap() ).cos(), ( roll / T::from( 2 ).unwrap() ).sin() );

        let w = cr * cp * cy + sr * sp * sy;
        let x = sr * cp * cy - cr * sp * sy;
        let y = cr * sp * cy + sr * cp * sy;
        let z = cr * cp * sy - sr * sp * cy;

        Self::new( w, x, y, z )
    }

    pub fn to_euler(&self) -> (T, T, T)
    where
        T: Div<Output = T> + Float + FloatConst,
    {
        let sinr_cosp = T::from( 2 ).unwrap() * ( *self.w() * *self.x() + *self.y() * *self.z() );
        let cosr_cosp = T::one() - T::from( 2 ).unwrap() * ( *self.x() * *self.x() + *self.y() * *self.y() );
        let roll = sinr_cosp.atan2( cosr_cosp );

        let sinp = T::from( 2 ).unwrap() * ( *self.w() * *self.y() - *self.z() * *self.x() );
        let pitch = if sinp.abs() >= T::one() {
            ( T::PI() / T::from( 2 ).unwrap() ).copysign( sinp )
        } else {
            sinp.asin()
        };

        let siny_cosp = T::from( 2 ).unwrap() * ( *self.w() * *self.z() + *self.x() * *self.y() );
        let cosy_cosp = T::one() - T::from( 2 ).unwrap() * ( *self.y() * *self.y() + *self.z() * *self.z() );
        let yaw = siny_cosp.atan2( cosy_cosp );

        ( yaw, pitch, roll )
    }

    fn squared_magnitude( &self ) -> T {
        self.w().powi( 2 ) + self.x().powi( 2 ) + self.y().powi( 2 ) + self.z().powi( 2 )
    }

    pub fn magnitude( &self ) -> T {
        self.squared_magnitude().sqrt()
    }

    pub fn conjugate( &self ) -> Self {
        Self::new( *self.w(), -*self.x(), -*self.y(), -*self.z() )
    }

    pub fn inverse( &self ) -> Self
    where
        Self: DivAssign<T>
    {
        let mut conjugate = self.conjugate();
        conjugate /= self.squared_magnitude();
        conjugate
    }

    pub fn w( &self ) -> &T { &self[[ 0, 0 ]].re }
    pub fn x( &self ) -> &T { &self[[ 0, 0 ]].im }
    pub fn y( &self ) -> &T { &self[[ 1, 0 ]].re }
    pub fn z( &self ) -> &T { &self[[ 1, 0 ]].im }
}

impl<T> Deref for Quaternion<T>
where
    T: Default + Copy + Debug
{
    type Target = Matrix2x2<Complex<T>>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T> Add for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( self, other: Self ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] + other.0[ i ];
        }
        result
    }
}

impl<T> Sub for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( self, other: Self ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] - other.0[ i ];
        }
        result
    }
}

impl<T> Mul for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( self, other: Self ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] * other.0[ i ];
        }
        result
    }
}

impl<T> Div for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( self, other: Self ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] / other.0[ i ];
        }
        result
    }
}

impl<T> Add<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] + scalar;
        }
        result
    }
}

impl<T> Sub<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] - scalar;
        }
        result
    }
}

impl<T> Mul<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] * scalar;
        }
        result
    }
}

impl<T> Div<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.0.size() {
            result.0[ i ] = self.0[ i ] / scalar;
        }
        result
    }
}

impl<T> AddAssign for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + AddAssign,
{
    fn add_assign( &mut self, other: Self ) {
        for i in 0..self.0.size() {
            self.0[ i ] += other.0[ i ];
        }
    }
}

impl<T> SubAssign for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + SubAssign,
{
    fn sub_assign( &mut self, other: Self ) {
        for i in 0..self.0.size() {
            self.0[ i ] -= other.0[ i ];
        }
    }
}

impl<T> MulAssign for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + MulAssign,
{
    fn mul_assign( &mut self, other: Self ) {
        for i in 0..self.0.size() {
            self.0[ i ] *= other.0[ i ];
        }
    }
}

impl<T> DivAssign for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + DivAssign,
{
    fn div_assign( &mut self, other: Self ) {
        for i in 0..self.0.size() {
            self.0[ i ] /= other.0[ i ];
        }
    }
}

impl<T> AddAssign<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + AddAssign,
{
    fn add_assign( &mut self, scalar: T ) {
        for i in 0..self.0.size() {
            self.0[ i ] += scalar;
        }
    }
}

impl<T> SubAssign<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + SubAssign,
{
    fn sub_assign( &mut self, scalar: T ) {
        for i in 0..self.0.size() {
            self.0[ i ] -= scalar;
        }
    }
}

impl<T> MulAssign<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + MulAssign,
{
    fn mul_assign( &mut self, scalar: T ) {
        for i in 0..self.0.size() {
            self.0[ i ] *= scalar;
        }
    }
}

impl<T> DivAssign<T> for Quaternion<T>
where
    T: Default + Copy + Debug + Num + NumAssign + DivAssign,
{
    fn div_assign( &mut self, scalar: T ) {
        for i in 0..self.0.size() {
            self.0[ i ] /= scalar;
        }
    }
}

// Dimensional Indexing
impl<T> Index<[usize; 2]> for Quaternion<T>
where
    T: Default + Copy + Debug,
{
    type Output = Complex<T>;

    fn index( &self, index: [usize; 2] ) -> &Self::Output {
        &self.0[ index ]
    }
}

impl<T> IndexMut<[usize; 2]> for Quaternion<T>
where
    T: Default + Copy + Debug,
{
    fn index_mut( &mut self, index: [usize; 2] ) -> &mut Self::Output {
        &mut self.0[ index ]
    }
}

// Flat Indexing
impl<T> Index<usize> for Quaternion<T>
where
    T: Default + Copy + Debug,
{
    type Output = Complex<T>;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.0[ index ]
    }
}

impl<T> IndexMut<usize> for Quaternion<T>
where
    T: Default + Copy + Debug,
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.0[ index ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_quat_test() {
        let quat = Quaternion::new( 1.0, 2.0, 3.0, 4.0 );
        assert_eq!( quat[ 0 ], Complex::new( 1.0, 2.0 ) );
        assert_eq!( quat[ 1 ], Complex::new( 3.0, 4.0 ) );
        assert_eq!( quat[ 2 ], Complex::new( -3.0, 4.0 ) );
        assert_eq!( quat[ 3 ], Complex::new( 1.0, -2.0 ) );
    }

    /*
    #[test]
    fn from_euler_test() {
        let quat = Quaternion::from_euler( 0.0, 0.0, 0.0 );
        assert_eq!( quat[ 0 ], Complex::new( 1.0, 0.0 ) );
        assert_eq!( quat[ 1 ], Complex::new( 0.0, 0.0 ) );
        assert_eq!( quat[ 2 ], Complex::new( 0.0, 0.0 ) );
        assert_eq!( quat[ 3 ], Complex::new( 0.0, 0.0 ) );
    }

    #[test]
    fn to_euler_test() {
        let quat = Quaternion::from_euler( 0.0, 0.0, 0.0 );
        let ( yaw, pitch, roll ) = quat.to_euler();
        assert_eq!( yaw, 0.0 );
        assert_eq!( pitch, 0.0 );
        assert_eq!( roll, 0.0 );
    }
    */

    #[test]
    fn quat_conj_test() {
        let quat = Quaternion::new( 1.0, 2.0, 3.0, 4.0 );
        let conj = quat.conjugate();
        println!( "{:?}", conj );
        assert_eq!( conj[ 0 ], Complex::new( 1.0, -2.0 ) );
        assert_eq!( conj[ 1 ], Complex::new( 3.0, -4.0 ) );
        assert_eq!( conj[ 2 ], Complex::new( -3.0, -4.0 ) );
        assert_eq!( conj[ 3 ], Complex::new( 1.0, 2.0 ) );
    }
}
