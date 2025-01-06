pub mod x;
pub mod xy;
pub mod xyx;
pub mod xyz;
pub mod xz;
pub mod xzx;
pub mod xzy;
pub mod y;
pub mod yx;
pub mod yxy;
pub mod yxz;
pub mod yz;
pub mod yzx;
pub mod yzy;
pub mod z;
pub mod zx;
pub mod zxy;
pub mod zxz;
pub mod zy;
pub mod zyx;
pub mod zyz;

// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Deref, Mul, AddAssign, Neg }
};
use num::{ Float, Num };

use crate::{
    matrix::{
        Matrix2x2,
        Matrix3x3
    },
    vector::{
        Vector2,
        Vector3
    },
    ops::Contract
};

#[derive( Clone, Default, Debug )]
pub struct Rot2<T>( Matrix2x2<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot2<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Float
{
    pub fn new( angle: T ) -> Self {
        let cφ = angle.cos();
        let sφ = angle.sin();
        Self ( Matrix2x2::new([
            cφ, -sφ,
            sφ,  cφ
        ]))
    }
}

impl<T> Deref for Rot2<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Float
{
    type Target = Matrix2x2<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T> Mul<Vector2<T>> for Rot2<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Vector2<T>;

    fn mul( self, rhs: Vector2<T> ) -> Self::Output {
        self.contract( rhs )
    }
}

impl<T> Mul for Rot2<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn mul( self, rhs: Self ) -> Self::Output {
        Self( self.contract( rhs.0 ) )
    }
}

#[derive( Clone, Default, Debug )]
pub struct Rot3<T>( Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Deref for Rot3<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T> Mul<Vector3<T>> for Rot3<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.contract( rhs )
    }
}

impl<T> Mul for Rot3<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn mul( self, rhs: Self ) -> Self::Output {
        Self( self.contract( rhs.0 ) )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rot2() {
        let rot = Rot2::new( -90.0.to_radians() );
        let vec = Vector2::new([ 1.0, 0.0 ]);
        let res = rot * vec;
        println!( "Vec: {:?}", vec );
        println!( "Res: {:?}", res );
        //assert_eq!( res, Vector2::new([ 0.0, 1.0 ]) );
    }
}
