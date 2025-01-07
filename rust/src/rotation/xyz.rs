// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Deref, Mul, AddAssign, Neg }
};
use num::{ Float, Num };

use crate::{
    matrix::Matrix3x3,
    vector::Vector3,
    ops::{
        MatrixMul,
        Transpose,
        TransposeAssignTo
    }
};

#[derive( Clone, Default, Debug )]
pub struct Rot3XYZ<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3XYZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    pub fn new( roll: T, pitch: T, yaw: T ) -> Self {
        let cφ = roll.cos();
        let sφ = roll.sin();
        let cψ = pitch.cos();
        let sψ = pitch.sin();
        let cθ = yaw.cos();
        let sθ = yaw.sin();
        let cθcψ = cθ * cψ;
        let cθsψ = cθ * sψ;
        let cφsψ = cφ * sψ;
        let cφcψ = cφ * cψ;
        let cθsφ = cθ * sφ;
        let sφsψ = sφ * sψ;
        let sφcψ = sφ * cψ;
        let cθcφ = cθ * cφ;
        let sφsθ = sφ * sθ;
        let cφsθ = cφ * sθ;
        let sφsθcψ = sφsθ * cψ;
        let sφsθsψ = sφsθ * sψ;
        let cφsθcψ = cφsθ * cψ;
        let cφsθsψ = cφsθ * sψ;
        Self ( Matrix3x3::new([
                   cθcψ,        cθsψ,  -sθ,
            sφsθcψ-cφsψ, sφsθsψ+cφcψ, cθsφ,
            cφsθcψ+sφsψ, cφsθsψ-sφcψ, cθcφ
        ]))
    }
}

impl<T> Deref for Rot3XYZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3XYZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3XYZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float,
    Matrix3x3<T>: MatrixMul<Vector3<T>, Output = Vector3<T>>
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.mat_mul( rhs )
    }
}

use crate::rotation::z::Rot3Z;

impl<T> Mul<Rot3Z<T>> for Rot3XYZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float,
    Matrix3x3<T>: MatrixMul<Matrix3x3<T>, Output = Matrix3x3<T>>
{
    type Output = Self;

    fn mul( self, rhs: Rot3Z<T> ) -> Self::Output {
        Self( self.mat_mul( rhs.0 ) )
    }
}

use crate::rotation::zyx::Rot3ZYX;

impl<T> Transpose for Rot3XYZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3ZYX<T>;

    fn transpose( self ) -> Self::Output {
        Rot3ZYX( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3XYZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3ZYX<T>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Rot3ZYX( self.0.transpose() );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector3;

    #[test]
    fn test_rot3xyz() {
        let rot = Rot3XYZ::new( 0.0.to_radians(), 0.0.to_radians(), 45.0.to_radians() );
        let vec = Vector3::new([ 1.0, 0.0, 0.0 ]);
        let res = rot * vec;
        println!( "Vec: {:?}", vec );
        println!( "Res: {:?}", res );
        //assert_eq!( res, Vector2::new([ 0.0, 1.0 ]) );
    }
}
