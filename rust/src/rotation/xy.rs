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
        Contract,
        Transpose,
        TransposeAssignTo
    }
};

#[derive( Clone, Default, Debug )]
pub struct Rot3XY<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3XY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    pub fn new( roll: T, pitch: T ) -> Self {
        let cφ = roll.cos();
        let sφ = roll.sin();
        let cψ = pitch.cos();
        let sψ = pitch.sin();
        let cφsψ = cφ * sψ;
        let cφcψ = cφ * cψ;
        let sφsψ = sφ * sψ;
        let sφcψ = sφ * cψ;
        Self ( Matrix3x3::new([
                   cψ,        sψ, -T::one(),
            sφcψ-cφsψ, sφsψ+cφcψ,        sφ,
            cφcψ+sφsψ, cφsψ-sφcψ,        cφ
        ]))
    }
}

impl<T> Deref for Rot3XY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3XY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3XY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.contract( rhs )
    }
}

use crate::rotation::y::Rot3Y;

impl<T> Mul<Rot3Y<T>> for Rot3XY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn mul( self, rhs: Rot3Y<T> ) -> Self::Output {
        Self( self.contract( rhs.0 ) )
    }
}

use crate::rotation::yx::Rot3YX;

impl<T> Transpose for Rot3XY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3YX<T>;

    fn transpose( self ) -> Self::Output {
        Rot3YX( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3XY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3YX<T>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Rot3YX( self.0.transpose() );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector3;

    #[test]
    fn test_rot3xyz() {
        let rot = Rot3XY::new( 0.0.to_radians(), 0.0.to_radians() );
        let vec = Vector3::new([ 1.0, 0.0, 0.0 ]);
        let res = rot * vec;
        println!( "Vec: {:?}", vec );
        println!( "Res: {:?}", res );
        //assert_eq!( res, Vector2::new([ 0.0, 1.0 ]) );
    }
}
