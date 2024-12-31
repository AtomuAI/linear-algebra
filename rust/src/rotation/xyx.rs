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
pub struct Rot3XYX<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3XYX<T>
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
        let sθsψ = sθ * sψ;
        let sθcψ = sθ * cψ;
        let sφsθ = sφ * sθ;
        let cφcψ = cφ * cψ;
        let cφsψ = cφ * sψ;
        let cφsθ = cφ * sθ;
        let sφcψ = sφ * cψ;
        let sφsψ = sφ * sψ;
        let sφcθ = sφ * cθ;
        let cφcθ = cφ * cθ;
        let sφcθsψ = sφcθ * sψ;
        let sφcθcψ = sφcθ * cψ;
        let cφcθsψ = cφcθ * sψ;
        let cφcθcψ = cφcθ * cψ;
        Self ( Matrix3x3::new([
              cθ,         sθsψ,        -sθcψ,
            sφsθ,  cφcψ-sφcθsψ,  cφsψ+sφcθcψ,
            cφsθ, -sφcψ-cφcθsψ, -sφsψ+cφcθcψ
        ]))
    }
}

impl<T> Deref for Rot3XYX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3XYX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3XYX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.contract( rhs )
    }
}

use crate::rotation::x::Rot3X;

impl<T> Mul<Rot3X<T>> for Rot3XYX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn mul( self, rhs: Rot3X<T> ) -> Self::Output {
        Self( self.contract( rhs.0 ) )
    }
}

impl<T> Transpose for Rot3XYX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn transpose( self ) -> Self::Output {
        Self( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3XYX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Self( self.0.transpose() );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rot_xyx() {
        let rot = Rot3XYX::new( 0.0, 0.0, 0.0 );
        assert_eq!( rot[[ 0, 0 ]], 1.0 );
        assert_eq!( rot[[ 0, 1 ]], 0.0 );
        assert_eq!( rot[[ 0, 2 ]], 0.0 );
        assert_eq!( rot[[ 1, 0 ]], 0.0 );
        assert_eq!( rot[[ 1, 1 ]], 1.0 );
        assert_eq!( rot[[ 1, 2 ]], 0.0 );
        assert_eq!( rot[[ 2, 0 ]], 0.0 );
        assert_eq!( rot[[ 2, 1 ]], 0.0 );
        assert_eq!( rot[[ 2, 2 ]], 1.0 );
    }
}
