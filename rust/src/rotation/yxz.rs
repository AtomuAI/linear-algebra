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
pub struct Rot3YXZ<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3YXZ<T>
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
        let cφcψ = cφ * cψ;
        let cφsψ = cφ * sψ;
        let cθsφ = cθ * sφ;
        let cθsψ = cθ * sψ;
        let cθcψ = cθ * cψ;
        let sφcψ = sφ * cψ;
        let sφsψ = sφ * sψ;
        let cφcθ = cφ * cθ;
        let sφsθ = sφ * sθ;
        let cφsθ = cφ * sθ;
        let sφsθsψ = sφsθ * sψ;
        let sφsθcψ = sφsθ * cψ;
        let cφsθsψ = cφsθ * sψ;
        let cφsθcψ = cφsθ * cψ;
        Self ( Matrix3x3::new([
            cφcψ-sφsθsψ, cφsψ+sφsθcψ, -cθsφ,
                  -cθsψ,        cθcψ,    sθ,
            sφcψ+cφsθsψ, sφsψ-cφsθcψ,  cφcθ
        ]))
    }
}

impl<T> Deref for Rot3YXZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3YXZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3YXZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.contract( rhs )
    }
}

use crate::rotation::z::Rot3Z;

impl<T> Mul<Rot3Z<T>> for Rot3YXZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn mul( self, rhs: Rot3Z<T> ) -> Self::Output {
        Self( self.contract( rhs.0 ) )
    }
}

use crate::rotation::zxy::Rot3ZXY;

impl<T> Transpose for Rot3YXZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3ZXY<T>;

    fn transpose( self ) -> Self::Output {
        Rot3ZXY( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3YXZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3ZXY<T>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Rot3ZXY( self.0.transpose() );
    }
}
