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
pub struct Rot3YX<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3YX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    pub fn new( roll: T, pitch: T ) -> Self {
        let cφ = roll.cos();
        let sφ = roll.sin();
        let cψ = pitch.cos();
        let sψ = pitch.sin();
        let cφcψ = cφ * cψ;
        let cφsψ = cφ * sψ;
        let sφcψ = sφ * cψ;
        let sφsψ = sφ * sψ;
        Self ( Matrix3x3::new([
            cφcψ-sφsψ, cφsψ+sφcψ,      -sφ,
                  -sψ,        cψ, T::one(),
            sφcψ+cφsψ, sφsψ-cφcψ,       cφ
        ]))
    }
}

impl<T> Deref for Rot3YX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3YX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3YX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.contract( rhs )
    }
}

use crate::rotation::x::Rot3X;

impl<T> Mul<Rot3X<T>> for Rot3YX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn mul( self, rhs: Rot3X<T> ) -> Self::Output {
        Self( self.contract( rhs.0 ) )
    }
}

use crate::rotation::xy::Rot3XY;

impl<T> Transpose for Rot3YX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3XY<T>;

    fn transpose( self ) -> Self::Output {
        Rot3XY( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3YX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3XY<T>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Rot3XY( self.0.transpose() );
    }
}
