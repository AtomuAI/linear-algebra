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
pub struct Rot3ZX<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3ZX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    pub fn new( roll: T, yaw: T ) -> Self {
        let cφ = roll.cos();
        let sφ = roll.sin();
        let cθ = yaw.cos();
        let sθ = yaw.sin();
        let sφcθ = sφ * cθ;
        let cθcφ = cθ * cφ;
        let sφsθ = sφ * sθ;
        let cφsθ = cφ * sθ;
        Self ( Matrix3x3::new([
             cφ+sφsθ, sφcθ, -cφ+sφsθ,
            -sφ+cφsθ, cθcφ,  sφ+cφsθ,
                  cθ,  -sθ,       cθ
        ]))
    }
}

impl<T> Deref for Rot3ZX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3ZX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3ZX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.contract( rhs )
    }
}

use crate::rotation::x::Rot3X;

impl<T> Mul<Rot3X<T>> for Rot3ZX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn mul( self, rhs: Rot3X<T> ) -> Self::Output {
        Self( self.contract( rhs.0 ) )
    }
}

use crate::rotation::xz::Rot3XZ;

impl<T> Transpose for Rot3ZX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3XZ<T>;

    fn transpose( self ) -> Self::Output {
        Rot3XZ( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3ZX<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3XZ<T>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Rot3XZ( self.0.transpose() );
    }
}
