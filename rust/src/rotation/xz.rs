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
pub struct Rot3XZ<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3XZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    pub fn new( roll: T, yaw: T ) -> Self {
        let cφ = roll.cos();
        let sφ = roll.sin();
        let cθ = yaw.cos();
        let sθ = yaw.sin();
        let cφcθ = cφ * cθ;
        let sφcθ = sφ * cθ;
        let cφsθ = cφ * sθ;
        let sφsθ = sφ * sθ;
        Self ( Matrix3x3::new([
                  cθ,    sθ,      -cθ,
            -cφsθ+sφ,  cφcθ,  cφsθ+sφ,
             sφsθ+cφ, -sφcθ, -sφsθ+cφ
        ]))
    }
}

impl<T> Deref for Rot3XZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3XZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3XZ<T>
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

impl<T> Mul<Rot3Z<T>> for Rot3XZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float,
    Matrix3x3<T>: MatrixMul<Matrix3x3<T>, Output = Matrix3x3<T>>
{
    type Output = Self;

    fn mul( self, rhs: Rot3Z<T> ) -> Self::Output {
        Self( self.mat_mul( rhs.0 ) )
    }
}

use crate::rotation::zx::Rot3ZX;

impl<T> Transpose for Rot3XZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3ZX<T>;

    fn transpose( self ) -> Self::Output {
        Rot3ZX( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3XZ<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3ZX<T>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Rot3ZX( self.0.transpose() );
    }
}
