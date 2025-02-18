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
pub struct Rot3YXY<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3YXY<T>
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
        let sθsφ = sθ * sφ;
        let cφsψ = cφ * sψ;
        let sθsψ = sθ * sψ;
        let sθcψ = sθ * cψ;
        let sφcψ = sφ * cψ;
        let sθcφ = sθ * cφ;
        let sφsψ = sφ * sψ;
        let sφcθ = sφ * cθ;
        let cφcθ = cφ * cθ;
        let sφcθsψ = sφcθ * sψ;
        let sφcθcψ = sφcθ * cψ;
        let cφcθsψ = cφcθ * sψ;
        let cφcθcψ = cφcθ * cψ;
        Self ( Matrix3x3::new([
            cφcψ-sφcθsψ,  sθsφ, -cφsψ-sφcθcψ,
                   sθsψ,    cθ,         sθcψ,
            sφcψ+cφcθsψ, -sθcφ, -sφsψ+cφcθcψ
        ]))
    }
}

impl<T> Deref for Rot3YXY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3YXY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3YXY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float,
    Matrix3x3<T>: MatrixMul<Vector3<T>, Output = Vector3<T>>
{
    type Output = Vector3<T>;

    fn mul( self, rhs: Vector3<T> ) -> Self::Output {
        self.mat_mul( rhs )
    }
}

use crate::rotation::y::Rot3Y;

impl<T> Mul<Rot3Y<T>> for Rot3YXY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float,
    Matrix3x3<T>: MatrixMul<Matrix3x3<T>, Output = Matrix3x3<T>>
{
    type Output = Self;

    fn mul( self, rhs: Rot3Y<T> ) -> Self::Output {
        Self( self.mat_mul( rhs.0 ) )
    }
}

impl<T> Transpose for Rot3YXY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn transpose( self ) -> Self::Output {
        Self( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3YXY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Self;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Self( self.0.transpose() );
    }
}
