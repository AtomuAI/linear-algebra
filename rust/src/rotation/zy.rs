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
pub struct Rot3ZY<T>( pub(crate) Matrix3x3<T> )
where
    T: 'static + Default + Copy + Debug;

impl<T> Rot3ZY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    pub fn new( pitch: T, yaw: T ) -> Self {
        let cψ = pitch.cos();
        let sψ = pitch.sin();
        let cθ = yaw.cos();
        let sθ = yaw.sin();
        let sψcθ = sψ * cθ;
        let cψcθ = cψ * cθ;
        let sθsψ = sθ * sψ;
        let sθcψ = sθ * cψ;
        Self ( Matrix3x3::new([
             cθ, cψ+sθsψ, sψ-sθcψ,
            -cθ, cψ-sθsψ, sψ+sθcψ,
             sθ,   -sψcθ,    cψcθ
        ]))
    }
}

impl<T> Deref for Rot3ZY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    type Target = Matrix3x3<T>;

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

use crate::rotation::Rot3;

impl<T> From<Rot3<T>> for Rot3ZY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + Num + Float
{
    fn from( rot: Rot3<T> ) -> Self {
        Self( rot.0 )
    }
}

impl<T> Mul<Vector3<T>> for Rot3ZY<T>
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

impl<T> Mul<Rot3Y<T>> for Rot3ZY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float,
    Matrix3x3<T>: MatrixMul<Matrix3x3<T>, Output = Matrix3x3<T>>
{
    type Output = Self;

    fn mul( self, rhs: Rot3Y<T> ) -> Self::Output {
        Self( self.mat_mul( rhs.0 ) )
    }
}

use crate::rotation::yz::Rot3YZ;

impl<T> Transpose for Rot3ZY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3YZ<T>;

    fn transpose( self ) -> Self::Output {
        Rot3YZ( self.0.transpose() )
    }
}

impl<T> TransposeAssignTo for Rot3ZY<T>
where
    T: Num + 'static + Default + Copy + Debug + Neg<Output = T> + AddAssign + Num + Float
{
    type Output = Rot3YZ<T>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        *res = Rot3YZ( self.0.transpose() );
    }
}
