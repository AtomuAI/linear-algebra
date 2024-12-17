// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Add, Sub, Mul, AddAssign, SubAssign, MulAssign, Deref, DerefMut },
};
use num::traits::{ Num, Pow };

use crate::vector::Vector;

#[derive(Debug, Clone)]
pub struct Polynomial<T, const DEG: usize>
where
    T: 'static + Debug + Copy + Default
{
    coefficients: Vector<T, DEG>,
}

impl<T, const DEG: usize> Polynomial<T, DEG>
where
    T: Debug + Copy + Default
{
    pub const fn new_const( coefficients: [T; DEG] ) -> Self {
        Polynomial {
            coefficients: Vector::new_const( coefficients )
        }
    }

    pub fn new( coefficients: [T; DEG] ) -> Self {
        Polynomial {
            coefficients: Vector::from( coefficients )
        }
    }

    pub const fn deg( &self ) -> usize {
        DEG
    }

    pub fn resize<const NEW_DEG: usize>( &self ) -> Polynomial<T, NEW_DEG>
    where
        T: Default + Copy + Debug,
    {
        Polynomial::<T, NEW_DEG> { coefficients: self.coefficients.resize::<NEW_DEG>() }
    }

    pub fn evaluate( &self, x: T ) -> T
    where
        T: Mul<Output = T> + AddAssign + Pow<usize, Output = T>
    {
        self.iter().enumerate()
            .map( |( i, &a )| a * x.pow( i ) )
            .fold( T::default(), |mut acc, val| { acc += val; acc }
        )
    }
}

impl<T, const DEG: usize> Default for Polynomial<T, DEG>
where
    T: 'static + Debug + Copy + Default
{
    fn default() -> Self {
        Polynomial { coefficients: Vector::default() }
    }
}

impl<T, const DEG: usize> From<Vector<T, DEG>> for Polynomial<T, DEG>
where
    T: 'static + Debug + Copy + Default
{
    fn from( coefficients: Vector<T, DEG> ) -> Self {
        Polynomial { coefficients }
    }
}

impl<T, const DEG: usize> Deref for Polynomial<T, DEG>
where
    T: 'static + Debug + Copy + Default
{
    type Target = Vector<T, DEG>;

    fn deref(&self) -> &Self::Target {
        &self.coefficients
    }
}

impl<T, const DEG: usize> DerefMut for Polynomial<T, DEG>
where
    T: 'static + Debug + Copy + Default
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coefficients
    }
}

impl<T, const ORD: usize> Add for Polynomial<T, ORD>
where
    T: Debug + Copy + Default + Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Polynomial { coefficients: self.coefficients + other.coefficients }
    }
}

impl<T, const ORD: usize> Sub for Polynomial<T, ORD>
where
    T: Debug + Copy + Default + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Polynomial { coefficients: self.coefficients - other.coefficients }
    }
}

impl<T, const ORD: usize> Mul for Polynomial<T, ORD>
where
    T: Debug + Copy + Default + AddAssign + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let mut result = Vector::<T, ORD>::default();

        self.coefficients.iter().enumerate().for_each(
            |( i, &a )| { other.coefficients.iter().enumerate().for_each(
                 |( j, &b )| if i + j < ORD { result[ i + j ] += a * b; }
            )}
        );

        Polynomial { coefficients: result }
    }
}

pub type Monomial<T> = Polynomial<T, 1>;
pub type Binomial<T> = Polynomial<T, 2>;
pub type Trinomial<T> = Polynomial<T, 3>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_const() {
        let polynomial = Polynomial::new_const([ 1, 2, 3, 4, 5 ]);
        assert_eq!( polynomial.deg(), 5 );
    }

    #[test]
    fn test_new() {
        let polynomial = Polynomial::new([ 1, 2, 3, 4, 5 ]);
        assert_eq!( polynomial.deg(), 5 );
    }

    #[test]
    fn test_evaluate() {
        let polynomial = Polynomial::new([ 1, 2, 3, 4, 5 ]);
        assert_eq!( polynomial.evaluate( 1 ), 15 );
    }
}