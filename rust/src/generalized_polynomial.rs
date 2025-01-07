// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Add, Sub, Mul, AddAssign, SubAssign, MulAssign, Deref, DerefMut },
};
use num::traits::{ Num, Pow };

use crate::{
    traits::ConstReSizeable,
    vector::Vector
};

#[derive(Debug, Clone)]
pub struct GeneralizedPolynomial<T, U, const TERM: usize>
where
    T: 'static + Debug + Copy + Default,
    U: 'static + Debug + Copy + Default
{
    terms: Vector<(T, U), TERM>,
}

impl<T, U, const TERM: usize> GeneralizedPolynomial<T, U, TERM>
where
    T: Debug + Copy + Default,
    U: Debug + Copy + Default
{
    pub const fn new_const( terms: [(T, U); TERM] ) -> Self {
        GeneralizedPolynomial {
            terms: Vector::new_const( terms )
        }
    }

    pub fn new( terms: [(T, U); TERM] ) -> Self {
        GeneralizedPolynomial {
            terms: Vector::from( terms )
        }
    }

    pub const fn terms( &self ) -> usize {
        TERM
    }

    pub fn resize<const NEW_TERM: usize>( &self ) -> GeneralizedPolynomial<T, U, NEW_TERM>
    where
        T: Default + Copy + Debug,
        U: Default + Copy + Debug,
        Self: ConstReSizeable
    {
        GeneralizedPolynomial::<T, U, NEW_TERM> { terms: self.terms.resize() }
    }

    pub fn evaluate( &self, x: T ) -> T
    where
        T: Mul<Output = T> + AddAssign + Pow<U, Output = T>
    {
        self.iter()
            .map( | &( a, b ) | a * x.pow( b ) )
            .fold( T::default(), |mut acc, val| { acc += val; acc }
        )
    }
}

impl<T, U, const TERM: usize> Default for GeneralizedPolynomial<T, U, TERM>
where
    T: 'static + Debug + Copy + Default,
    U: 'static + Debug + Copy + Default
{
    fn default() -> Self {
        GeneralizedPolynomial { terms: Vector::default() }
    }
}

impl<T, U, const TERM: usize> From<Vector<(T, U), TERM>> for GeneralizedPolynomial<T, U, TERM>
where
    T: 'static + Debug + Copy + Default,
    U: 'static + Debug + Copy + Default
{
    fn from( terms: Vector<(T, U), TERM> ) -> Self {
        GeneralizedPolynomial { terms }
    }
}

impl<T, U, const TERM: usize> Deref for GeneralizedPolynomial<T, U, TERM>
where
    T: 'static + Debug + Copy + Default,
    U: 'static + Debug + Copy + Default
{
    type Target = Vector<(T, U), TERM>;

    fn deref( &self ) -> &Self::Target {
        &self.terms
    }
}

impl<T, U, const TERM: usize> DerefMut for GeneralizedPolynomial<T, U, TERM>
where
    T: 'static + Debug + Copy + Default,
    U: 'static + Debug + Copy + Default
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.terms
    }
}

/*
impl<T, U, const TERM: usize> Add for GeneralizedPolynomial<T, U, TERM>
where
    T: Debug + Copy + Default + Add<Output = T>,
    U: 'static + Debug + Copy + Default
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        GeneralizedPolynomial { terms: self.terms + other.terms }
    }
}

impl<T, U, const TERM: usize> Sub for GeneralizedPolynomial<T, U, TERM>
where
    T: Debug + Copy + Default + Sub<Output = T>,
    U: 'static + Debug + Copy + Default
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        GeneralizedPolynomial { terms: self.terms - other.terms }
    }
}

impl<T, U, const TERM: usize> Mul for GeneralizedPolynomial<T, U, TERM>
where
    T: Debug + Copy + Default + AddAssign + Mul<Output = T>,
    U: 'static + Debug + Copy + Default
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let mut result = Vector::<T, TERM>::default();

        self.terms.iter().enumerate().for_each(
            |( i, &a )| { other.terms.iter().enumerate().for_each(
                 |( j, &b )| if i + j < TERM { result[ i + j ] += a * b; }
            )}
        );

        GeneralizedPolynomial { terms: result }
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate() {
        let polynomial = GeneralizedPolynomial::<i32, u32, 5>::new([ (1, 1), (2, 2), (3, 3), (4, 4), (5, 5) ]);
        assert_eq!( polynomial.evaluate( 1 ), 15 );
    }
}
