// Copyright 2024 Bewusstsein Labs

mod test;

use std::ops::{ Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign };

use bewusstsein::memory::{
    memory::Memory,
    storage::{
        Storage,
        owned::Owned
    }
};

use crate::{
    shape::Shape,
    slice::Slice
};

#[ derive( Debug ) ]
pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

#[derive( Debug, Clone)]
pub struct Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq,
    S: Storage<T, M>,
    M: Storage<T, M>

{
    shape: Shape<S>,
    storage: Storage<T, M>
}

impl<T, S, M> Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq,
    S: Memory<usize>,
    M: Memory<T>
{
    fn new( shape: Shape<S> ) -> Self {
        let vol = shape.vol();
        Self {
            shape,
            storage: Storage::new::<T, M>( vol )
        }
    }

    fn from( shape: Shape<S>, src: &[T] ) -> Result<Self, Error> {
        match shape.vol() == src.len() {
            false => Err( Error::MismatchedSizes ),
            true => Ok( Self {
                shape,
                storage: Storage::from::<T, M>( src )
            })
        }

    }

    fn dim( &self ) -> usize {
        self.shape.dim()
    }

    fn size( &self ) -> usize {
        self.storage.len()
    }

    fn shape( &self ) -> &Shape<S> {
        &self.shape
    }

    fn slice<'a>( &'a mut self, start: Shape<S>, end: Shape<S>, strides: Shape<S> ) -> Slice<'a, T, S, M> {
        Slice::new( self, start, end, strides )
    }

    fn reshape( &mut self, shape: Shape<S> ) -> Result<(), Error> {
        match shape.vol() == self.storage.len() {
            false => Err( Error::MismatchedSizes ),
            true => {
                self.shape = shape;
                Ok( () )
            }
        }
    }

    fn fill( &mut self, value: T ) {
        for i in 0..self.storage.len() {
            self.storage[ i ] = value.clone();
        }
    }

    fn zero( &mut self ) {
        self.fill( T::default() );
    }

    fn identity( &mut self, value: T ) {
        for i in 0..self.storage.len() {
            if i % ( self.shape[ 0 ] + 1 ) == 0 {
                self.storage[ i ] = value.clone();
            } else {
                self.storage[ i ] = T::default();
            }
        }
    }

    fn set( &mut self, storage: M::Type ) -> Result<(), Error> {
        match storage.len() == self.storage.len() {
            false => Err( Error::MismatchedSizes ),
            true => {
                self.storage = storage;
                Ok( () )
            }
        }
    }

    fn flat_idx( &self, index: &Shape<S> ) -> usize {
        let dim = self.shape.dim();
        let mut flat_idx = 0;
        let mut stride = 1;
        for ( i, &dim_index ) in index.iter().rev().enumerate() {
            flat_idx += dim_index * stride;
            stride *= self.shape[ dim - 1 - i ];
        }
        flat_idx
    }

    fn add( a: &Tensor<T, S, M>, b: &Tensor<T, S, M>, c: &mut Tensor<T, S, M> ) -> Result<(), Error>
    where
        T: Add<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.storage.len() {
            c.storage[ i ] = a.storage[ i ].clone() + b.storage[ i ].clone();
        }
        Ok( () )
    }

    fn sub( a: &Tensor<T, S, M>, b: &Tensor<T, S, M>, c: &mut Tensor<T, S, M> ) -> Result<(), Error>
    where
        T: Sub<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.storage.len() {
            c.storage[ i ] = a.storage[ i ].clone() - b.storage[ i ].clone();
        }
        Ok( () )
    }

    fn mul( a: &Tensor<T, S, M>, b: &Tensor<T, S, M>, c: &mut Tensor<T, S, M> ) -> Result<(), Error>
    where
        T: Mul<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.storage.len() {
            c.storage[ i ] = a.storage[ i ].clone() * b.storage[ i ].clone();
        }
        Ok( () )
    }

    fn div( a: &Tensor<T, S, M>, b: &Tensor<T, S, M>, c: &mut Tensor<T, S, M> ) -> Result<(), Error>
    where
        T: Div<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.storage.len() {
            c.storage[ i ] = a.storage[ i ].clone() / b.storage[ i ].clone();
        }
        Ok( () )
    }

    fn product<O, P>(
        a: &Tensor<T, S, M>,
        b: &Tensor<T, O, M>,
        c: &mut Tensor<T, P, M>,
    ) -> Result<(), Error>
    where
        T: Mul<Output = T>,
        O: Memory<usize>,
        P: Memory<usize>,
    {
        let a_dim = a.dim();
        let b_dim = b.dim();
        let c_dim = c.dim();
        let mut expected_shape = Shape::new::<P>( c_dim );
        for i in 0..a_dim {
            expected_shape[ i ] = a.shape[ i ];
        }
        for i in 0..b_dim {
            expected_shape[ a_dim + i ] = b.shape[ i ];
        }

        if c.shape != expected_shape {
            return Err( Error::MismatchedShapes );
        }

        for ( i, a_elem ) in a.storage.iter().enumerate() {
            for ( j, b_elem ) in b.storage.iter().enumerate() {
                let result_index = i * b.storage.len() + j;
                c.storage[ result_index ] = a_elem.clone() * b_elem.clone();
            }
        }

        Ok( () )
    }
}

impl<T, S, M> Add for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Add<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn add( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() + other.storage[ i ].clone();
        }
        result
    }
}

impl<T, S, M> Sub for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Sub<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn sub( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() - other.storage[ i ].clone();
        }
        result
    }
}

impl<T, S, M> Mul for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Mul<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn mul( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() * other.storage[ i ].clone();
        }
        result
    }
}

impl<T, S, M> Div for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Div<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn div( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() / other.storage[ i ].clone();
        }
        result
    }
}

impl<T, S, M> Add<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Add<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn add( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() + scalar.clone();
        }
        result
    }
}

impl<T, S, M> Sub<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Sub<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn sub( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() - scalar.clone();
        }
        result
    }
}

impl<T, S, M:> Mul<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Mul<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn mul( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() * scalar.clone();
        }
        result
    }
}

impl<T, S, M> Div<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + Div<Output = T>,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = Self;

    fn div( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.storage.len() {
            result.storage[ i ] = self.storage[ i ].clone() / scalar.clone();
        }
        result
    }
}

impl<T, S, M> AddAssign for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + AddAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn add_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.storage.len() {
            self.storage[ i ] += other.storage[ i ].clone();
        }
    }
}

impl<T, S, M> SubAssign for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + SubAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn sub_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.storage.len() {
            self.storage[ i ] -= other.storage[ i ].clone();
        }
    }
}

impl<T, S, M> MulAssign for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + MulAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn mul_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.storage.len() {
            self.storage[ i ] *= other.storage[ i ].clone();
        }
    }
}

impl<T, S, M> DivAssign for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + DivAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn div_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.storage.len() {
            self.storage[ i ] /= other.storage[ i ].clone();
        }
    }
}

impl<T, S, M> AddAssign<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + AddAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn add_assign( &mut self, scalar: T ) {
        for i in 0..self.storage.len() {
            self.storage[ i ] += scalar.clone();
        }
    }
}

impl<T, S, M> SubAssign<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + SubAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn sub_assign( &mut self, scalar: T ) {
        for i in 0..self.storage.len() {
            self.storage[ i ] -= scalar.clone();
        }
    }
}

impl<T, S, M> MulAssign<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + MulAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn mul_assign( &mut self, scalar: T ) {
        for i in 0..self.storage.len() {
            self.storage[ i ] *= scalar.clone();
        }
    }
}

impl<T, S, M> DivAssign<T> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq + DivAssign,
    S: Memory<usize>,
    M: Memory<T>
{
    fn div_assign( &mut self, scalar: T ) {
        for i in 0..self.storage.len() {
            self.storage[ i ] /= scalar.clone();
        }
    }
}

// Dimensional Indexing
impl<T, S, M> Index<Shape<S>> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = T;

    fn index( &self, index: Shape<S> ) -> &Self::Output {
        let flat_idx = self.flat_idx( &index );
        &self.storage[ flat_idx ]
    }
}

impl<T, S, M> IndexMut<Shape<S>> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq,
    S: Memory<usize>,
    M: Memory<T>
{
    fn index_mut( &mut self, index: Shape<S> ) -> &mut Self::Output {
        let flat_idx = self.flat_idx( &index );
        &mut self.storage[ flat_idx ]
    }
}

// Flat Indexing
impl<T, S, M> Index<usize> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq,
    S: Memory<usize>,
    M: Memory<T>
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.storage[ index ]
    }
}

impl<T, S, M> IndexMut<usize> for Tensor<T, S, M>
where
    T: Default + Clone + Copy + PartialEq,
    S: Memory<usize>,
    M: Memory<T>
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.base.storage[ index ]
    }
}