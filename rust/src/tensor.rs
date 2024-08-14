// Copyright 2024 Bewusstsein Labs

mod test;

use std::ops::{ Index, IndexMut };

pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

pub struct Tensor<T, const N: usize> {
    shape: [ usize; N ],
    data: Vec<T>,
}

impl<T, const N: usize> Tensor<T, N>
where
    T: Default + Clone + PartialEq,
{
    pub fn new( shape: [ usize; N ] ) -> Self {
        let size = shape.iter().fold( 1, |acc, &x| acc * x );
        let data = vec![ T::default(); size as usize ];
        Tensor { shape, data }
    }

    pub fn dim( &self ) -> usize {
        self.shape.len()
    }

    pub fn size( &self ) -> usize {
        self.data.len()
    }

    pub fn shape( &self ) -> [ usize; N ] {
        self.shape
    }

    pub fn reshape( &mut self, shape: [ usize; N ] ) -> Result<(), Error> {
        if shape.iter().fold( 1, |acc, &x| acc * x ) != self.data.len() {
            return Err( Error::MismatchedSizes );
        }
        self.shape = shape;
        Ok( () )
    }

    pub fn fill( &mut self, value: T ) {
        for i in 0..self.data.len() {
            self.data[ i ] = value.clone();
        }
    }

    pub fn zero( &mut self ) {
        self.fill( T::default() );
    }

    pub fn identity( &mut self, value: T ) {
        for i in 0..self.data.len() {
            if i % ( self.shape[ 0 ] + 1 ) == 0 {
                self.data[ i ] = value.clone();
            } else {
                self.data[ i ] = T::default();
            }
        }
    }

    pub fn set( &mut self, data: Vec<T> ) -> Result<(), Error> {
        if data.len() != self.data.len() {
            return Err( Error::MismatchedSizes );
        }
        self.data = data;
        Ok( () )
    }

    fn calculate_flat_index( &self, index: &[ usize; N ] ) -> usize {
        let mut flat_index = 0;
        let mut stride = 1;
        for ( i, &dim_index ) in index.iter().rev().enumerate() {
            flat_index += dim_index * stride;
            stride *= self.shape[ N - 1 - i ];
        }
        flat_index
    }

    pub fn tensor_mult( lhs: &Tensor<T, N>, rhs: &Tensor<T, N> ) -> Result<Tensor<T, N>, Error>
    where
        T: Default + Clone + PartialEq + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    {
        if lhs.shape != rhs.shape {
            return Err( Error::MismatchedShapes );
        }
        let mut result = Tensor::new( lhs.shape );
        for i in 0..lhs.data.len() {
            result.data[ i ] = lhs.data[ i ].clone() * rhs.data[ i ].clone();
        }
        Ok( result )
    }

    pub fn tensor_add( lhs: &Tensor<T, N>, rhs: &Tensor<T, N> ) -> Result<Tensor<T, N>, Error>
    where
        T: Default + Clone + PartialEq + std::ops::Add<Output = T>,
    {
        if lhs.shape != rhs.shape {
            return Err( Error::MismatchedShapes );
        }
        let mut result = Tensor::new( lhs.shape );
        for i in 0..lhs.data.len() {
            result.data[ i ] = lhs.data[ i ].clone() + rhs.data[ i ].clone();
        }
        Ok( result )
    }

    pub fn tensor_sub( lhs: &Tensor<T, N>, rhs: &Tensor<T, N> ) -> Result<Tensor<T, N>, Error>
    where
        T: Default + Clone + PartialEq + std::ops::Sub<Output = T>,
    {
        if lhs.shape != rhs.shape {
            return Err( Error::MismatchedShapes );
        }
        let mut result = Tensor::new( lhs.shape );
        for i in 0..lhs.data.len() {
            result.data[ i ] = lhs.data[ i ].clone() - rhs.data[ i ].clone();
        }
        Ok( result )
    }

    pub fn tensor_div( lhs: &Tensor<T, N>, rhs: &Tensor<T, N> ) -> Result<Tensor<T, N>, Error>
    where
        T: Default + Clone + PartialEq + std::ops::Div<Output = T>,
    {
        if lhs.shape != rhs.shape {
            return Err( Error::MismatchedShapes );
        }
        let mut result = Tensor::new( lhs.shape );
        for i in 0..lhs.data.len() {
            result.data[ i ] = lhs.data[ i ].clone() / rhs.data[ i ].clone();
        }
        Ok( result )
    }

    pub fn tensor_contraction( lhs: &Tensor<T, N>, rhs: &Tensor<T, N>, i: usize, j: usize  ) -> Result<Tensor<T, N>, Error>
    where
        T: Default + Clone + PartialEq + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    {
        if lhs.shape[ 1 ] != rhs.shape[ 0 ] {
            return Err( Error::MismatchedShapes );
        }
        let mut result = Tensor::new( [ lhs.shape[ i ], rhs.shape[ j ] ] );
        for i in 0..lhs.shape[ i ] {
            for j in 0..rhs.shape[ j ] {
                for k in 0..lhs.shape[ j ] {
                    result[ [ i, j ] ] = result[ [ i, j ] ].clone() + lhs[ [ i, k ] ].clone() * rhs[ [ k, j ] ].clone();
                }
            }
        }
        Ok( result )
    }
}

impl<T, const N: usize> Index<[ usize; N ]> for Tensor<T, N>
where
    T: Default + Clone + PartialEq,
{
    type Output = T;

    fn index( &self, index: [ usize; N ] ) -> &Self::Output {
        let flat_index = self.calculate_flat_index( &index );
        &self.data[ flat_index ]
    }
}

impl<T, const N: usize> IndexMut<[ usize; N ]> for Tensor<T, N>
where
    T: Default + Clone + PartialEq,
{
    fn index_mut( &mut self, index: [ usize; N ] ) -> &mut Self::Output {
        let flat_index = self.calculate_flat_index( &index );
        &mut self.data[ flat_index ]
    }
}