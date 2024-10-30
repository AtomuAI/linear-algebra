use std::ops::{ Deref, DerefMut };

use bewusstsein::memory::{ Memory, Dynamic, Static };

#[derive(Debug)]
pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

pub struct Base<T, const N: usize, M>
where
    T: Default + Clone + Copy + PartialEq,
    M: Memory<T>
{
    pub shape: [ usize; N ],
    pub data: M::Type,
}

pub trait BaseTraits<T, const N: usize, M>
where
    Self: Sized,
    T: Default + Clone + Copy + PartialEq,
    M: Memory<T>
{
    fn new( shape: [ usize; N ] ) -> Result<Self, Error>;
    fn from( shape: [ usize; N ], data: M::Type ) -> Result<Self, Error>;
    fn dim( &self ) -> usize;
    fn size( &self ) -> usize;
    fn shape( &self ) -> [ usize; N ];
}

impl<T, const N: usize> BaseTraits<T, N, Dynamic> for Base<T, N, Dynamic>
where
    T: Default + Clone + Copy + PartialEq
{
    fn new( shape: [ usize; N ] ) -> Result<Self, Error> {
        let size = shape.iter().fold( 1, |acc, &x| acc * x );
        let data = vec![ T::default(); size as usize ];
        Ok( Base { shape, data } )
    }

    fn from( shape: [ usize; N ], data: <Dynamic as Memory<T>>::Type ) -> Result<Self, Error> {
        if data.len() != shape.iter().fold( 1, |acc, &x| acc * x ) {
            return Err( Error::MismatchedSizes );
        }
        Ok( Base { shape, data } )
    }

    fn dim( &self ) -> usize {
        self.shape.len()
    }

    fn size( &self ) -> usize {
        self.data.len()
    }

    fn shape( &self ) -> [ usize; N ] {
        self.shape
    }
}

impl<T, const N: usize, const O: usize> BaseTraits<T, N, Static<O>> for Base<T, N, Static<O>>
where
    Self: Sized,
    T: Default + Clone + Copy + PartialEq
{
    fn new( shape: [ usize; N ] ) -> Result<Self, Error> {
        let size = O;
        let data = [ T::default(); O ];
        if shape.iter().fold( 1, |acc, &x| acc * x ) != O {
            panic!("Mismatched sizes");
        }
        Ok( Base { shape, data } )
    }

    fn from( shape: [ usize; N ], data: <Static<O> as Memory<T>>::Type ) -> Result<Self, Error> {
        if O != shape.iter().fold( 1, |acc, &x| acc * x ) {
            return Err( Error::MismatchedSizes );
        }
        Ok( Base { shape, data } )
    }

    fn dim( &self ) -> usize {
        N
    }

    fn size( &self ) -> usize {
        O
    }

    fn shape( &self ) -> [ usize; N ] {
        self.shape
    }
}

impl<T, const N: usize, M> Deref for Base<T, N, M>
where
    Self: Sized,
    T: Default + Clone + Copy + PartialEq,
    M: Memory<T>
{
    type Target = M::Type;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, const N: usize, M> DerefMut for Base<T, N, M>
where
    Self: Sized,
    T: Default + Clone + Copy + PartialEq,
    M: Memory<T>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}