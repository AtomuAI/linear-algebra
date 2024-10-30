// Copyright 2024 Bewusstsein Labs

mod test;

use std::ops::{ Index, IndexMut };

use bewusstsein::memory::{
    memory::Memory,
    storage::{
        Storage,
        owned::Owned
    }
};

#[derive( Debug )]
pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

#[derive( Debug, Clone, Copy )]
pub struct Shape<M>
where
    M: Memory<usize>
{
    storage: Storage<usize, M>
}

impl<M> Shape<M>
where
    M: Memory<usize>
{
    pub fn new( dim: usize ) -> Self {
        Shape {
            storage: Storage<M>::new( dim )
        }
    }

    pub fn from( data: M::Type ) -> Self {
        Shape {
            storage: Storage<M>::from( data )
        }
    }

    pub fn dim( &self ) -> usize {
        self.storage.cap()
    }

    pub fn vol( &self ) -> usize {
        self.storage.iter().fold( 1, |acc, &x| acc * x )
    }
}

impl<T> Index<usize> for Shape<M> {
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.storage[ index ]
    }
}

impl<T> IndexMut<usize> for Shape<M> {
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.storage[ index ]
    }
}