// Copyright 2024 Bewusstsein Labs

//mod test;

use std::ops::{ Deref, DerefMut };

#[derive( Debug )]
pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

#[derive( Debug, Clone, Copy )]
pub struct Shape<const N: usize> {
    dim: [usize; N]
}

impl<const N: usize> Shape<N> {
    pub fn new() -> Self {
        Self {
            dim: [0; N]
        }
    }

    pub fn dim( &self ) -> usize {
        N
    }

    pub fn vol( &self ) -> usize {
        self.dim.iter().fold( 1, |acc, &x| acc * x )
    }
}

impl<const N: usize> Default for Shape<N> {
    fn default() -> Self {
        Self { dim: [0; N] }
    }
}

impl<const N: usize> PartialEq for Shape<N> {
    fn eq( &self, other: &Self ) -> bool {
        self.dim == other.dim
    }
}

impl<const N: usize> Eq for Shape<N> {}

impl<const N: usize> Deref for Shape<N> {
    type Target = [usize; N];

    fn deref( &self ) -> &Self::Target {
        &self.dim
    }
}

impl<const N: usize> DerefMut for Shape<N> {
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.dim
    }
}

impl<const N: usize> From<[usize; N]> for Shape<N> {
    fn from( slice: [usize; N] ) -> Self {
        Self {
            dim: slice
        }
    }
}
