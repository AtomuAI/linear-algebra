// Copyright 2024 Bewusstsein Labs

//mod test;

use std::ops::{ Deref, DerefMut };

use const_expr_bounds::{ Assert, IsTrue, IsFalse };

#[derive( Debug )]
pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

#[derive( Debug, Clone, Copy )]
pub struct Shape<const N: usize> ( [usize; N] );

impl<const N: usize> Shape<N> {
    pub const fn new_const( dim: [usize; N] ) -> Self {
        Self ( dim )
    }

    pub fn new() -> Self {
        Self ( [0; N] )
    }

    #[inline(always)]
    pub const fn ord( &self ) -> usize {
        N
    }

    pub fn vol( &self ) -> usize {
        self.iter().product()
    }
}

impl<const N: usize> Default for Shape<N> {
    fn default() -> Self {
        Self ( [0; N] )
    }
}

impl<const N: usize> PartialEq for Shape<N> {
    fn eq( &self, other: &Self ) -> bool {
        self.0 == other.0
    }
}

impl<const N: usize> Eq for Shape<N> {}

impl<const N: usize> Deref for Shape<N> {
    type Target = [usize; N];

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> DerefMut for Shape<N> {
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize> From<[usize; N]> for Shape<N> {
    fn from( src: [usize; N] ) -> Self {
        Self ( src )
    }
}

/*
impl<const LHS_N: usize> From<Shape<{LHS_N - 1}>> for Shape<LHS_N>
where
    Assert<{LHS_N - 1 > 0}>: IsTrue,
    [(); LHS_N - 1]:
{
    fn from( src: Shape<{LHS_N - 1}> ) -> Self {
        let mut this = Self::default();
        this.iter_mut().zip( src.iter() ).for_each( |( a, b )| *a = *b );
        this
    }
}
    */
