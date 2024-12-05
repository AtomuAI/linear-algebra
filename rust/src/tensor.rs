// Copyright 2024 Bewusstsein Labs

//mod test;

use std::ops::{ Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign };

use memory::{ Memory, MemoryType, MemoryTraits, stack::Stack, heap::Heap };

use crate::{
    shape::Shape,
    //slice::Slice
};

#[ derive( Debug ) ]
pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

#[ derive( Clone ) ]
pub struct Tensor<T, const D: usize, M>
where
    T: 'static + Default + Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    shape: Shape<D>,
    memory: Memory<T, M>
}

impl<T, const D: usize, M> Default for Tensor<T, D, M>
where
    T: 'static + Default + Copy,
    M: MemoryType,
    M::Data<T>: Default,
    Memory<T, M>: MemoryTraits<Type = T> + Default
{
    fn default() -> Self {
        Self {
            shape: Shape::default(),
            memory: Memory::default()
        }
    }
}

impl<T, const D: usize, const N: usize> Tensor<T, D, Stack<N>>
where
T: 'static + Default + Copy
{
    pub fn new( shape: Shape<D> ) -> Self {
        Self {
            shape,
            memory: Memory::<T, Stack<N>>::new( () )
        }
    }

    pub fn take( memory: [T; N] ) -> Self {
        Self {
            shape: Shape::<D>::new(),
            memory: Memory::<T, Stack<N>>::take( memory )
        }
    }
}

impl<T, const D: usize> Tensor<T, D, Heap>
where
    T: 'static + Default + Copy
{
    pub fn new( shape: Shape<D> ) -> Self {
        let vol = shape.vol();
        let mut memory = Memory::<T, Heap>::new( vol );
        memory.resize( vol, T::default() );
        Self {
            shape,
            memory
        }
    }

    pub fn take( memory: Vec<T> ) -> Self {
        Self {
            shape: Shape::<D>::new(),
            memory: Memory::<T, Heap>::take( memory )
        }
    }
}

pub trait TensorTraits<T, const D: usize, M>
where
    T: 'static + Default + Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Sized
{
    fn dim( &self ) -> usize;
    fn size( &self ) -> usize;
    fn shape( &self ) -> &Shape<D>;
    //fn slice<'a>( &'a mut self, start: Shape<D>, end: Shape<D>, strides: Shape<D> ) -> Slice<'a, T, S, M>;
    fn reshape( &mut self, shape: Shape<D> ) -> Result<(), Error>;
    fn fill( &mut self, value: T );
    fn zero( &mut self );
    fn identity( &mut self, value: T );
    fn flat_idx( &self, index: &[usize; D] ) -> usize;
    fn iter( &self ) -> std::slice::Iter<T>;
    fn iter_mut( &mut self ) -> std::slice::IterMut<T>;
    fn add( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where T: Add<Output = T>;

    fn sub( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where T: Sub<Output = T>;

    fn mul( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where T: Mul<Output = T>;

    fn div( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where T: Div<Output = T>;

    fn product<O, P>(
        a: &Tensor<T, D, M>,
        b: &Tensor<T, D, O>,
        c: &mut Tensor<T, D, P>,
    ) -> Result<(), Error>
    where
        T: Mul<Output = T>,
        M: MemoryType,
        O: MemoryType,
        P: MemoryType,
        Tensor<T, D, M>: TensorTraits<T, D, M>,
        Tensor<T, D, O>: TensorTraits<T, D, O>,
        Tensor<T, D, P>: TensorTraits<T, D, P>,
        Memory<T, M>: MemoryTraits<Type = T>,
        Memory<T, O>: MemoryTraits<Type = T>,
        Memory<T, P>: MemoryTraits<Type = T>;
}

impl<T, const D: usize, M> TensorTraits<T, D, M> for Tensor<T, D, M>
where
    T: Default + Clone + Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
{
    fn dim( &self ) -> usize {
        self.shape.dim()
    }

    fn size( &self ) -> usize {
        self.memory.len()
    }

    fn shape( &self ) -> &Shape<D> {
        &self.shape
    }

    //fn slice<'a>( &'a mut self, start: Shape<D>, end: Shape<D>, strides: Shape<D> ) -> Slice<'a, T, S, M> {
    //    Slice::new( self, start, end, strides )
    //}

    fn reshape( &mut self, shape: Shape<D> ) -> Result<(), Error> {
        match shape.vol() == self.memory.len() {
            false => Err( Error::MismatchedSizes ),
            true => {
                self.shape = shape;
                Ok( () )
            }
        }
    }

    fn fill( &mut self, value: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] = value.clone();
        }
    }

    fn zero( &mut self ) {
        self.fill( T::default() );
    }

    fn identity( &mut self, value: T ) {
        for i in 0..self.memory.len() {
            if i % ( self.shape[ 0 ] + 1 ) == 0 {
                self.memory[ i ] = value.clone();
            } else {
                self.memory[ i ] = T::default();
            }
        }
    }

    fn flat_idx( &self, index: &[usize; D] ) -> usize {
        let dim = self.shape.dim();
        let mut flat_idx = 0;
        let mut stride = 1;
        for ( i, &dim_index ) in index.iter().rev().enumerate() {
            flat_idx += dim_index * stride;
            stride *= self.shape[ dim - 1 - i ];
        }
        flat_idx
    }

    fn iter( &self ) -> std::slice::Iter<T> {
        self.memory.iter()
    }

    fn iter_mut( &mut self ) -> std::slice::IterMut<T> {
        self.memory.iter_mut()
    }

    fn add( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where
        T: Add<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.memory.len() {
            c.memory[ i ] = a.memory[ i ] + b.memory[ i ];
        }
        Ok( () )
    }

    fn sub( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where
        T: Sub<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.memory.len() {
            c.memory[ i ] = a.memory[ i ] - b.memory[ i ];
        }
        Ok( () )
    }

    fn mul( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where
        T: Mul<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.memory.len() {
            c.memory[ i ] = a.memory[ i ] * b.memory[ i ];
        }
        Ok( () )
    }

    fn div( a: &Tensor<T, D, M>, b: &Tensor<T, D, M>, c: &mut Tensor<T, D, M> ) -> Result<(), Error>
    where
        T: Div<Output = T>,
    {
        if a.shape != b.shape || b.shape != c.shape {
            return Err( Error::MismatchedShapes );
        }
        for i in 0..a.memory.len() {
            c.memory[ i ] = a.memory[ i ] / b.memory[ i ];
        }
        Ok( () )
    }

    fn product<O, P>(
        a: &Tensor<T, D, M>,
        b: &Tensor<T, D, O>,
        c: &mut Tensor<T, D, P>,
    ) -> Result<(), Error>
    where
        T: Mul<Output = T>,
        M: MemoryType,
        O: MemoryType,
        P: MemoryType,
        Tensor<T, D, M>: TensorTraits<T, D, M>,
        Tensor<T, D, O>: TensorTraits<T, D, O>,
        Tensor<T, D, P>: TensorTraits<T, D, P>,
        Memory<T, M>: MemoryTraits<Type = T>,
        Memory<T, O>: MemoryTraits<Type = T>,
        Memory<T, P>: MemoryTraits<Type = T>
    {
        let a_dim = a.dim();
        let b_dim = b.dim();
        let c_dim = c.dim();
        let mut expected_shape = Shape::<D>::new();
        expected_shape[ 0 ] = c_dim;
        for i in 0..a_dim {
            expected_shape[ i ] = a.shape[ i ];
        }
        for i in 0..b_dim {
            expected_shape[ a_dim + i ] = b.shape[ i ];
        }

        if c.shape != expected_shape {
            return Err( Error::MismatchedShapes );
        }

        for ( i, a_elem ) in a.memory.iter().enumerate() {
            for ( j, b_elem ) in b.memory.iter().enumerate() {
                let result_index = i * b.memory.len() + j;
                c.memory[ result_index ] = *a_elem * *b_elem;
            }
        }

        Ok( () )
    }
}

impl<T, const D: usize, M> Add for Tensor<T, D, M>
where
    T: Default + Copy + Add<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn add( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] + other.memory[ i ];
        }
        result
    }
}

impl<T, const D: usize, M> Sub for Tensor<T, D, M>
where
    T: Default + Copy + Sub<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] - other.memory[ i ];
        }
        result
    }
}

impl<T, const D: usize, M> Mul for Tensor<T, D, M>
where
    T: Default + Copy + Mul<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] * other.memory[ i ];
        }
        result
    }
}

impl<T, const D: usize, M> Div for Tensor<T, D, M>
where
    T: Default + Copy + Div<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn div( self, other: Self ) -> Self::Output {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] / other.memory[ i ];
        }
        result
    }
}

impl<T, const D: usize, M> Add<T> for Tensor<T, D, M>
where
    T: Default + Copy + Add<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn add( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] + scalar;
        }
        result
    }
}

impl<T, const D: usize, M> Sub<T> for Tensor<T, D, M>
where
    T: Default + Copy + Sub<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] - scalar;
        }
        result
    }
}

impl<T, const D: usize, M> Mul<T> for Tensor<T, D, M>
where
    T: Default + Copy + Mul<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] * scalar;
        }
        result
    }
}

impl<T, const D: usize, M> Div<T> for Tensor<T, D, M>
where
    T: Default + Copy + Div<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Clone
{
    type Output = Self;

    fn div( self, scalar: T ) -> Self::Output {
        let mut result = self.clone();
        for i in 0..self.memory.len() {
            result.memory[ i ] = self.memory[ i ] / scalar;
        }
        result
    }
}

impl<T, const D: usize, M> AddAssign for Tensor<T, D, M>
where
    T: Default + Copy + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn add_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.memory.len() {
            self.memory[ i ] += other.memory[ i ];
        }
    }
}

impl<T, const D: usize, M> SubAssign for Tensor<T, D, M>
where
    T: Default + Copy + SubAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn sub_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.memory.len() {
            self.memory[ i ] -= other.memory[ i ];
        }
    }
}

impl<T, const D: usize, M> MulAssign for Tensor<T, D, M>
where
    T: Default + Copy + MulAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn mul_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.memory.len() {
            self.memory[ i ] *= other.memory[ i ];
        }
    }
}

impl<T, const D: usize, M> DivAssign for Tensor<T, D, M>
where
    T: Default + Copy + DivAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn div_assign( &mut self, other: Self ) {
        if other.shape != self.shape {
            panic!( "Mismatched shapes" );
        }
        for i in 0..self.memory.len() {
            self.memory[ i ] /= other.memory[ i ];
        }
    }
}

impl<T, const D: usize, M> AddAssign<T> for Tensor<T, D, M>
where
    T: Default + Copy + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn add_assign( &mut self, scalar: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] += scalar;
        }
    }
}

impl<T, const D: usize, M> SubAssign<T> for Tensor<T, D, M>
where
    T: Default + Copy + SubAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn sub_assign( &mut self, scalar: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] -= scalar;
        }
    }
}

impl<T, const D: usize, M> MulAssign<T> for Tensor<T, D, M>
where
    T: Default + Copy + MulAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn mul_assign( &mut self, scalar: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] *= scalar;
        }
    }
}

impl<T, const D: usize, M> DivAssign<T> for Tensor<T, D, M>
where
    T: Default + Copy + DivAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn div_assign( &mut self, scalar: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] /= scalar;
        }
    }
}

// Dimensional Indexing
impl<T, const D: usize, M> Index<[usize; D]> for Tensor<T, D, M>
where
    T: Default + Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: TensorTraits<T, D, M>
{
    type Output = T;

    fn index( &self, index: [usize; D] ) -> &Self::Output {
        let flat_idx = self.flat_idx( &index );
        &self.memory[ flat_idx ]
    }
}

impl<T, const D: usize, M> IndexMut<[usize; D]> for Tensor<T, D, M>
where
    T: Default + Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: TensorTraits<T, D, M>
{
    fn index_mut( &mut self, index: [usize; D] ) -> &mut Self::Output {
        let flat_idx = self.flat_idx( &index );
        &mut self.memory[ flat_idx ]
    }
}

// Flat Indexing
impl<T, const D: usize, M> Index<usize> for Tensor<T, D, M>
where
    T: Default + Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.memory[ index ]
    }
}

impl<T, const D: usize, M> IndexMut<usize> for Tensor<T, D, M>
where
    T: Default + Copy,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.memory[ index ]
    }
}

#[cfg( test )]
mod tests {
    use super::*;

    #[ test ]
    fn test_iter() {
        let mut tensor = Tensor::<f32, 1, Heap>::new( Shape::<1>::from( [3] ) );
        tensor[ 0 ] = 1.0;
        tensor[ 1 ] = 2.0;
        tensor[ 2 ] = 3.0;

        let mut iter = tensor.iter();
        assert_eq!( iter.next(), Some( &1.0 ) );
        assert_eq!( iter.next(), Some( &2.0 ) );
        assert_eq!( iter.next(), Some( &3.0 ) );
        assert_eq!( iter.next(), None );
    }
}
