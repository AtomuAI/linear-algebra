// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign }
};

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

#[ derive( Clone, Debug ) ]
pub struct Tensor<T, const DIM: usize, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    shape: Shape<DIM>,
    memory: Memory<T, M>
}

impl<T, const DIM: usize, M> Copy for Tensor<T, DIM, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType + Clone,
    Memory<T, M>: MemoryTraits<Type = T> + Copy
{}

impl<T, const DIM: usize, M> Default for Tensor<T, DIM, M>
where
    T: 'static + Default + Copy + Debug,
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

impl<T, const DIM: usize, M> PartialEq for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T> + PartialEq
{
    fn eq( &self, other: &Self ) -> bool {
        self.shape == other.shape &&
        self.memory == other.memory
    }
}

#[allow(clippy::needless_lifetimes)]
trait TensorAccess<T, const DIM: usize, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Sized
{
    fn _shape<'a>( &'a self ) -> &'a Shape<DIM>;
    fn memory<'a>( &'a self ) -> &'a Memory<T, M>;
    fn shape_mut<'b>( &'b mut self ) -> &'b mut Shape<DIM>;
    fn memory_mut<'b>( &'b mut self ) -> &'b mut Memory<T, M>;
}

pub trait TensorTraits<T, const DIM: usize, M>: TensorAccess<T, DIM, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Sized
{
    fn new( shape: Shape<DIM> ) -> Self;
    fn take( shape: Shape<DIM>, src: <Memory<T, M> as MemoryTraits>::Take ) -> Self;

    fn dim( &self ) -> usize {
        self._shape().dim()
    }

    fn size( &self ) -> usize {
        self.memory().len()
    }

    fn shape( &self ) -> &Shape<DIM> {
        self._shape()
    }

    //fn slice<'a>( &'a mut self, start: Shape<DIM>, end: Shape<DIM>, strides: Shape<DIM> ) -> Slice<'a, T, S, M>;

    fn reshape( &mut self, shape: Shape<DIM> ) -> Result<(), Error> {
        match shape.vol() == self.memory().len() {
            false => Err( Error::MismatchedSizes ),
            true => {
                *self.shape_mut() = shape;
                Ok( () )
            }
        }
    }

    fn fill( &mut self, value: T ) {
        for i in 0..self.memory().len() {
            self.memory_mut()[ i ] = value;
        }
    }

    fn zero( &mut self ) {
        self.fill( T::default() );
    }

    fn identity( &mut self, value: T ) {
        for i in 0..self.memory().len() {
            if i % ( self.shape()[ 0 ] + 1 ) == 0 {
                self.memory_mut()[ i ] = value;
            } else {
                self.memory_mut()[ i ] = T::default();
            }
        }
    }

    fn flat_idx( &self, index: &[usize; DIM] ) -> usize {
        let dim = self.shape().dim();
        let mut flat_idx = 0;
        let mut stride = 1;
        for ( i, &dim_index ) in index.iter().rev().enumerate() {
            flat_idx += dim_index * stride;
            stride *= self.shape()[ dim - 1 - i ];
        }
        flat_idx
    }

    fn iter<'a>( &'a self ) -> std::slice::Iter<'a, T>
    where
        M: 'a
    {
        self.memory().iter()
    }

    fn iter_mut<'a>( &'a mut self ) -> std::slice::IterMut<'a, T>
    where
        M: 'a
    {
        self.memory_mut().iter_mut()
    }
}

#[allow(clippy::needless_lifetimes)]
impl<T, const DIM: usize, M> TensorAccess<T, DIM, M> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    #[inline(always)]
    fn _shape<'a>( &'a self ) -> &'a Shape<DIM> {
        &self.shape
    }

    #[inline(always)]
    fn memory<'a>( &'a self ) -> &'a Memory<T, M> {
        &self.memory
    }

    #[inline(always)]
    fn shape_mut<'b>( &'b mut self ) -> &'b mut Shape<DIM> {
        &mut self.shape
    }

    #[inline(always)]
    fn memory_mut<'b>( &'b mut self ) -> &'b mut Memory<T, M> {
        &mut self.memory
    }
}

impl<T, const DIM: usize, const N: usize> TensorTraits<T, DIM, Stack<N>> for Tensor<T, DIM, Stack<N>>
where
    T: Default + Clone + Copy + Debug,
    Memory<T, Stack<N>>: MemoryTraits<Type = T, New = (), Take = [T; N]>,
{
    fn new( shape: Shape<DIM> ) -> Self {
        Self {
            shape,
            memory: Memory::<T, Stack<N>>::new( () )
        }
    }

    fn take( shape: Shape<DIM>, memory: [T; N] ) -> Self {
        if shape.vol() != N { panic!( "Mismatched sizes" ); }
        Self {
            shape,
            memory: Memory::<T, Stack<N>>::take( memory )
        }
    }
}

impl<T, const DIM: usize> TensorTraits<T, DIM, Heap> for Tensor<T, DIM, Heap>
where
    T: Default + Clone + Copy + Debug,
    Memory<T, Heap>: MemoryTraits<Type = T, New = usize, Take = Vec<T>>,
{
    fn new( shape: Shape<DIM> ) -> Self {
        let vol = shape.vol();
        let mut memory = Memory::<T, Heap>::new( vol );
        memory.resize( vol, T::default() );
        Self {
            shape,
            memory
        }
    }

    fn take( shape: Shape<DIM>, memory: Vec<T> ) -> Self {
        if shape.vol() != memory.len() { panic!( "Mismatched sizes" ); }
        Self {
            shape,
            memory: Memory::<T, Heap>::take( memory )
        }
    }
}

/*
pub fn product<T, const DIM: usize, const E: usize, const F: usize, M, N, O>(
    a: &Tensor<T, DIM, M>,
    b: &Tensor<T, E, N>,
    c: &mut Tensor<T, F, O>
)
where
    T: Default + Copy + Debug + Mul<Output = T> + Add<Output = T> + AddAssign,
    M: MemoryType,
    N: MemoryType,
    O: MemoryType,
    Tensor<T, DIM, M>: TensorTraits<T, DIM, M>,
    Tensor<T, E, N>: TensorTraits<T, E, N>,
    Tensor<T, F, O>: TensorTraits<T, F, O>,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, N>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T>,
{
    let a_shape = a.shape;
    let b_shape = b.shape;
    let c_shape = c.shape;

    if a_shape[1] != b_shape[0] || c_shape[0] != a_shape[0] || c_shape[1] != b_shape[1] {
        panic!("Mismatched Shapes");
    }

    for i in 0..a_shape[0] {
        for j in 0..b_shape[1] {
            let mut sum = T::default();
            for k in 0..a_shape[1] {
                let a_index = i * a_shape[1] + k;
                let b_index = k * b_shape[1] + j;
                sum += a[a_index] * b[b_index];
            }
            let c_index = i * c_shape[1] + j;
            c[c_index] = sum;
        }
    }
}
*/

pub fn contract<T, const D: usize, const E: usize, const F: usize, M, N, O>(
    a: &Tensor<T, D, M>,
    b: &Tensor<T, E, N>,
    c: &mut Tensor<T, F, O>,
    contract_dims_a: &[usize],
    contract_dims_b: &[usize],
)
where
    T: Default + Copy + Debug + Mul<Output = T> + Add<Output = T> + AddAssign,
    M: MemoryType,
    N: MemoryType,
    O: MemoryType,
    Tensor<T, D, M>: TensorTraits<T, D, M>,
    Tensor<T, E, N>: TensorTraits<T, E, N>,
    Tensor<T, F, O>: TensorTraits<T, F, O>,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, N>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T>,
{
    if contract_dims_a.len() != contract_dims_b.len() {
        panic!("Mismatched Sizes");
    }
    for (&dim_a, &dim_b) in contract_dims_a.iter().zip(contract_dims_b.iter()) {
        if a.shape()[dim_a] != b.shape()[dim_b] {
            panic!("Mismatched Shapes");
        }
    }

    let mut index_a = [0; D];
    let mut index_b = [0; E];
    let mut index_c = [0; F];

    let non_contract_dims_a: Vec<_> = (0..D).filter(|d| !contract_dims_a.contains(d)).collect();
    let non_contract_dims_b: Vec<_> = (0..E).filter(|d| !contract_dims_b.contains(d)).collect();

    let contract_size = contract_dims_a.iter().map(|&d| a.shape()[d]).product();

    for i in 0..c.size() { // Iterate over all elements of `c`
        let mut remaining = i;

        for d in (0..F).rev() { // Convert flat index to multi-dimensional index for `c`
            index_c[d] = remaining % c.shape()[d];
            remaining /= c.shape()[d];
        }

        for (c_idx, &a_idx) in non_contract_dims_a.iter().enumerate() { // Map `c` indices to `a` and `b` non-contracted dimensions
            index_a[a_idx] = index_c[c_idx];
        }
        for (c_idx, &b_idx) in non_contract_dims_b.iter().enumerate() {
            index_b[b_idx] = index_c[non_contract_dims_a.len() + c_idx];
        }

        let mut sum = T::default();

        for j in 0..contract_size { // Sum over contracted dimensions
            let mut remaining = j;
            for (&dim_a, &dim_b) in contract_dims_a.iter().zip(contract_dims_b.iter()) {
                index_a[dim_a] = remaining % a.shape()[dim_a];
                index_b[dim_b] = remaining % b.shape()[dim_b];
                remaining /= a.shape()[dim_a];
            }
            sum += a[index_a] * b[index_b];
        }
        c[index_c] = sum;
    }
}

impl<T, const DIM: usize, M> Add for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Add<Output = T>,
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

impl<T, const DIM: usize, M> Sub for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Sub<Output = T>,
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

impl<T, const DIM: usize, M> Mul for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Mul<Output = T>,
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

impl<T, const DIM: usize, M> Div for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Div<Output = T>,
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

impl<T, const DIM: usize, M> Add<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Add<Output = T>,
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

impl<T, const DIM: usize, M> Sub<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Sub<Output = T>,
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

impl<T, const DIM: usize, M> Mul<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Mul<Output = T>,
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

impl<T, const DIM: usize, M> Div<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + Div<Output = T>,
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

impl<T, const DIM: usize, M> AddAssign for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + AddAssign,
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

impl<T, const DIM: usize, M> SubAssign for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + SubAssign,
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

impl<T, const DIM: usize, M> MulAssign for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + MulAssign,
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

impl<T, const DIM: usize, M> DivAssign for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + DivAssign,
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

impl<T, const DIM: usize, M> AddAssign<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + AddAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn add_assign( &mut self, scalar: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] += scalar;
        }
    }
}

impl<T, const DIM: usize, M> SubAssign<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + SubAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn sub_assign( &mut self, scalar: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] -= scalar;
        }
    }
}

impl<T, const DIM: usize, M> MulAssign<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + MulAssign,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn mul_assign( &mut self, scalar: T ) {
        for i in 0..self.memory.len() {
            self.memory[ i ] *= scalar;
        }
    }
}

impl<T, const DIM: usize, M> DivAssign<T> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug + DivAssign,
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
impl<T, const DIM: usize, M> Index<[usize; DIM]> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: TensorTraits<T, DIM, M>
{
    type Output = T;

    fn index( &self, index: [usize; DIM] ) -> &Self::Output {
        let flat_idx = self.flat_idx( &index );
        &self.memory[ flat_idx ]
    }
}

impl<T, const DIM: usize, M> IndexMut<[usize; DIM]> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: TensorTraits<T, DIM, M>
{
    fn index_mut( &mut self, index: [usize; DIM] ) -> &mut Self::Output {
        let flat_idx = self.flat_idx( &index );
        &mut self.memory[ flat_idx ]
    }
}

// Flat Indexing
impl<T, const DIM: usize, M> Index<usize> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.memory[ index ]
    }
}

impl<T, const DIM: usize, M> IndexMut<usize> for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.memory[ index ]
    }
}

impl<T, const DIM: usize, M> IntoIterator for Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T> + IntoIterator<Item = T, IntoIter = std::vec::IntoIter<T>>
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter( self ) -> Self::IntoIter {
        self.memory.into_iter()
    }
}

impl<'a, T, const DIM: usize, M> IntoIterator for &'a Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T> + IntoIterator<Item = T, IntoIter = std::slice::Iter<'a, T>>,
    Tensor<T, DIM, M>: TensorTraits<T, DIM, M>
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter( self ) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const DIM: usize, M> IntoIterator for &'a mut Tensor<T, DIM, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T> + IntoIterator<Item = T, IntoIter = std::slice::Iter<'a, T>>,
    Tensor<T, DIM, M>: TensorTraits<T, DIM, M>
{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter( self ) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg( test )]
mod tests {
    use super::*;

    #[test]
    fn new_stack_test() {
        let tensor = Tensor::<f32, 1, Stack<3>>::new( [3].into() );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn take_stack_test() {
        let tensor = Tensor::<f32, 1, Stack<3>>::take(
            [3].into(),
            [1.0, 2.0, 3.0]
        );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn new_heap_test() {
        let tensor = Tensor::<f32, 1, Heap>::new( [3].into() );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn take_heap_test() {
        let tensor = Tensor::<f32, 1, Heap>::take(
            [3].into(),
            [ 1.0, 2.0, 3.0 ].into()
        );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn identity_test() {
        let mut tensor = Tensor::<f32, 2, Heap>::new( [2, 2].into() );
        tensor.identity( 1.0 );
        println!( "tensor: {:?}", tensor );
        assert_eq!( tensor[ [0, 0] ], 1.0 );
        assert_eq!( tensor[ [0, 1] ], 0.0 );
        assert_eq!( tensor[ [1, 0] ], 0.0 );
        assert_eq!( tensor[ [1, 1] ], 1.0 );
    }

    #[test]
    fn iter_test() {
        let mut tensor = Tensor::<f32, 1, Heap>::new( [3].into() );
        tensor[ 0 ] = 1.0;
        tensor[ 1 ] = 2.0;
        tensor[ 2 ] = 3.0;

        let mut iter = tensor.iter();
        assert_eq!( iter.next(), Some( &1.0 ) );
        assert_eq!( iter.next(), Some( &2.0 ) );
        assert_eq!( iter.next(), Some( &3.0 ) );
        assert_eq!( iter.next(), None );
    }

    #[test]
    fn contract_test() {
        let a = Tensor::<f32, 2, Heap>::take(
            [2, 2].into(),
            [
                1.0, 2.0,
                3.0, 4.0
            ].into()
        );

        let b = Tensor::<f32, 2, Heap>::take(
            [2, 2].into(),
            [
                1.0, 2.0,
                3.0, 4.0
            ].into()
        );

        let mut c = Tensor::<f32, 2, Stack<4>>::take(
            [2, 2].into(),
            [
                0.0, 0.0,
                0.0, 0.0
            ]
        );

        println!( "Before:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        contract( &a, &b, &mut c, &[1], &[0] );

        println!( "After:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        assert_eq!( c[ [0, 0] ], 7.0 );
        assert_eq!( c[ [0, 1] ], 10.0 );
        assert_eq!( c[ [1, 0] ], 15.0 );
        assert_eq!( c[ [1, 1] ], 22.0 );
    }
}
