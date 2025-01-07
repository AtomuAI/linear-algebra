// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign },
    marker::PhantomData
};
use num::traits::Num;

use const_expr_bounds::{ Assert, IsTrue, IsFalse };
use memory::{ Memory, MemoryType, MemoryTraits, stack::Stack, heap::Heap };
use crate::{
    vector::Vector,
    matrix::Matrix,
    slice::Slice
};

use crate::{
    traits::{
        ConstShaped,
    },
    ops::{
        InnerProduct,
        InnerProductAssignTo,
        OuterProduct,
        OuterProductAssignTo,
        Transpose,
        TransposeAssign,
        TransposeAssignTo,
        Contract,
        ContractAssignTo,
    },
    shape::Shape,
    //slice::Slice
};

#[ derive( Debug ) ]
pub enum Error {
    MismatchedSizes,
    MismatchedShapes,
}

/// A tensor type of generic element, order and storage type.
///
#[ derive( Clone, Debug ) ]
pub struct Tensor<T, const ORD: usize, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    shape: Shape<ORD>,
    memory: Memory<T, M>
}

impl<T, const ORD: usize, M> Copy for Tensor<T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType + Clone,
    Memory<T, M>: MemoryTraits<Type = T> + Copy
{}

impl<T, const ORD: usize, M> Default for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> PartialEq for Tensor<T, ORD, M>
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
pub(crate) trait TensorAccess<T, const ORD: usize, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Sized
{
    fn _shape<'a>( &'a self ) -> &'a Shape<ORD>;
    fn memory<'a>( &'a self ) -> &'a Memory<T, M>;
    fn shape_mut<'b>( &'b mut self ) -> &'b mut Shape<ORD>;
    fn memory_mut<'b>( &'b mut self ) -> &'b mut Memory<T, M>;
}

pub trait TensorTraits<T, const ORD: usize, M>: TensorAccess<T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Sized
{
    /// Creates a new [`Tensor`] of some order with a given shape and memory.
    ///
    fn new( shape: Shape<ORD>, src: <Memory<T, M> as MemoryTraits>::New ) -> Self;

    /// Creates a new zero filled [`Tensor`].
    ///
    fn zero( shape: Shape<ORD> ) -> Self;

    /// Creates a new identity (diagonally) filled [`Tensor`].
    ///
    fn eye( shape: Shape<ORD> ) -> Self where T: Num;

    /// Get the order of the [`Tensor`].
    ///
    fn ord( &self ) -> usize {
        self._shape().ord()
    }

    /// Get the total size of the [`Tensor`].
    ///
    fn size( &self ) -> usize {
        self.memory().len()
    }

    /// Get the shape of the [`Tensor`].
    ///
    fn shape( &self ) -> &Shape<ORD> {
        self._shape()
    }

    fn slice<'a>( &'a mut self, start: Shape<ORD>, end: Shape<ORD>, strides: Shape<ORD> ) -> Slice<'a, T, ORD, M>;

    /// Reshape the [`Tensor`] to a new shape.
    ///
    fn reshape( &mut self, shape: Shape<ORD> ) -> Result<(), Error> {
        match shape.vol() == self.memory().len() {
            false => Err( Error::MismatchedSizes ),
            true => {
                *self.shape_mut() = shape;
                Ok( () )
            }
        }
    }

    /// Fill the [`Tensor`] with a given value.
    ///
    fn fill( &mut self, value: T ) {
        for i in 0..self.memory().len() {
            self.memory_mut()[ i ] = value;
        }
    }

    /// Clear the [`Tensor`].
    fn clear( &mut self ) {
        self.fill( T::default() );
    }

    /// Get the flat index of a multi-dimensional index.
    ///
    fn idx( &self, index: &[usize; ORD] ) -> usize {
        let dim = self.shape().ord();
        let mut idx = 0;
        let mut stride = 1;
        for ( i, &dim_index ) in index.iter().rev().enumerate() {
            idx += dim_index * stride;
            stride *= self.shape()[ dim - 1 - i ];
        }
        idx
    }

    /// Returns an iterator over the [`Tensor`].
    ///
    fn iter<'a>( &'a self ) -> std::slice::Iter<'a, T>
    where
        M: 'a
    {
        self.memory().iter()
    }

    /// Returns a mutable iterator over the [`Tensor`].
    ///
    fn iter_mut<'a>( &'a mut self ) -> std::slice::IterMut<'a, T>
    where
        M: 'a
    {
        self.memory_mut().iter_mut()
    }
}

#[allow(clippy::needless_lifetimes)]
impl<T, const ORD: usize, M> TensorAccess<T, ORD, M> for Tensor<T, ORD, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    #[inline(always)]
    fn _shape<'a>( &'a self ) -> &'a Shape<ORD> {
        &self.shape
    }

    #[inline(always)]
    fn memory<'a>( &'a self ) -> &'a Memory<T, M> {
        &self.memory
    }

    #[inline(always)]
    fn shape_mut<'b>( &'b mut self ) -> &'b mut Shape<ORD> {
        &mut self.shape
    }

    #[inline(always)]
    fn memory_mut<'b>( &'b mut self ) -> &'b mut Memory<T, M> {
        &mut self.memory
    }
}

impl<T, const ORD: usize, const N: usize> TensorTraits<T, ORD, Stack<N>> for Tensor<T, ORD, Stack<N>>
where
    T: Default + Clone + Copy + Debug,
    Memory<T, Stack<N>>: MemoryTraits<Type = T, New = [T; N]>,
{
    fn new( shape: Shape<ORD>, memory: [T; N] ) -> Self {
        if shape.vol() != N { panic!( "Mismatched sizes" ); }
        Self {
            shape,
            memory: Memory::<T, Stack<N>>::new( memory )
        }
    }

    fn zero( shape: Shape<ORD> ) -> Self {
        Self::new( shape, [T::default(); N] )
    }

    fn eye( shape: Shape<ORD> ) -> Self
    where
        T: Num
    {
        let mut memory = [ T::default(); N ];
        let mut stride = 0;
        for ( i, elem ) in memory.iter_mut().enumerate() {
            if i == stride {
                *elem = T::one();
                stride += shape[ 0 ] + 1;
            }
        }
        Self::new( shape, memory )
    }

    fn slice<'a>( &'a mut self, start: Shape<ORD>, end: Shape<ORD>, strides: Shape<ORD> ) -> Slice<'a, T, ORD, Stack<N>> {
        Slice::new( self, start, end, strides )
    }
}

impl<T, const ORD: usize> TensorTraits<T, ORD, Heap> for Tensor<T, ORD, Heap>
where
    T: Default + Clone + Copy + Debug,
    Memory<T, Heap>: MemoryTraits<Type = T, New = Vec<T>>,
{
    fn new( shape: Shape<ORD>, memory: Vec<T> ) -> Self {
        if shape.vol() != memory.len() { panic!( "Mismatched sizes" ); }
        Self {
            shape,
            memory: Memory::<T, Heap>::new( memory )
        }
    }

    fn zero( shape: Shape<ORD> ) -> Self {
        Self::new( shape, vec![ T::default(); shape.vol() ] )
    }

    fn eye( shape: Shape<ORD> ) -> Self
    where
        T: Num
    {
        let mut memory = vec![ T::default(); shape.vol() ];
        let mut stride = 0;
        for ( i, elem ) in memory.iter_mut().enumerate() {
            if i == stride {
                *elem = T::one();
                stride += shape[ 0 ] + 1;
            }
        }
        Self::new( shape, memory )
    }

    fn slice<'a>( &'a mut self, start: Shape<ORD>, end: Shape<ORD>, strides: Shape<ORD> ) -> Slice<'a, T, ORD, Heap> {
        Slice::new( self, start, end, strides )
    }
}

pub const fn contracted_dim( lhs_dim: usize, rhs_dim: usize, ctr_dim: usize ) -> usize {
    ( lhs_dim + rhs_dim ) - ( 2 * ctr_dim )
}

impl<T, M, N, O, const LHS_DIM: usize, const RHS_DIM: usize, const CTR_DIM: usize> Contract<CTR_DIM, &Tensor<T, RHS_DIM, N>, Tensor<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O>> for &Tensor<T, LHS_DIM, M>
where
    T: Default + Copy + Debug + Mul<Output = T> + Add<Output = T> + AddAssign,
    M: MemoryType,
    N: MemoryType,
    O: MemoryType,
    Tensor<T, LHS_DIM, M>: TensorTraits<T, LHS_DIM, M>,
    Tensor<T, RHS_DIM, N>: TensorTraits<T, RHS_DIM, N>,
    Tensor<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O>: TensorTraits<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O>,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, N>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T> + Default,
    M::Data<T>: Default,
    Assert<{ CTR_DIM <= LHS_DIM }>: IsTrue,
    Assert<{ CTR_DIM <= RHS_DIM }>: IsTrue,
    [(); contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) ]:
{
    fn contract( self, lhs_contract_dims: [ usize; CTR_DIM ], rhs_contract_dims: [ usize; CTR_DIM ], rhs: &Tensor<T, RHS_DIM, N>, ) -> Tensor<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O> {
        let mut res = Tensor::<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O>::default();

        for ( &lhs_dim, &rhs_dim ) in lhs_contract_dims.iter().zip( rhs_contract_dims.iter() ) {
            if self.shape()[ lhs_dim ] != rhs.shape()[ rhs_dim ] {
                panic!( "Mismatched Shapes" );
            }
        }

        let mut lhs_index = [ 0; LHS_DIM ];
        let mut rhs_index = [ 0; RHS_DIM ];
        let mut res_index = [ 0; contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) ];

        let lhs_non_contract_dims: Vec<usize> = ( 0..LHS_DIM ).filter( |dim| !lhs_contract_dims.contains( dim ) ).collect();
        let rhs_non_contract_dims: Vec<usize> = ( 0..RHS_DIM ).filter( |dim| !rhs_contract_dims.contains( dim ) ).collect();

        let contract_size = lhs_contract_dims.iter().map( |&dim| self.shape()[ dim ] ).product();

        for i in 0..res.size() {
            let mut remaining = i;

            for dim in ( 0..contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) ).rev() { // Convert flat index to multi-dimensional index for `c`
                res_index[ dim ] = remaining % res.shape()[ dim ];
                remaining /= res.shape()[ dim ];
            }

            for ( res_idx, &lhs_idx ) in lhs_non_contract_dims.iter().enumerate() { // Map `c` indices to `a` and `b` non-contracted dimensions
                lhs_index[ lhs_idx ] = res_index[ res_idx ];
            }
            for ( res_idx, &rhs_idx ) in rhs_non_contract_dims.iter().enumerate() {
                rhs_index[ rhs_idx ] = res_index[ lhs_non_contract_dims.len() + res_idx ];
            }

            let mut sum = T::default();

            for j in 0..contract_size { // Sum over contracted dimensions
                let mut remaining = j;
                for ( &lhs_dim, &rhs_dim ) in lhs_contract_dims.iter().zip( rhs_contract_dims.iter()) {
                    lhs_index[ lhs_dim ] = remaining % self.shape()[ lhs_dim ];
                    rhs_index[ rhs_dim ] = remaining % rhs.shape()[ rhs_dim ];
                    remaining /= self.shape()[ lhs_dim ];
                }
                sum += self[ lhs_index ] * rhs[ rhs_index ];
            }
            res[ res_index ] = sum;
        }

        res
    }
}

impl<T, M, N, O, const LHS_DIM: usize, const RHS_DIM: usize, const CTR_DIM: usize> ContractAssignTo<CTR_DIM, &Tensor<T, RHS_DIM, N>, Tensor<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O>> for &Tensor<T, LHS_DIM, M>
where
    T: Default + Copy + Debug + Mul<Output = T> + Add<Output = T> + AddAssign,
    M: MemoryType,
    N: MemoryType,
    O: MemoryType,
    Tensor<T, LHS_DIM, M>: TensorTraits<T, LHS_DIM, M>,
    Tensor<T, RHS_DIM, N>: TensorTraits<T, RHS_DIM, N>,
    Tensor<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O>: TensorTraits<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O>,
    Memory<T, M>: MemoryTraits<Type = T>,
    Memory<T, N>: MemoryTraits<Type = T>,
    Memory<T, O>: MemoryTraits<Type = T> + Default,
    M::Data<T>: Default,
    Assert<{ CTR_DIM <= LHS_DIM }>: IsTrue,
    Assert<{ CTR_DIM <= RHS_DIM }>: IsTrue,
    [(); contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) ]:
{
    fn contract_assign_to( self, lhs_contract_dims: [ usize; CTR_DIM ], rhs_contract_dims: [ usize; CTR_DIM ], rhs: &Tensor<T, RHS_DIM, N>, res: &mut Tensor<T, { contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) }, O> ) {
        for ( &lhs_dim, &rhs_dim ) in lhs_contract_dims.iter().zip( rhs_contract_dims.iter() ) {
            if self.shape()[ lhs_dim ] != rhs.shape()[ rhs_dim ] {
                panic!( "Mismatched Shapes" );
            }
        }

        let mut lhs_index = [ 0; LHS_DIM ];
        let mut rhs_index = [ 0; RHS_DIM ];
        let mut res_index = [ 0; contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) ];

        let lhs_non_contract_dims: Vec<usize> = ( 0..LHS_DIM ).filter( |dim| !lhs_contract_dims.contains( dim ) ).collect();
        let rhs_non_contract_dims: Vec<usize> = ( 0..RHS_DIM ).filter( |dim| !rhs_contract_dims.contains( dim ) ).collect();

        let contract_size = lhs_contract_dims.iter().map( |&dim| self.shape()[ dim ] ).product();

        for i in 0..res.size() {
            let mut remaining = i;

            for dim in ( 0..contracted_dim( LHS_DIM, RHS_DIM, CTR_DIM ) ).rev() { // Convert flat index to multi-dimensional index for `c`
                res_index[ dim ] = remaining % res.shape()[ dim ];
                remaining /= res.shape()[ dim ];
            }

            for ( res_idx, &lhs_idx ) in lhs_non_contract_dims.iter().enumerate() { // Map `c` indices to `a` and `b` non-contracted dimensions
                lhs_index[ lhs_idx ] = res_index[ res_idx ];
            }
            for ( res_idx, &rhs_idx ) in rhs_non_contract_dims.iter().enumerate() {
                rhs_index[ rhs_idx ] = res_index[ lhs_non_contract_dims.len() + res_idx ];
            }

            let mut sum = T::default();

            for j in 0..contract_size { // Sum over contracted dimensions
                let mut remaining = j;
                for ( &lhs_dim, &rhs_dim ) in lhs_contract_dims.iter().zip( rhs_contract_dims.iter()) {
                    lhs_index[ lhs_dim ] = remaining % self.shape()[ lhs_dim ];
                    rhs_index[ rhs_dim ] = remaining % rhs.shape()[ rhs_dim ];
                    remaining /= self.shape()[ lhs_dim ];
                }
                sum += self[ lhs_index ] * rhs[ rhs_index ];
            }
            res[ res_index ] = sum;
        }
    }
}

impl<T, const COL: usize> From<Vector<T, COL>> for Tensor<T, 1, Stack<COL>>
where
    T: Default + Copy + Debug,
    Memory<T, Stack<COL>>: MemoryTraits<Type = T, New = [T; COL]>,
    Self: Clone + TensorTraits<T, 1, Stack<COL>>
{
    fn from( src: Vector<T, COL> ) -> Self {
        Self::new(
            src.shape(),
            *src
        )
    }
}

impl<T, const COL: usize, const ROW: usize> From<Matrix<T, COL, ROW>> for Tensor<T, 2, Stack<{COL * ROW}>>
where
    T: Default + Copy + Debug,
    Memory<T, Stack<{COL * ROW}>>: MemoryTraits<Type = T, New = [T; COL * ROW]>,
    Matrix<T, COL, ROW>: ConstShaped<2>,
    Self: Clone + TensorTraits<T, 2, Stack<{COL * ROW}>>,
    [(); COL * ROW]:
{
    fn from( src: Matrix<T, COL, ROW> ) -> Self {
        Self::new(
            src.shape(),
            *src
        )
    }
}

// TODO: Should be TryFrom
impl<T, const COL: usize> From<Vector<T, COL>> for Tensor<T, 1, Heap>
where
    T: Default + Copy + Debug,
    Memory<T, Stack<COL>>: MemoryTraits<Type = T, New = [T; COL]>,
    Vector<T, COL>: ConstShaped<1>,
    Self: Clone + TensorTraits<T, 1, Stack<COL>>
{
    fn from( src: Vector<T, COL> ) -> Self {
        if src.shape()[ 0 ] != COL { panic!( "Mismatched column length" ); }
        let mut this = Self::default();
        this.iter_mut().zip( src.iter() )
            .for_each( |( a, &b )| *a = b );
        this
    }
}

// TODO: Should be TryFrom
impl<T, const COL: usize, const ROW: usize> From<Matrix<T, COL, ROW>> for Tensor<T, 2, Heap>
where
    T: Default + Copy + Debug,
    Memory<T, Stack<{COL * ROW}>>: MemoryTraits<Type = T, New = [T; COL * ROW]>,
    Matrix<T, COL, ROW>: ConstShaped<2>,
    Self: Clone + TensorTraits<T, 2, Stack<{COL * ROW}>>,
    [(); COL * ROW]:
{
    fn from( src: Matrix<T, COL, ROW> ) -> Self {
        if src.shape()[ 0 ] != COL { panic!( "Mismatched column length" ); }
        if src.shape()[ 1 ] != ROW { panic!( "Mismatched row length" ); }
        let mut this = Self::default();
        this.iter_mut().zip( src.iter() )
            .for_each( |( a, &b )| *a = b );
        this
    }
}

// TODO: Should be TryFrom
impl<T, const ORD: usize, const CAP: usize> From<Tensor<T, ORD, Stack<CAP>>> for Tensor<T, ORD, Heap>
where
    T: Default + Copy + Debug,
    Memory<T, Stack<CAP>>: MemoryTraits<Type = T, New = [T; CAP]>,
    Self: Default + Clone + TensorTraits<T, 2, Heap>,
{
    fn from( src: Tensor<T, ORD, Stack<CAP>> ) -> Self {
        if src.shape().vol() != CAP { panic!("Mismatched size"); }
        let mut this = Self::default();
        this.iter_mut().zip( src.iter() )
            .for_each( |( a, &b )| *a = b );
        this
    }
}

// TODO: Should be TryFrom
impl<T, const ORD: usize, const CAP: usize> From<Tensor<T, ORD, Heap>> for Tensor<T, ORD, Stack<CAP>>
where
    T: Default + Copy + Debug,
    Memory<T, Stack<CAP>>: MemoryTraits<Type = T, New = [T; CAP]>,
    Self: Default + Clone + TensorTraits<T, 2, Stack<CAP>>,
{
    fn from( src: Tensor<T, ORD, Heap> ) -> Self {
        if src.shape().vol() != CAP { panic!("Mismatched size"); }
        let mut this = Self::default();
        this.iter_mut().zip( src.iter() )
            .for_each( |( a, &b )| *a = b );
        this
    }
}

/*
impl<T, const ORD: usize, M> From<Tensor<T, { ORD - 1 }, M>> for Tensor<T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: Default + Clone + Sized + TensorTraits<T, ORD, M>,
    Tensor<T, { ORD - 1 }, M>: Default + Clone + Sized + TensorTraits<T, { ORD - 1 }, M>,
    [(); ORD - 1]:
{
    fn from( src: Tensor<T, { ORD - 1 }, M> ) -> Self {
        Self::new(
            src.shape(),
            *src.memory()
        )
    }
}
    */

impl<T, const ORD: usize, M> Add for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> Sub for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> Mul for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> Div for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> Add<T> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> Sub<T> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> Mul<T> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> Div<T> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> AddAssign for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> SubAssign for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> MulAssign for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> DivAssign for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> AddAssign<T> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> SubAssign<T> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> MulAssign<T> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> DivAssign<T> for Tensor<T, ORD, M>
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
impl<T, const ORD: usize, M> Index<[usize; ORD]> for Tensor<T, ORD, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: TensorTraits<T, ORD, M>
{
    type Output = T;

    fn index( &self, index: [usize; ORD] ) -> &Self::Output {
        let idx = self.idx( &index );
        &self.memory[ idx ]
    }
}

impl<T, const ORD: usize, M> IndexMut<[usize; ORD]> for Tensor<T, ORD, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Self: TensorTraits<T, ORD, M>
{
    fn index_mut( &mut self, index: [usize; ORD] ) -> &mut Self::Output {
        let idx = self.idx( &index );
        &mut self.memory[ idx ]
    }
}

// Flat Indexing
impl<T, const ORD: usize, M> Index<usize> for Tensor<T, ORD, M>
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

impl<T, const ORD: usize, M> IndexMut<usize> for Tensor<T, ORD, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.memory[ index ]
    }
}

impl<T, const ORD: usize, M> IntoIterator for Tensor<T, ORD, M>
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

impl<'a, T, const ORD: usize, M> IntoIterator for &'a Tensor<T, ORD, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T> + IntoIterator<Item = T, IntoIter = std::slice::Iter<'a, T>>,
    Tensor<T, ORD, M>: TensorTraits<T, ORD, M>
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter( self ) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const ORD: usize, M> IntoIterator for &'a mut Tensor<T, ORD, M>
where
    T: Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T> + IntoIterator<Item = T, IntoIter = std::slice::Iter<'a, T>>,
    Tensor<T, ORD, M>: TensorTraits<T, ORD, M>
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
        let tensor = Tensor::<f32, 1, Stack<3>>::new(
            [3].into(),
            [0.0, 0.0, 0.0]
        );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn take_stack_test() {
        let tensor = Tensor::<f32, 1, Stack<3>>::new(
            [3].into(),
            [1.0, 2.0, 3.0]
        );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn new_heap_test() {
        let tensor = Tensor::<f32, 1, Heap>::new(
            [3].into(),
            [ 0.0, 0.0, 0.0 ].into()
        );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn take_heap_test() {
        let tensor = Tensor::<f32, 1, Heap>::new(
            [3].into(),
            [ 1.0, 2.0, 3.0 ].into()
        );
        assert_eq!( tensor.memory.len(), 3 );
    }

    #[test]
    fn identity_test() {
        let tensor = Tensor::<f32, 2, Heap>::eye(
            [2, 2].into()
        );
        println!( "tensor: {:?}", tensor );
        assert_eq!( tensor[ [0, 0] ], 1.0 );
        assert_eq!( tensor[ [0, 1] ], 0.0 );
        assert_eq!( tensor[ [1, 0] ], 0.0 );
        assert_eq!( tensor[ [1, 1] ], 1.0 );
    }

    #[test]
    fn iter_test() {
        let mut tensor = Tensor::<f32, 1, Heap>::new(
            [3].into(),
            [ 0.0, 0.0, 0.0 ].into()
        );
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
        use crate::ops::ContractAssignTo;
        let a = Tensor::<f32, 3, Heap>::new(
            [2, 2, 2].into(),
            [
                1.0, 2.0,
                3.0, 4.0,

                1.0, 2.0,
                3.0, 4.0,
            ].into()
        );

        let b = Tensor::<f32, 3, Heap>::new(
            [2, 2, 2].into(),
            [
                1.0, 2.0,
                3.0, 4.0,

                1.0, 2.0,
                3.0, 4.0,
            ].into()
        );

        let mut c = Tensor::<f32, 4, Stack<{2*2*2*2}>>::new(
            [2, 2, 2, 2].into(),
            [ 0.0; 2*2*2*2 ]
        );

        println!( "Before:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        (&a).contract_assign_to( [ 1 ], [ 0 ], &b, &mut c );

        println!( "After:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        //assert_eq!( c[ [0, 0] ], 7.0 );
        //assert_eq!( c[ [0, 1] ], 10.0 );
        //assert_eq!( c[ [1, 0] ], 15.0 );
        //assert_eq!( c[ [1, 1] ], 22.0 );
    }

    #[test]
    fn slice_test() {
        let mut tensor = Tensor::<f32, 2, Heap>::new(
            [2, 2].into(),
            [
                1.0, 2.0,
                3.0, 4.0,
            ].into()
        );

        let slice = tensor.slice(
            [0, 0].into(),
            [2, 2].into(),
            [1, 1].into()
        );

        for i in 0..slice.size() {
            println!( "slice[{}]: {}", i, slice[ i ] );
        }

        for ( i, elem ) in tensor.iter().enumerate() {
            println!( "tensor[{}]: {}", i, elem );
        }
    }
}
