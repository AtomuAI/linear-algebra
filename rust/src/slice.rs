// Copyright 2024 Bewusstsein Labs

use std::{
    collections::btree_map::Values, fmt::Debug, ops::{ Index, IndexMut }
};

use memory::{ Memory, MemoryType, MemoryTraits, stack::Stack, heap::Heap };

use crate::{
    shape::Shape,
    traits::{
        ConstOrder,
        ConstShaped,
        DynShaped,
        ConstSized,
        ConstReShapeable,
        ConstReSizeable,
        ConstReOrder,
        Sliceable,
        Fillable,
        Zeroable,
        Clearable
    },
    tensor::{ Tensor, TensorTraits, Error }
};

#[derive( Debug )]
pub struct Slice<'a, T, const ORD: usize, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    tensor: &'a mut Tensor<T, ORD, M>,
    start: Shape<ORD>,
    end: Shape<ORD>,
    strides: Shape<ORD>,
}

impl<'a, T, const ORD: usize, M> Slice<'a, T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, ORD, M>: TensorTraits<T, ORD, M>
{
    pub fn new( tensor: &'a mut Tensor<T, ORD, M>, start: Shape<ORD>, end: Shape<ORD>, strides: Shape<ORD> ) -> Self {
        start.iter().zip( tensor.shape().iter() ).for_each( |( &s, &t )| assert!( s < t ) );
        end.iter().zip( tensor.shape().iter() ).for_each( |( &e, &t )| assert!( e <= t ) );
        Slice {
            tensor,
            start,
            end,
            strides,
        }
    }

    pub const fn ord( &self ) -> usize
    where
    Tensor<T, ORD, M>: ConstOrder
    {
        Tensor::<T, ORD, M>::ORD
    }

    pub fn size( &self ) -> usize {
        let mut size = 1;
        for i in 0..ORD {
            size *= ( self.end[ i ] - self.start[ i ] ) / self.strides[ i ];
        }
        size
    }

    pub fn shape( &self ) -> Shape<ORD> {
        let mut shape = Shape::<ORD>::default();
        for i in 0..ORD {
            shape[ i ] = ( self.end[ i ] - self.start[ i ] ) / self.strides[ i ];
        }
        shape
    }

    pub fn tensor( &self ) -> Result<Tensor<T, ORD, M>, Error>
    where
        Tensor::<T, ORD, M>: Default
    {
        let shape = self.shape();
        let mut tensor = Tensor::<T, ORD, M>::default();
        for i in 0..self.size() {
            let mut index = [ 0; ORD ];
            let mut stride = 1;
            for j in 0..ORD {
                index[ j ] = ( i / stride ) % shape[ j ];
                stride *= shape[ j ];
            }
            tensor[ index ] = self[ index ];
        }
        Ok( tensor )
    }

    pub fn fill( &mut self, value: T ) {
        let shape = self.shape();
        for i in 0..self.size() {
            let mut index = [ 0; ORD ];
            let mut stride = 1;
            for j in 0..ORD {
                index[ j ] = ( i / stride ) % shape[ j ];
                stride *= shape[ j ];
            }
            self[ index ] = value;
        }
    }

    pub fn zero( &mut self ) {
        self.fill( T::default() );
    }

    fn idx( &self, index: [ usize; ORD ] ) -> usize {
        let mut flat_index = 0;
        let mut stride = 1;
        for ( i, &dim_index ) in index.iter().rev().enumerate() {
            let tensor_index = self.start[ ORD - 1 - i ] + dim_index * self.strides[ ORD - 1 - i ];
            flat_index += tensor_index * stride;
            stride *= self.tensor.shape()[ ORD - 1 - i ];
        }
        flat_index
    }
}

// Dimensional Indexing
impl<'a, T, const ORD: usize, M> Index<[usize; ORD]> for Slice<'a, T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, ORD, M>: TensorTraits<T, ORD, M>
{
    type Output = T;

    fn index( &self, index: [ usize; ORD ] ) -> &Self::Output {
        let flat_index = self.idx( index );
        &self.tensor[ flat_index ]
    }
}

impl<'a, T, const ORD: usize, M> IndexMut<[usize; ORD]> for Slice<'a, T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, ORD, M>: TensorTraits<T, ORD, M>
{
    fn index_mut( &mut self, index: [ usize; ORD ] ) -> &mut Self::Output {
        let flat_index = self.idx( index );
        &mut self.tensor[ flat_index ]
    }
}

// Flat Indexing
impl<'a, T, const ORD: usize, M> Index<usize> for Slice<'a, T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.tensor[ index ]
    }
}

impl<'a, T, const ORD: usize, M> IndexMut<usize> for Slice<'a, T, ORD, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.tensor[ index ]
    }
}
