// Copyright 2024 Bewusstsein Labs

use std::fmt::Debug;

use memory::{ Memory, MemoryType, MemoryTraits };
use crate::{
    shape::Shape,
    vector::Vector,
    matrix::Matrix,
    slice::Slice
};

pub trait ConstOrder {
    const ORD: usize;
}

pub trait DynOrder {
    fn ord( &self ) -> usize;
}

pub trait ConstShaped<const ORD: usize> {
    const SHAPE: Shape<ORD>;
}

pub trait DynShaped<const ORD: usize> {
    fn shape( &self ) -> Shape<ORD>;
}

pub trait ConstSized {
    const SIZE: usize;
}

pub trait DynSized {
    fn size( &self ) -> usize;
}

pub trait ConstReShapeable<Res = Self> {
    fn reshape( self ) -> Res;
}

pub trait DynReShapeable {
    fn reshape( &mut self );
}

pub trait ConstReSizeable<Res = Self> {
    fn resize( self ) -> Res;
}

pub trait DynReSizeable {
    fn resize( &mut self );
}

pub trait ConstReOrder<Res = Self> {
    fn reorder( self ) -> Res;
}

pub trait DynReOrder {
    fn reorder( &mut self );
}

pub trait Sliceable<T, const ORD: usize, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn slice<'a>( &'a self, start: Shape<ORD>, end: Shape<ORD>, strides: Shape<ORD> ) -> Slice<'a, T, ORD, M>;
}

pub trait SliceableMut<T, const ORD: usize, M>
where
    T: 'static + Default + Copy + Debug,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn slice_mut<'a>( &'a mut self, start: Shape<ORD>, end: Shape<ORD>, strides: Shape<ORD> ) -> Slice<'a, T, ORD, M>;
}

pub trait Fillable<T> {
    fn fill( &mut self, value: T );
}

pub trait Zeroable<T> {
    fn zero( &mut self );
}

pub trait Clearable<T> {
    fn clear( &mut self );
}
