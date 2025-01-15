// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut, Index, IndexMut, Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign }
};
use num::traits::Num;
use const_expr_bounds::{ Assert, IsTrue };
use memory::{ stack::Stack, heap::Heap };

use crate::{
    traits::{
        Columns,
        Rows,
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
    ops::{
        Determinant,
        MatrixMul,
        MatrixMulAssignTo,
        InnerProduct,
        InnerProductAssignTo,
        OuterProduct,
        OuterProductAssignTo,
        Transpose,
        TransposeAssign,
        TransposeAssignTo,
        KroneckerProduct,
        KroneckerProductAssignTo
    },
    vector::Vector,
    shape::Shape,
    tensor::{ Tensor, TensorAccess, TensorTraits }
};

/// A matrix type of generic element and size.
///
#[derive( Clone, Copy )]
pub struct Matrix<T, const COL: usize, const ROW: usize>( [ T; COL * ROW ] ) where T: 'static + Copy + Default + Debug, [ (); COL * ROW ]:;

impl<T, const COL: usize, const ROW: usize> Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    /// Creates a new const [`Matrix`].
    ///
    pub const fn new_const( src: [T; COL * ROW] ) -> Self {
        Self ( src )
    }

    /// Creates a new [`Matrix`].
    ///
    pub fn new( src: [T; COL * ROW] ) -> Self {
        Self ( src )
    }

    /// Creates a new zero filled [`Matrix`].
    ///
    pub fn zero() -> Self
    where
        T: Num
    {
        Self ( [T::zero(); COL * ROW] )
    }

    /// Creates a new identity (diagonally) filled [`Matrix`].
    ///
    pub fn eye() -> Self
    where
        T: Num
    {
        Self ( {
            let mut data = [ T::zero(); COL * ROW ];
            ( 0..COL.min( ROW ) ).for_each( |i| data[ Self::idx( i, i ) ] = T::one() );
            data
        })
    }

    /// Returns the order of the [`Matrix`].
    ///
    /// The order of a [`Matrix`] is always 2.
    ///
    #[inline(always)]
    pub const fn ord( &self ) -> usize
    where
        Self: ConstOrder
    {
        Self::ORD
    }

    /// Returns the total number of elements in the [`Matrix`].
    ///
    /// The total number of elements in a [`Matrix`] is always `COL * ROW`.
    ///
    #[inline(always)]
    pub const fn area() -> usize {
        COL * ROW
    }

    /// Returns the shape of the [`Matrix`].
    ///
    /// The shape of a [`Matrix`] is always `[COL, ROW]`.
    ///
    #[inline(always)]
    pub const fn shape( &self ) -> Shape<2>
    where
        Self: ConstShaped<2>
    {
        Self::SHAPE
    }

    /// Returns the index of the [`Matrix`] at the given column and row.
    ///
    #[inline(always)]
    pub fn idx( col: usize, row: usize ) -> usize {
        ( row * COL ) + col
    }

    /// Returns the index of the [`Matrix`] at the given column and row.
    ///
    #[inline(always)]
    pub const fn idx_const( col: usize, row: usize ) -> usize {
        ( row * COL ) + col
    }

    /// Returns the column and row of the [`Matrix`] at the given index.
    ///
    #[inline(always)]
    pub fn coord( idx: usize ) -> [usize; 2] {
        [ idx % COL, idx / COL ]
    }

    /// Returns the column and row of the [`Matrix`] at the given index.
    ///
    #[inline(always)]
    pub const fn coord_const( idx: usize ) -> [usize; 2] {
        [ idx % COL, idx / COL ]
    }

    /// Returns an iterator over the elements of the [`Matrix`].
    ///
    /// The iterator yields references to the elements of the [`Matrix`] in order.
    ///
    pub fn iter_row_wise( &self ) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Returns an iterator over mutable references to the elements of the [`Matrix`].
    ///
    /// The iterator yields mutable references to the elements of the [`Matrix`] in order.
    ///
    pub fn iter_row_wise_mut( &mut self ) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }

    /// Returns an iterator over the elements of the [`Matrix`].
    ///
    /// The iterator yields references to the elements of the [`Matrix`] in order.
    ///
    pub fn iter_col_wise( &self ) -> impl Iterator<Item = &T> {
        ( 0..COL ).flat_map( move |col| {
            ( 0..ROW ).map( move |row| {
                &self[[ col, row ]]
            })
        })
    }

    /// Returns an iterator over mutable references to the elements of the [`Matrix`].
    ///
    /// The iterator yields mutable references to the elements of the [`Matrix`] in order.
    ///
    pub fn iter_col_wise_mut( &mut self ) -> impl Iterator<Item = &mut T> {
        let self_ptr = self as *mut Self;
        ( 0..COL ).flat_map( move |col| {
            ( 0..ROW ).map( move |row| unsafe {
                &mut (*self_ptr)[[ col, row ]]
            })
        })
    }

    /// Returns an iterator over the elements of the [`Matrix`] by row.
    ///
    /// The iterator yields references to the elements of the [`Matrix`] by row.
    ///
    pub fn iter_row( &self, row: usize ) -> impl Iterator<Item = &T> {
        self.0[ Self::idx( row, 0_usize )..Self::idx( row, COL ) ].iter()
    }

    /// Returns an iterator over mutable references to the elements of the [`Matrix`] by row.
    ///
    /// The iterator yields mutable references to the elements of the [`Matrix`] by row.
    ///
    pub fn iter_row_mut( &mut self, row: usize ) -> impl Iterator<Item = &mut T> {
        self.0[ Self::idx( row, 0_usize )..Self::idx( row, COL ) ].iter_mut()
    }

    /// Returns an iterator over the elements of the [`Matrix`] by column.
    ///
    /// The iterator yields references to the elements of the [`Matrix`] by column.
    ///
    pub fn iter_col( &self, col: usize ) -> impl Iterator<Item = &T> {
        ( 0..ROW ).map( move |row| &self[[ col, row ]] )
    }

    /// Returns an iterator over mutable references to the elements of the [`Matrix`] by column.
    ///
    /// The iterator yields mutable references to the elements of the [`Matrix`] by column.
    ///
    pub fn iter_col_mut( &mut self, col: usize ) -> impl Iterator<Item = &mut T> {
        ( 0..ROW ).map( move |row| unsafe { &mut *( &mut self[[ col, row ]] as *mut _ ) } ) // SAFETY: We ensure that we only yield each element once, so no aliasing occurs.
    }

    /// Returns an iterator over the rows of the [`Matrix`].
    ///
    /// The iterator yields slices of the rows of the [`Matrix`].
    ///
    pub fn iter_rows( &self ) -> impl Iterator<Item = &[T]> {
        self.0.chunks( COL )
    }

    /// Returns an iterator over mutable references to the rows of the [`Matrix`].
    ///
    /// The iterator yields mutable slices of the rows of the [`Matrix`].
    ///
    pub fn iter_rows_mut( &mut self ) -> impl Iterator<Item = &mut [T]> {
        self.0.chunks_mut( COL )
    }

    /// Returns an iterator over the columns of the [`Matrix`].
    ///
    /// The iterator yields slices of the columns of the [`Matrix`].
    ///
    pub fn iter_cols( &self ) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        ( 0..COL ).map( move |col| self.iter_col( col ) )
    }

    /// Returns an iterator over mutable references to the columns of the [`Matrix`].
    ///
    /// The iterator yields mutable slices of the columns of the [`Matrix`].
    ///
    pub fn iter_cols_mut( &mut self ) -> impl Iterator<Item = impl Iterator<Item = &mut T>> {
        ( 0..COL ).map( move |col| {
            let matrix_ptr = self.0.as_mut_ptr();
            ( 0..ROW ).map( move |row| unsafe { &mut *matrix_ptr.add( Self::idx( col, row ) ) } ) // SAFETY: We ensure that we only yield each element once, so no aliasing occurs.
        })
    }
}

impl<T, const COL: usize, const ROW: usize> Columns for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    const COLS: usize = COL;
}

impl<T, const COL: usize, const ROW: usize> Rows for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    const ROWS: usize = ROW;
}

impl<T, const COL: usize, const ROW: usize> ConstOrder for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    const ORD: usize = 1;
}

impl<T, const COL: usize, const ROW: usize> ConstShaped<1> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    const SHAPE: Shape<1> = Shape::new_const([ COL ]);
}

impl<T, const COL: usize, const ROW: usize> ConstSized for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    const SIZE: usize = COL;
}

impl<T, const OLD_COL: usize, const OLD_ROW: usize,const NEW_COL: usize, const NEW_ROW: usize> ConstReShapeable<Matrix<T, NEW_COL, NEW_ROW>> for Matrix<T, OLD_COL, OLD_ROW>
where
    T: 'static + Copy + Default + Debug,
    Assert<{ NEW_COL * NEW_ROW == OLD_COL * OLD_ROW }>: IsTrue,
    [ (); OLD_COL * OLD_ROW ]:,
    [ (); NEW_COL * NEW_ROW ]:
{
    fn reshape( self ) -> Matrix<T, NEW_COL, NEW_ROW> {
        unsafe { *( &self as *const Matrix<T, OLD_COL, OLD_ROW> as *const Matrix<T, NEW_COL, NEW_ROW> ) } // SAFETY: This is safe because we have asserted that the total number of elements is the same
    }
}

impl<T, const OLD_COL: usize, const OLD_ROW: usize,const NEW_COL: usize, const NEW_ROW: usize> ConstReSizeable<Matrix<T, NEW_COL, NEW_ROW>> for Matrix<T, OLD_COL, OLD_ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); OLD_COL * OLD_ROW ]:,
    [ (); NEW_COL * NEW_ROW ]:
{
    fn resize( self ) -> Matrix<T, NEW_COL, NEW_ROW> {
        let mut res = Matrix::<T, NEW_COL, NEW_ROW>( [ T::default(); NEW_COL * NEW_ROW ] );
        let min_rows = OLD_ROW.min( NEW_ROW );
        let min_cols = OLD_COL.min( NEW_COL );
        for row in 0..min_rows {
            let old_start = row * OLD_COL;
            let old_end = old_start + min_cols;
            let new_start = row * NEW_COL;
            let new_end = new_start + min_cols;
            res.0[ new_start..new_end ].copy_from_slice( &self.0[ old_start..old_end ] );
        }
        res
    }
}

impl<T, const COL: usize, const ROW: usize> ConstReOrder<Vector<T, { COL * ROW }>> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn reorder( self ) -> Vector<T, { COL * ROW }> {
        unsafe { *( &self as *const Matrix<T, COL, ROW> as *const Vector<T, { COL * ROW }> ) } // SAFETY: This is safe because we have asserted that the total number of elements is the same
    }
}

impl<T, const COL: usize, const ROW: usize> Fillable<T> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn fill( &mut self, value: T ) {
        self.0 = [ value; COL * ROW ];
    }
}

impl<T, const COL: usize, const ROW: usize> Zeroable<T> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug + Num,
    [ (); COL * ROW ]:
{
    fn zero( &mut self ) {
        self.0 = [ T::zero(); COL * ROW ];
    }
}

impl<T, const COL: usize, const ROW: usize> Clearable<T> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug + Num,
    [ (); COL * ROW ]:
{
    fn clear( &mut self ) {
        self.0 = [ T::default(); COL * ROW ];
    }
}

impl<T, const COL: usize, const ROW: usize> std::fmt::Display for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn fmt( &self, f: &mut std::fmt::Formatter<'_> ) -> std::fmt::Result {
        for row in self.iter_rows() {
            for item in row {
                write!( f, "{:?} ", item )?;
            }
            writeln!( f )?;
        }
        Ok(())
    }
}

impl<T, const COL: usize, const ROW: usize> Debug for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn fmt( &self, f: &mut std::fmt::Formatter<'_> ) -> std::fmt::Result {
        writeln!( f, "Matrix{{" )?;
        for row in self.iter_rows() {
            write!( f, "  [" )?;
            for item in row {
                write!( f, "{:?}, ", item )?;
            }
            writeln!( f, "]" )?;
        }
        write!( f, "}}" )
    }
}

impl<T, const COL: usize, const ROW: usize> Deref for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    type Target = [ T; COL * ROW ];

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T, const COL: usize, const ROW: usize> DerefMut for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const COL: usize, const ROW: usize> Index<usize> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.0[ index ]
    }
}

impl<T, const COL: usize, const ROW: usize> IndexMut<usize> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.0[ index ]
    }
}

impl<T, const COL: usize, const ROW: usize> Index<[usize; 2]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    type Output = T;

    fn index( &self, index: [usize; 2] ) -> &Self::Output {
        &self.0[ Self::idx( index[ 0 ], index[ 1 ] ) ]

    }
}

impl<T, const COL: usize, const ROW: usize> IndexMut<[usize; 2]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn index_mut( &mut self, index: [usize; 2] ) -> &mut Self::Output {
        &mut self.0[ Self::idx( index[ 0 ], index[ 1 ] ) ]
    }
}

impl<T, const COL: usize, const ROW: usize> Default for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:
{
    fn default() -> Self {
        Self ( [ T::default(); COL * ROW ] )
    }
}

impl<T, const COL: usize, const ROW: usize> From<[T; COL * ROW]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: [T; COL * ROW] ) -> Self {
        Self ( src )
    }
}

impl<T, const COL: usize, const ROW: usize> From<[[T; COL]; ROW]> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    [ (); COL * ROW ]:,
{
    fn from(src: [[T; COL]; ROW]) -> Self {
        let mut data = [T::default(); COL * ROW];

        src.iter().enumerate().for_each(
            |( i, row )| { row.iter().enumerate().for_each(
                |( j, &item )| data[ Self::idx( i, j ) ] = item
            )}
        );

        Self (data)
    }
}

#[allow(clippy::identity_op)]
impl<T, const COL: usize, const ROW: usize> From<Vector<T, {COL * ROW}>> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    Vector<T, {COL * ROW}>: ConstReOrder<Matrix<T, COL, ROW>>,
    [(); COL * ROW]:
{
    fn from( src: Vector<T, {COL * ROW}> ) -> Self {
        src.reorder()
    }
}

impl<T, const COL: usize, const ROW: usize> From<Tensor<T, 2, Stack<{COL * ROW}>>> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: Tensor<T, 2, Stack<{COL * ROW}>> ) -> Self {
        Self ( ***src.memory() )
    }
}

impl<T, const COL: usize, const ROW: usize> From<Tensor<T, 2, Heap>> for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug,
    Tensor<T, 2, Heap>: DynShaped<2>,
    [ (); COL * ROW ]:
{
    fn from( src: Tensor<T, 2, Heap> ) -> Self {
        if src.shape()[0] != COL { panic!( "Mismatched column length" ); }
        if src.shape()[1] != ROW { panic!( "Mismatched row length" ); }
        let mut this = Self::default();
        this.iter_mut().zip( src.iter() )
            .for_each( |( a, &b )| *a = b );
        this
    }
}

impl<T, const COL: usize, const ROW: usize> PartialEq for Matrix<T, COL, ROW>
where
    T: 'static + Copy + Default + Debug + PartialEq,
    [ (); COL * ROW ]:
{
    fn eq( &self, other: &Self ) -> bool {
        self.0 == other.0
    }
}


impl<T, const COL: usize, const ROW: usize> Add for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn add( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Sub for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn sub( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Mul for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn mul( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Div for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn div( self, other: Self ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Add<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn add( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Sub<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn sub( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Mul<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn mul( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> Div<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone,
    [ (); COL * ROW ]:
{
    type Output = Self;

    fn div( self, scalar: T ) -> Self::Output {
        let mut result = Self::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
        result
    }
}

impl<T, const COL: usize, const ROW: usize> AddAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + AddAssign,
    [ (); COL * ROW ]:
{
    fn add_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a += b );
    }
}

impl<T, const COL: usize, const ROW: usize> SubAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + SubAssign,
    [ (); COL * ROW ]:
{
    fn sub_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a -= b );
    }
}

impl<T, const COL: usize, const ROW: usize> MulAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + MulAssign,
    [ (); COL * ROW ]:
{
    fn mul_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a *= b );
    }
}

impl<T, const COL: usize, const ROW: usize> DivAssign for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + DivAssign,
    [ (); COL * ROW ]:
{
    fn div_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a /= b );
    }
}

impl<T, const COL: usize, const ROW: usize> AddAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + AddAssign,
    [ (); COL * ROW ]:
{
    fn add_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a += scalar );
    }
}

impl<T, const COL: usize, const ROW: usize> SubAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + SubAssign,
    [ (); COL * ROW ]:
{
    fn sub_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a -= scalar );
    }
}

impl<T, const COL: usize, const ROW: usize> MulAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + MulAssign,
    [ (); COL * ROW ]:
{
    fn mul_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a *= scalar );
    }
}

impl<T, const COL: usize, const ROW: usize> DivAssign<T> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + DivAssign,
    [ (); COL * ROW ]:
{
    fn div_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a /= scalar );
    }
}

/*
impl<T, const SHARED: usize> Determinant for Matrix<T, SHARED, SHARED>
where
    T: Default + Copy + Debug + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Num,
    [ (); SHARED * SHARED ]:,
{
    type Output = T;

    fn determinant( self ) -> Self::Output {

    }
}
*/

impl<T, const SHARED: usize, const LHS_ROW: usize> MatrixMul<Vector<T, SHARED>> for Matrix<T, SHARED, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); SHARED * LHS_ROW ]:,
    [ (); SHARED ]:
{
    type Output = Vector<T, SHARED>;

    fn mat_mul( self, rhs: Vector<T, SHARED> ) -> Self::Output {
        let mut result = Vector::<T, SHARED>::default();
        for i in 0..SHARED {
            for j in 0..LHS_ROW {
                result[ i ] += self[[ i, j ]] * rhs[ j ];
            }
        }
        result
    }
}

impl<T, const SHARED: usize, const LHS_ROW: usize, const RHS_COL: usize> MatrixMul<Matrix<T, RHS_COL, SHARED>> for Matrix<T, SHARED, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); SHARED * LHS_ROW ]:,
    [ (); RHS_COL * SHARED ]:,
    [ (); RHS_COL * LHS_ROW ]:
{
    type Output = Matrix<T, RHS_COL, LHS_ROW>;

    fn mat_mul( self, rhs: Matrix<T, RHS_COL, SHARED> ) -> Self::Output {
        let mut result = Matrix::<T, RHS_COL, LHS_ROW>::default();
        for i in 0..RHS_COL {
            for j in 0..SHARED {
                for k in 0..LHS_ROW {
                    result[[ i, k ]] += self[[ j, k ]] * rhs[[ i, j ]];
                }
            }
        }
        result
    }
}

impl<T, const SHARED: usize, const LHS_ROW: usize> MatrixMulAssignTo<Vector<T, SHARED>> for Matrix<T, SHARED, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); SHARED * LHS_ROW ]:,
    [ (); SHARED ]:
{
    type Output = Vector<T, SHARED>;

    fn mat_mul_assign_to( self, rhs: Vector<T, SHARED>, res: &mut Self::Output ) {
        for i in 0..SHARED {
            for j in 0..LHS_ROW {
                res[ i ] += self[[ i, j ]] * rhs[ j ];
            }
        }
    }
}

impl<T, const SHARED: usize, const LHS_ROW: usize, const RHS_COL: usize> MatrixMulAssignTo<Matrix<T, RHS_COL, SHARED>> for Matrix<T, SHARED, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); SHARED * LHS_ROW ]:,
    [ (); RHS_COL * SHARED ]:,
    [ (); RHS_COL * LHS_ROW ]:
{
    type Output = Matrix<T, RHS_COL, LHS_ROW>;

    fn mat_mul_assign_to( self, rhs: Matrix<T, RHS_COL, SHARED>, res: &mut Self::Output ) {
        for i in 0..RHS_COL {
            for j in 0..SHARED {
                for k in 0..LHS_ROW {
                    res[[ i, k ]] += self[[ j, k ]] * rhs[[ i, j ]];
                }
            }
        }
    }
}

impl<T, const COL: usize, const ROW: usize> InnerProduct<Matrix<T, ROW, COL>> for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); COL * ROW ]:,
    [(); ROW * COL]:,
    [(); COL * COL]:
{
    type Output = Matrix<T, COL, COL>;

    fn inner_product( self, rhs: Matrix<T, ROW, COL> ) -> Self::Output {
        let mut res = Matrix::<T, COL, COL>::default();
        for i in 0..COL {
            for j in 0..ROW {
                for k in 0..COL {
                    res[[ i, k ]] += self[[ i, j ]] * rhs[[ j, k ]];
                }
            }
        }
        res
    }
}

impl<T, const COL: usize, const ROW: usize> InnerProduct for &Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); COL * ROW ]:,
    [(); ROW * COL]:,
    [(); COL * COL]:
{
    type Output = Vector<T, COL>;

    fn inner_product( self, rhs: Self ) -> Self::Output {
        let mut res = Self::Output::default();
        for row in 0..ROW {
            for col in 0..COL {
                    res[ col ] += self[[ col, row ]] * rhs[[ col, row ]];
            }
        }
        res
    }
}

impl<T, const COL: usize, const ROW: usize> InnerProductAssignTo for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); COL * ROW ]:,
    [(); ROW * COL]:,
    [(); COL * COL]:
{
    type Output = Vector<T, COL>;

    fn inner_product_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        for row in 0..ROW {
            for col in 0..COL {
                    res[ col ] += self[[ col, row ]] * rhs[[ col, row ]];
            }
        }
    }
}

impl<T, const COL: usize, const ROW: usize> InnerProductAssignTo for &Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug + Mul<Output = T> + AddAssign,
    [ (); COL * ROW ]:,
    [(); ROW * COL]:,
    [(); COL * COL]:
{
    type Output = Vector<T, COL>;

    fn inner_product_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        for row in 0..ROW {
            for col in 0..COL {
                    res[ col ] += self[[ col, row ]] * rhs[[ col, row ]];
            }
        }
    }
}

impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> OuterProduct<Matrix<T, RHS_COL, RHS_ROW>> for Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:
{
    type Output = Tensor<T, 4, Stack<{LHS_COL * RHS_COL * LHS_ROW * RHS_ROW}>>;

    fn outer_product( self, rhs: Matrix<T, RHS_COL, RHS_ROW> ) -> Self::Output {
        let mut res = Tensor::<T, 4, Stack<{LHS_COL * RHS_COL * LHS_ROW * RHS_ROW}>>::new(
            [ LHS_COL, RHS_COL, LHS_ROW, RHS_ROW ].into(),
            [ T::default(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW ]
        );
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                for k in 0..LHS_ROW {
                    for l in 0..RHS_ROW {
                        res[[ i, j, k, l ]] = self[[ i, j ]] * rhs[[ k, l ]];
                    }
                }
            }
        }
        res
    }
}

impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> OuterProduct<&Matrix<T, RHS_COL, RHS_ROW>> for &Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:
{
    type Output = Tensor<T, 4, Stack<{LHS_COL * RHS_COL * LHS_ROW * RHS_ROW}>>;

    fn outer_product( self, rhs: &Matrix<T, RHS_COL, RHS_ROW> ) -> Self::Output {
        let mut res = Tensor::<T, 4, Stack<{LHS_COL * RHS_COL * LHS_ROW * RHS_ROW}>>::new(
            [ LHS_COL, RHS_COL, LHS_ROW, RHS_ROW ].into(),
            [ T::default(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW ]
        );
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                for k in 0..LHS_ROW {
                    for l in 0..RHS_ROW {
                        res[[ i, j, k, l ]] = self[[ i, j ]] * rhs[[ k, l ]];
                    }
                }
            }
        }
        res
    }
}

impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> OuterProductAssignTo<Matrix<T, RHS_COL, RHS_ROW>> for Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:
{
    type Output = Tensor<T, 4, Stack<{LHS_COL * RHS_COL * LHS_ROW * RHS_ROW}>>;

    fn outer_product_assign_to( self, rhs: Matrix<T, RHS_COL, RHS_ROW>, res: &mut Self::Output ) {
        if res.shape()[0] != LHS_COL { panic!( "Mismatched column length" ); }
        if res.shape()[1] != RHS_COL { panic!( "Mismatched column length" ); }
        if res.shape()[2] != LHS_ROW { panic!( "Mismatched row length" ); }
        if res.shape()[3] != RHS_ROW { panic!( "Mismatched row length" ); }
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                for k in 0..LHS_ROW {
                    for l in 0..RHS_ROW {
                        res[[ i, j, k, l ]] = self[[ i, j ]] * rhs[[ k, l ]];
                    }
                }
            }
        }
    }
}

impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> OuterProductAssignTo<&Matrix<T, RHS_COL, RHS_ROW>> for &Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:
{
    type Output = Tensor<T, 4, Stack<{LHS_COL * RHS_COL * LHS_ROW * RHS_ROW}>>;

    fn outer_product_assign_to( self, rhs: &Matrix<T, RHS_COL, RHS_ROW>, res: &mut Self::Output ) {
        if res.shape()[0] != LHS_COL { panic!( "Mismatched column length" ); }
        if res.shape()[1] != RHS_COL { panic!( "Mismatched column length" ); }
        if res.shape()[2] != LHS_ROW { panic!( "Mismatched row length" ); }
        if res.shape()[3] != RHS_ROW { panic!( "Mismatched row length" ); }
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                for k in 0..LHS_ROW {
                    for l in 0..RHS_ROW {
                        res[[ i, j, k, l ]] = self[[ i, j ]] * rhs[[ k, l ]];
                    }
                }
            }
        }
    }
}

impl<T, const COL: usize> Transpose for Matrix<T, COL, COL>
where
    T: Default + Copy + Debug,
    [(); COL * COL]:,
{
    type Output = Self;

    fn transpose( mut self ) -> Self::Output {
        for row in 0..COL {
            for col in ( row + 1 )..COL {
                let idx1 = Self::idx( col, row );
                let idx2 = Self::idx( row, col );
                let a = self.0[ idx1 ];
                let b = self.0[ idx2 ];
                self.0[ idx1 ] = b;
                self.0[ idx2 ] = a;
            }
        }
        self
    }
}

//TODO: Implement in-place non-square matrix transpose
#[cfg(any())]
impl<T, const COL: usize, const ROW: usize> Transpose for Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug,
    [(); COL * COL]:,
{
    type Output = Self;

    fn transpose( mut self ) -> Self::Output {
        for row in 0..COL {
            for col in ( row + 1 )..COL {
                let idx1 = Self::idx( col, row );
                let idx2 = Self::idx( row, col );
                let a = self.0[ idx1 ];
                let b = self.0[ idx2 ];
                self.0[ idx1 ] = b;
                self.0[ idx2 ] = a;
            }
        }
        unsafe { *( &self as *const Matrix<T, COL, ROW> as *const Matrix<T, ROW, COL> ) } // SAFETY: The transposed matrix is the same size as the original matrix.
    }
}

impl<T, const COL: usize> TransposeAssign for Matrix<T, COL, COL>
where
    T: Default + Copy + Debug,
    [(); COL * COL]:,
{
    fn transpose_assign( &mut self ) {
        for row in 0..COL {
            for col in ( row + 1 )..COL {
                let idx1 = Self::idx( col, row );
                let idx2 = Self::idx( row, col );
                let a = self.0[ idx1 ];
                let b = self.0[ idx2 ];
                self.0[ idx1 ] = b;
                self.0[ idx2 ] = a;
            }
        }
    }
}


impl<T, const COL: usize, const ROW: usize> Transpose for &Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug,
    [ (); COL * ROW ]:,
    [(); ROW * COL]:
{
    type Output = Matrix<T, ROW, COL>;

    fn transpose( self ) -> Self::Output {
        let mut res = Matrix::<T, ROW, COL>::default();
        for row in 0..ROW {
            for col in 0..COL {
                let idx1 = Matrix::<T, COL, ROW>::idx( col, row );
                let idx2 = Matrix::<T, ROW, COL>::idx( row, col );
                res[ idx2 ] = self.0[ idx1 ];
            }
        }
        res
    }
}

impl<T, const COL: usize, const ROW: usize> TransposeAssignTo for &Matrix<T, COL, ROW>
where
    T: Default + Copy + Debug,
    [ (); COL * ROW ]:,
    [(); ROW * COL]:
{
    type Output = Matrix<T, ROW, COL>;

    fn transpose_assign_to( self, res: &mut Self::Output ) {
        for row in 0..ROW {
            for col in 0..COL {
                let idx1 = Matrix::<T, COL, ROW>::idx( col, row );
                let idx2 = Matrix::<T, ROW, COL>::idx( row, col );
                res[ idx2 ] = self.0[ idx1 ];
            }
        }
    }
}

/*
impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> KroneckerProduct<Matrix<T, RHS_COL, RHS_ROW>> for Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: OuterProduct<Matrix<T, RHS_COL, RHS_ROW>>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL]:,
    [(); LHS_ROW * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:,
{
    type Output = Matrix<T, {LHS_COL * RHS_COL}, {LHS_ROW * RHS_ROW}>;

    fn kronecker_product( self, rhs: Matrix<T, RHS_COL, RHS_ROW> ) -> Self::Output {
        (&self).outer_product( &rhs ).into()
    }
}

impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> KroneckerProduct<&Matrix<T, RHS_COL, RHS_ROW>> for &Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: OuterProduct<Matrix<T, RHS_COL, RHS_ROW>>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL]:,
    [(); LHS_ROW * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:,
{
    type Output = Matrix<T, {LHS_COL * RHS_COL}, {LHS_ROW * RHS_ROW}>;

    fn kronecker_product( self, rhs: &Matrix<T, RHS_COL, RHS_ROW> ) -> Self::Output {
        self.outer_product( rhs ).into()
    }
}

impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> KroneckerProductAssignTo<Matrix<T, RHS_COL, RHS_ROW>> for Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: OuterProduct<Matrix<T, RHS_COL, RHS_ROW>>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL]:,
    [(); LHS_ROW * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:,
{
    type Output = Matrix<T, {LHS_COL * RHS_COL}, {LHS_ROW * RHS_ROW}>;

    fn kronecker_product_assign_to( self, rhs: Matrix<T, RHS_COL, RHS_ROW>, res: &mut Self::Output ) {
        (&self).outer_product_assign_to( &rhs, &mut (*res).into() );
    }
}

impl<T, const LHS_COL: usize, const LHS_ROW: usize, const RHS_COL: usize, const RHS_ROW: usize> KroneckerProductAssignTo<&Matrix<T, RHS_COL, RHS_ROW>> for &Matrix<T, LHS_COL, LHS_ROW>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: OuterProduct<Matrix<T, RHS_COL, RHS_ROW>>,
    [(); LHS_COL * LHS_ROW]:,
    [(); RHS_COL * RHS_ROW]:,
    [(); LHS_COL * RHS_COL]:,
    [(); LHS_ROW * RHS_ROW]:,
    [(); LHS_COL * RHS_COL * LHS_ROW * RHS_ROW]:,
{
    type Output = Matrix<T, {LHS_COL * RHS_COL}, {LHS_ROW * RHS_ROW}>;

    fn kronecker_product_assign_to( self, rhs: &Matrix<T, RHS_COL, RHS_ROW>, res: &mut Self::Output ) {
        self.outer_product_assign_to( rhs, &mut (*res).into() );
    }
}
*/

pub type Matrix2x2<T> = Matrix<T, 2, 2>;
pub type Matrix2x3<T> = Matrix<T, 2, 3>;
pub type Matrix2x4<T> = Matrix<T, 2, 4>;
pub type Matrix2x5<T> = Matrix<T, 2, 5>;
pub type Matrix2x6<T> = Matrix<T, 2, 6>;
pub type Matrix2x7<T> = Matrix<T, 2, 7>;
pub type Matrix2x8<T> = Matrix<T, 2, 8>;
pub type Matrix2x9<T> = Matrix<T, 2, 9>;

pub type Matrix3x2<T> = Matrix<T, 3, 2>;
pub type Matrix3x3<T> = Matrix<T, 3, 3>;
pub type Matrix3x4<T> = Matrix<T, 3, 4>;
pub type Matrix3x5<T> = Matrix<T, 3, 5>;
pub type Matrix3x6<T> = Matrix<T, 3, 6>;
pub type Matrix3x7<T> = Matrix<T, 3, 7>;
pub type Matrix3x8<T> = Matrix<T, 3, 8>;
pub type Matrix3x9<T> = Matrix<T, 3, 9>;

pub type Matrix4x2<T> = Matrix<T, 4, 2>;
pub type Matrix4x3<T> = Matrix<T, 4, 3>;
pub type Matrix4x4<T> = Matrix<T, 4, 4>;
pub type Matrix4x5<T> = Matrix<T, 4, 5>;
pub type Matrix4x6<T> = Matrix<T, 4, 6>;
pub type Matrix4x7<T> = Matrix<T, 4, 7>;
pub type Matrix4x8<T> = Matrix<T, 4, 8>;
pub type Matrix4x9<T> = Matrix<T, 4, 9>;

pub type Matrix5x2<T> = Matrix<T, 5, 2>;
pub type Matrix5x3<T> = Matrix<T, 5, 3>;
pub type Matrix5x4<T> = Matrix<T, 5, 4>;
pub type Matrix5x5<T> = Matrix<T, 5, 5>;
pub type Matrix5x6<T> = Matrix<T, 5, 6>;
pub type Matrix5x7<T> = Matrix<T, 5, 7>;
pub type Matrix5x8<T> = Matrix<T, 5, 8>;
pub type Matrix5x9<T> = Matrix<T, 5, 9>;

pub type Matrix6x2<T> = Matrix<T, 6, 2>;
pub type Matrix6x3<T> = Matrix<T, 6, 3>;
pub type Matrix6x4<T> = Matrix<T, 6, 4>;
pub type Matrix6x5<T> = Matrix<T, 6, 5>;
pub type Matrix6x6<T> = Matrix<T, 6, 6>;
pub type Matrix6x7<T> = Matrix<T, 6, 7>;
pub type Matrix6x8<T> = Matrix<T, 6, 8>;
pub type Matrix6x9<T> = Matrix<T, 6, 9>;

pub type Matrix7x2<T> = Matrix<T, 7, 2>;
pub type Matrix7x3<T> = Matrix<T, 7, 3>;
pub type Matrix7x4<T> = Matrix<T, 7, 4>;
pub type Matrix7x5<T> = Matrix<T, 7, 5>;
pub type Matrix7x6<T> = Matrix<T, 7, 6>;
pub type Matrix7x7<T> = Matrix<T, 7, 7>;
pub type Matrix7x8<T> = Matrix<T, 7, 8>;
pub type Matrix7x9<T> = Matrix<T, 7, 9>;

pub type Matrix8x2<T> = Matrix<T, 8, 2>;
pub type Matrix8x3<T> = Matrix<T, 8, 3>;
pub type Matrix8x4<T> = Matrix<T, 8, 4>;
pub type Matrix8x5<T> = Matrix<T, 8, 5>;
pub type Matrix8x6<T> = Matrix<T, 8, 6>;
pub type Matrix8x7<T> = Matrix<T, 8, 7>;
pub type Matrix8x8<T> = Matrix<T, 8, 8>;
pub type Matrix8x9<T> = Matrix<T, 8, 9>;

pub type Matrix9x2<T> = Matrix<T, 9, 2>;
pub type Matrix9x3<T> = Matrix<T, 9, 3>;
pub type Matrix9x4<T> = Matrix<T, 9, 4>;
pub type Matrix9x5<T> = Matrix<T, 9, 5>;
pub type Matrix9x6<T> = Matrix<T, 9, 6>;
pub type Matrix9x7<T> = Matrix<T, 9, 7>;
pub type Matrix9x8<T> = Matrix<T, 9, 8>;
pub type Matrix9x9<T> = Matrix<T, 9, 9>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let matrix = Matrix2x2::<f32>::zero();
        assert_eq!( matrix[ 0 ], 0.0 );
    }

    #[test]
    fn default_test() {
        let matrix = Matrix2x2::<f32>::default();
        assert_eq!( matrix[ 0 ], 0.0 );
    }

    #[test]
    fn iter_test() {
        let src = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ];
        let matrix = Matrix3x3::<u32>::from( src );
        for ( i, value ) in matrix.deref().iter().enumerate() {
            assert_eq!( value, &src[ i ] );
        }
    }

    #[test]
    fn mat_mul_test() {
        use crate::ops::MatrixMulAssignTo;

        let a = Matrix2x2::<f32>::from([
            1.0, 2.0,
            3.0, 4.0
        ]);

        let b = Matrix2x2::<f32>::from([
            1.0, 2.0,
            3.0, 4.0
        ]);

        let mut c = Matrix2x2::<f32>::from([
            0.0, 0.0,
            0.0, 0.0
        ]);

        println!( "Before:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        a.mat_mul_assign_to( b, &mut c );

        println!( "After:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        assert_eq!( c[[0, 0]], 7.0 );
        assert_eq!( c[[1, 0]], 10.0 );
        assert_eq!( c[[0, 1]], 15.0 );
        assert_eq!( c[[1, 1]], 22.0 );
    }

    #[test]
    fn transpose_inplace_square_test() {
        let src = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ];
        let matrix = Matrix3x3::<u32>::from( src );
        let transposed = matrix.transpose();
        println!( "Before:" );
        println!( "matrix: {:?}", matrix );
        println!( "After:" );
        println!( "transposed: {:?}", transposed );
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!( matrix[[ i, j ]], transposed[[ j, i ]] );
            }
        }
    }

    #[test]
    fn transpose_non_square_test() {
        let src = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ];
        let matrix = Matrix4x3::<u32>::from( src );
        let transposed = matrix.transpose();
        println!( "Before:" );
        println!( "matrix: {:?}", matrix );
        println!( "After:" );
        println!( "transposed: {:?}", transposed );
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!( matrix[[ i, j ]], transposed[[ j, i ]] );
            }
        }
    }
}
