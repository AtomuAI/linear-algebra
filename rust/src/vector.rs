// Copyright 2024 Bewusstsein Labs

//mod test;

use std::{
    default, fmt::Debug, ops::{ Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign }
};
use num::traits::Num;

use const_expr_bounds::{ Assert, IsTrue };
use memory::{ stack::Stack, heap::Heap };
use arithmetic::{ AddAssignTo, SubAssignTo, MulAssignTo, DivAssignTo };

use crate::{
    traits::{
        X,
        XMut,
        Y,
        YMut,
        Z,
        ZMut,
        Columns,
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
        Append,
        AppendAssignTo,
        Split,
        SplitAssignTo,
        InnerProduct,
        OuterProduct,
        OuterProductAssignTo,
        TensorProduct,
        TensorProductAssignTo,
    },
    matrix::Matrix,
    shape::Shape,
    tensor::{ Tensor, TensorAccess, TensorTraits }
};

/// A vector type of generic element and size.
///
#[derive( Clone, Copy, Debug )]
pub struct Vector<T: 'static + Default + Copy + Debug, const COL: usize>( [ T; COL ] );

impl<T, const COL: usize> Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    /// Creates a new const [`Vector`].
    ///
    pub const fn new_const( src: [T; COL] ) -> Self {
        Self ( src )
    }

    /// Creates a new [`Vector`].
    ///
    pub fn new( src: [T; COL] ) -> Self {
        Self ( src )
    }

    /// Creates a new zero filled [`Vector`].
    ///
    pub fn zero() -> Self
    where
        T: Num
    {
        Self ( [T::zero(); COL] )
    }

    /// Returns the order of the [`Vector`].
    ///
    /// The order of a [`Vector`] is always 1.
    ///
    #[inline(always)]
    pub const fn ord( &self ) -> usize
    where
        Self: ConstOrder
    {
        Self::ORD
    }

    /// Returns the shape of the [`Vector`].
    ///
    /// The shape of a [`Vector`] is always a single column.
    ///
    #[inline(always)]
    pub const fn shape( &self ) -> Shape<1>
    where
        Self: ConstShaped<1>
    {
        Self::SHAPE
    }

    /// Returns the dot product of two [`Vector`]s.
    ///
    /// The dot product of two [`Vector`]s is the sum of the products of the corresponding elements of the two [`Vector`]s.
    ///
    pub fn dot( &self, other: &Self ) -> T
    where
        T: Default + Copy + Debug + Add<Output = T> + Mul<Output = T>,
        Self: InnerProduct<Self, Output = T>
    {
        self.inner_product( other )
    }

    /// Returns an iterator over the elements of the [`Vector`].
    ///
    /// The iterator yields references to the elements of the [`Vector`] in order.
    ///
    pub fn iter( &self ) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Returns an iterator over mutable references to the elements of the [`Vector`].
    ///
    /// The iterator yields mutable references to the elements of the [`Vector`] in order.
    ///
    pub fn iter_mut( &mut self ) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }
}

impl<T, const COL: usize> X<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn x( &self ) -> &T {
        &self[ 0 ]
    }
}

impl<T, const COL: usize> XMut<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn x_mut( &mut self ) -> &mut T {
        &mut self[ 0 ]
    }
}

impl<T, const COL: usize> Y<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn y( &self ) -> &T {
        &self[ 1 ]
    }
}

impl<T, const COL: usize> YMut<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn y_mut( &mut self ) -> &mut T {
        &mut self[ 1 ]
    }
}

impl<T, const COL: usize> Z<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn z( &self ) -> &T {
        &self[ 2 ]
    }
}

impl<T, const COL: usize> ZMut<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn z_mut( &mut self ) -> &mut T {
        &mut self[ 2 ]
    }
}

impl<T, const COL: usize> Columns for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    const COLS: usize = COL;
}

impl<T, const COL: usize> ConstOrder for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    const ORD: usize = 1;
}

impl<T, const COL: usize> ConstShaped<1> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    const SHAPE: Shape<1> = Shape::new_const([ COL ]);
}

impl<T, const COL: usize> ConstSized for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    const SIZE: usize = COL;
}

impl<T, const OLD_COL: usize, const NEW_COL: usize> ConstReSizeable<Vector<T, NEW_COL>> for Vector<T, OLD_COL>
where
    T: 'static + Copy + Default + Debug
{
    fn resize( self ) -> Vector<T, NEW_COL> {
        let mut res = Vector::<T, NEW_COL>([T::default(); NEW_COL]);
        let len = OLD_COL.min( NEW_COL );
        res.0[ ..len ].copy_from_slice( &self.0[ ..len ] );
        res
    }
}

impl<T, const OLD_COL: usize, const NEW_COL: usize, const NEW_ROW: usize> ConstReOrder<Matrix<T, NEW_COL, NEW_ROW>> for Vector<T, OLD_COL>
where
    Assert<{ NEW_COL * NEW_ROW == OLD_COL }>: IsTrue,
    T: 'static + Copy + Default + Debug,
    [ (); NEW_COL * NEW_ROW ]:
{
    fn reorder( self ) -> Matrix<T, NEW_COL, NEW_ROW> {
        unsafe { *( &self as *const Vector<T, OLD_COL> as *const Matrix<T, NEW_COL, NEW_ROW> ) } // SAFETY: This is safe because we have asserted that the total number of elements is the same
    }
}

impl<T, const COL: usize> Fillable<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn fill( &mut self, value: T ) {
        self.0 = [ value; COL ];
    }
}

impl<T, const COL: usize> Zeroable<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug + Num
{
    fn zero( &mut self ) {
        self.0 = [ T::zero(); COL ];
    }
}

impl<T, const COL: usize> Clearable<T> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug + Num
{
    fn clear( &mut self ) {
        self.0 = [ T::default(); COL ];
    }
}

impl<T, const COL: usize> Deref for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    type Target = [T; COL];

    fn deref( &self ) -> &Self::Target {
        &self.0
    }
}

impl<T, const COL: usize> DerefMut for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const COL: usize> Index<usize> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    type Output = T;

    fn index( &self, index: usize ) -> &Self::Output {
        &self.0[ index ]
    }
}

impl<T, const COL: usize> IndexMut<usize> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn index_mut( &mut self, index: usize ) -> &mut Self::Output {
        &mut self.0[ index ]
    }
}

impl<T, const COL: usize> Default for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn default() -> Self {
        Self ( [T::default(); COL] )
    }
}

impl<T, const COL: usize> From<[T; COL]> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: [T; COL] ) -> Self {
        Self ( src )
    }
}

impl<T, const COL: usize, const ROW: usize> From<Matrix<T, COL, ROW>> for Vector<T, {COL * ROW}>
where
    T: 'static + Copy + Default + Debug,
    Matrix<T, COL, ROW>: ConstReOrder<Vector<T, {COL * ROW}>>,
    [(); COL * ROW]:
{
    fn from( src: Matrix<T, COL, ROW> ) -> Self {
        src.reorder()
    }
}

impl<T, const COL: usize> From<Tensor<T, 1, Stack<COL>>> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug
{
    fn from( src: Tensor<T, 1, Stack<COL>> ) -> Self {
        Self ( ***src.memory() )
    }
}

impl<T, const COL: usize> From<Tensor<T, 1, Heap>> for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug,
    Tensor<T, 1, Heap>: DynShaped<1>,
{
    fn from( src: Tensor<T, 1, Heap> ) -> Self {
        if src.shape()[0] != COL { panic!( "Mismatched column length" ); }
        let mut this = Self::default();
        this.iter_mut().zip( src.iter() )
            .for_each( |( a, &b )| *a = b );
        this
    }
}

impl<T, const COL: usize> PartialEq for Vector<T, COL>
where
    T: 'static + Copy + Default + Debug + PartialEq
{
    fn eq( &self, other: &Self ) -> bool {
        self.0 == other.0
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> Add<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( mut self, other: Vector<T, RHS_COL> ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a + b );
        self
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> Sub<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( mut self, other: Vector<T, RHS_COL> ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a - b );
        self
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> Mul<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( mut self, other: Vector<T, RHS_COL> ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a * b );
        self
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> Div<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( mut self, other: Vector<T, RHS_COL> ) -> Self::Output {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a = *a / b );
        self
    }
}

impl<T, const COL: usize> Add<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn add( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a + scalar );
        self
    }
}

impl<T, const COL: usize> Sub<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn sub( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a - scalar );
        self
    }
}

impl<T, const COL: usize> Mul<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn mul( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a * scalar );
        self
    }
}

impl<T, const COL: usize> Div<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Self;

    fn div( mut self, scalar: T ) -> Self::Output {
        self.iter_mut()
            .for_each( |a| *a = *a / scalar );
        self
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> AddAssign<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, other: Vector<T, RHS_COL> ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a += b );
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> SubAssign<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, other: Vector<T, RHS_COL> ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a -= b );
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> MulAssign<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, other: Vector<T, RHS_COL> ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a *= b );
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> DivAssign<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, other: Vector<T, RHS_COL> ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a /= b );
    }
}

impl<T, const COL: usize> AddAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a += scalar );
    }
}

impl<T, const COL: usize> SubAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a -= scalar );
    }
}

impl<T, const COL: usize> MulAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a *= scalar );
    }
}

impl<T, const COL: usize> DivAssign<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a /= scalar );
    }
}

impl<T, const COL: usize> AddAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Self;

    fn add_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
    }
}

impl<T, const COL: usize> SubAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Self;

    fn sub_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
    }
}

impl<T, const COL: usize> MulAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Self;

    fn mul_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
    }
}

impl<T, const COL: usize> DivAssignTo for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Self;

    fn div_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
    }
}

impl<T, const COL: usize> AddAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Self;

    fn add_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
    }
}

impl<T, const COL: usize> SubAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Self;

    fn sub_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
    }
}

impl<T, const COL: usize> MulAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Self;

    fn mul_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
    }
}

impl<T, const COL: usize> DivAssignTo<T> for Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Self;

    fn div_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
    }
}

//

impl<T, const COL: usize> Add for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn add( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
        result
    }
}

impl<T, const COL: usize> Sub for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn sub( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
        result
    }
}

impl<T, const COL: usize> Mul for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn mul( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
        result
    }
}

impl<T, const COL: usize> Div for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn div( self, other: Self ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( other.iter() ).zip( result.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
        result
    }
}

impl<T, const COL: usize> Add<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn add( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
        result
    }
}

impl<T, const COL: usize> Sub<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn sub( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
        result
    }
}

impl<T, const COL: usize> Mul<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn mul( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
        result
    }
}

impl<T, const COL: usize> Div<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>,
    Self: Clone
{
    type Output = Vector<T, COL>;

    fn div( self, scalar: T ) -> Self::Output {
        let mut result = Vector::<T, COL>::default();
        self.iter().zip( result.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
        result
    }
}

impl<T, const COL: usize> AddAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a += b );
    }
}

impl<T, const COL: usize> SubAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a -= b );
    }
}

impl<T, const COL: usize> MulAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a *= b );
    }
}

impl<T, const COL: usize> DivAssign for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, other: Self ) {
        self.iter_mut().zip( other.iter() )
            .for_each( |( a, &b )| *a /= b );
    }
}

impl<T, const COL: usize> AddAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + AddAssign,
{
    fn add_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a += scalar );
    }
}

impl<T, const COL: usize> SubAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + SubAssign,
{
    fn sub_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a -= scalar );
    }
}

impl<T, const COL: usize> MulAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + MulAssign,
{
    fn mul_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a *= scalar );
    }
}

impl<T, const COL: usize> DivAssign<T> for &mut Vector<T, COL>
where
    T: Default + Copy + Debug + DivAssign,
{
    fn div_assign( &mut self, scalar: T ) {
        self.iter_mut()
            .for_each( |a| *a /= scalar );
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> Append<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug,
    [(); LHS_COL + RHS_COL]:
{
    type Output = Vector<T, {LHS_COL + RHS_COL}>;

    fn append( self, other: Vector<T, RHS_COL> ) -> Self::Output {
        let mut result = Vector::<T, { LHS_COL + RHS_COL }>( [ T::default(); LHS_COL + RHS_COL ] );
        result.0[ ..LHS_COL ].copy_from_slice( &self.0 );
        result.0[ LHS_COL.. ].copy_from_slice( &other.0 );
        result
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> AppendAssignTo<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug,
    [(); LHS_COL + RHS_COL]:
{
    type Output = Vector<T, {LHS_COL + RHS_COL}>;

    fn append_assign_to( self, rhs: Vector<T, RHS_COL>, res: &mut Self::Output ) {
        res.0[ ..LHS_COL ].copy_from_slice( &self.0 );
        res.0[ LHS_COL.. ].copy_from_slice( &rhs.0 );
    }
}

impl<T, const A_COL: usize, const B_COL: usize> Split<A_COL, B_COL> for Vector<T, {A_COL + B_COL}>
where
    T: Default + Copy + Debug,
    [(); A_COL + B_COL]:,
    [(); A_COL]:,
    [(); B_COL]:
{
    type OutputA = Vector<T, A_COL>;
    type OutputB = Vector<T, B_COL>;

    fn split( self ) -> ( Self::OutputA, Self::OutputB ) {
        let mut a = Vector::<T, A_COL>( [ T::default(); A_COL ] );
        let mut b = Vector::<T, B_COL>( [ T::default(); B_COL ] );
        a.0.copy_from_slice( &self.0[ ..A_COL ] );
        b.0.copy_from_slice( &self.0[ A_COL.. ] );
        ( a, b )
    }
}

impl<T, const A_COL: usize, const B_COL: usize> SplitAssignTo<A_COL, B_COL> for Vector<T, {A_COL + B_COL}>
where
    T: Default + Copy + Debug,
    [(); A_COL + B_COL]:,
    [(); A_COL]:,
    [(); B_COL]:
{
    type OutputA = Vector<T, A_COL>;
    type OutputB = Vector<T, B_COL>;

    fn split_assign_to( self, res: ( &mut Self::OutputA, &mut Self::OutputB ) ) {
        res.0.0.copy_from_slice( &self.0[ ..A_COL ] );
        res.1.0.copy_from_slice( &self.0[ A_COL.. ] );
    }
}

impl<T, const COL: usize> AddAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Vector<T, COL>;

    fn add_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a + b );
    }
}

impl<T, const COL: usize> SubAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Vector<T, COL>;

    fn sub_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a - b );
    }
}

impl<T, const COL: usize> MulAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Vector<T, COL>;

    fn mul_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a * b );
    }
}

impl<T, const COL: usize> DivAssignTo for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Vector<T, COL>;

    fn div_assign_to( self, rhs: Self, res: &mut Self::Output ) {
        self.iter().zip( rhs.iter() ).zip( res.iter_mut() )
            .for_each( |( ( &a, &b ), c )| *c = a / b );
    }
}

impl<T, const COL: usize> AddAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Add<Output = T>
{
    type Output = Vector<T, COL>;

    fn add_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a + scalar );
    }
}

impl<T, const COL: usize> SubAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>
{
    type Output = Vector<T, COL>;

    fn sub_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a - scalar );
    }
}

impl<T, const COL: usize> MulAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T>
{
    type Output = Vector<T, COL>;

    fn mul_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a * scalar );
    }
}

impl<T, const COL: usize> DivAssignTo<T> for &Vector<T, COL>
where
    T: Default + Copy + Debug + Div<Output = T>
{
    type Output = Vector<T, COL>;

    fn div_assign_to( self, scalar: T, res: &mut Self::Output ) {
        self.iter().zip( res.iter_mut() )
            .for_each( |( &a, c )| *c = a / scalar );
    }
}

impl<T, const COL: usize> InnerProduct for Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T> + Add<Output = T>
{
    type Output = T;

    fn inner_product( self, rhs: Self ) -> Self::Output {
        self.iter().zip( rhs.iter() )
            .fold( T::default(), |acc, ( &a, &b )| acc + a * b )
    }
}

impl<T, const COL: usize> InnerProduct for &Vector<T, COL>
where
    T: Default + Copy + Debug + Mul<Output = T> + Add<Output = T>
{
    type Output = T;

    fn inner_product( self, rhs: Self ) -> Self::Output {
        self.iter().zip( rhs.iter() )
            .fold( T::default(), |acc, ( &a, &b )| acc + a * b )
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> OuterProduct<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Matrix<T, LHS_COL, RHS_COL>;

    fn outer_product( self, rhs: Vector<T, RHS_COL> ) -> Self::Output {
        let mut res = Matrix::<T, LHS_COL, RHS_COL>::default();
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                res[[ i, j ]] = self[ i ] * rhs[ j ];
            }
        }
        res
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> OuterProduct<&Vector<T, RHS_COL>> for &Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Matrix<T, LHS_COL, RHS_COL>;

    fn outer_product( self, rhs: &Vector<T, RHS_COL> ) -> Self::Output {
        let mut res = Matrix::<T, LHS_COL, RHS_COL>::default();
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                res[[ i, j ]] = self[ i ] * rhs[ j ];
            }
        }
        res
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> OuterProductAssignTo<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Matrix<T, LHS_COL, RHS_COL>;

    fn outer_product_assign_to( self, rhs: Vector<T, RHS_COL>, res: &mut Self::Output ) {
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                res[[ i, j ]] = self[ i ] * rhs[ j ];
            }
        }
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> OuterProductAssignTo<&Vector<T, RHS_COL>> for &Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Matrix<T, LHS_COL, RHS_COL>;

    fn outer_product_assign_to( self, rhs: &Vector<T, RHS_COL>, res: &mut Self::Output ) {
        for i in 0..LHS_COL {
            for j in 0..RHS_COL {
                res[[ i, j ]] = self[ i ] * rhs[ j ];
            }
        }
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> TensorProduct<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Self: OuterProduct<Vector<T, RHS_COL>>,
    Matrix<T, LHS_COL, RHS_COL>: ConstReOrder<Vector<T, {LHS_COL * RHS_COL}>>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Vector<T, {LHS_COL * RHS_COL}>;

    fn tensor_product( self, rhs: Vector<T, RHS_COL> ) -> Self::Output {
        (&self).outer_product( &rhs ).reorder()
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> TensorProduct<&Vector<T, RHS_COL>> for &Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Matrix<T, LHS_COL, RHS_COL>: ConstReOrder<Vector<T, {LHS_COL * RHS_COL}>>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Vector<T, {LHS_COL * RHS_COL}>;

    fn tensor_product( self, rhs: &Vector<T, RHS_COL> ) -> Self::Output {
        self.outer_product( rhs ).reorder()
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> TensorProductAssignTo<Vector<T, RHS_COL>> for Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Vector<T, { LHS_COL * RHS_COL }>: ConstReOrder<Matrix<T, LHS_COL, RHS_COL>>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Vector<T, {LHS_COL * RHS_COL}>;

    fn tensor_product_assign_to( self, rhs: Vector<T, RHS_COL>, res: &mut Self::Output ) {
        (&self).outer_product_assign_to( &rhs, &mut (*res).reorder() );
    }
}

impl<T, const LHS_COL: usize, const RHS_COL: usize> TensorProductAssignTo<&Vector<T, RHS_COL>> for &Vector<T, LHS_COL>
where
    T: Default + Copy + Debug + Mul<Output = T>,
    Vector<T, { LHS_COL * RHS_COL }>: ConstReOrder<Matrix<T, LHS_COL, RHS_COL>>,
    [(); LHS_COL * RHS_COL]:
{
    type Output = Vector<T, {LHS_COL * RHS_COL}>;

    fn tensor_product_assign_to( self, rhs: &Vector<T, RHS_COL>, res: &mut Self::Output ) {
        self.outer_product_assign_to( rhs, &mut (*res).reorder() );
    }
}

pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector4<T> = Vector<T, 4>;
pub type Vector5<T> = Vector<T, 5>;
pub type Vector6<T> = Vector<T, 6>;
pub type Vector7<T> = Vector<T, 7>;
pub type Vector8<T> = Vector<T, 8>;
pub type Vector9<T> = Vector<T, 9>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_test() {
        let vector = Vector2::<f32>::zero();
        assert_eq!( vector[ 0 ], 0.0 );
    }

    #[test]
    fn default_test() {
        let vector = Vector2::<f32>::default();
        assert_eq!( vector[ 0 ], 0.0 );
    }

    #[test]
    fn iter_test() {
        let src = [ 1, 2, 3, 4, 5 ];
        let vector = Vector::<u32, 5>::from( src );
        for ( i, value ) in vector.iter().enumerate() {
            assert_eq!( value, &src[ i ] );
        }
    }

    #[test]
    fn mat_mul_test() {
        use crate::matrix::Matrix2x2;
        use crate::ops::MatrixMulAssignTo;

        let a = Vector2::<f32>::from([
            1.0, 2.0
        ]);

        let b = Matrix2x2::<f32>::from([
            1.0, 2.0,
            3.0, 4.0
        ]);

        let mut c = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        /*
        let d = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        let e = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        let mut f = Vector2::<f32>::from([
            0.0, 0.0
        ]);

        <&Vector2<f32>>::add_assign_to( &d, &e, &mut f );
        */

        println!( "Before:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        b.mat_mul_assign_to( a, &mut c );

        println!( "After:");
        println!( "a: {:?}", a );
        println!( "b: {:?}", b );
        println!( "c: {:?}", c );

        assert_eq!( c[0], 7.0 );
        assert_eq!( c[1], 10.0 );
    }

    #[test]
    fn add_test() {
        let a = Vector2::<f32>::from([
            1.0, 2.0
        ]);

        let b = Vector2::<f32>::from([
            3.0, 4.0
        ]);

        let c = &a + &b;

        println!( "{:?} = <{:?}, {:?}>", c, a, b );
    }

    #[test]
    fn dot_test() {
        let a = Vector2::<f32>::from([
            1.0, 2.0
        ]);

        let b = Vector2::<f32>::from([
            3.0, 4.0
        ]);

        let c = a.dot( &b );

        println!( "{:?} = <{:?}, {:?}>", c, a, b );
        assert_eq!( c, 11.0 );
    }

    #[test]
    fn append_test() {
        let a = Vector2::<f32>::from([
            1.0, 2.0
        ]);

        let b = Vector3::<f32>::from([
            3.0, 4.0, 5.0
        ]);

        let c = a.append( b );

        println!( "{:?} = <{:?}, {:?}>", c, a, b );
    }

    #[test]
    fn split_test() {
        let a = Vector5::<f32>::from([
            1.0, 2.0, 3.0, 4.0, 5.0
        ]);

        let ( b, c ): ( Vector3<f32>, Vector2<f32> ) = a.split();

        println!( "<{:?}, {:?}> = {:?}", b, c, a );
    }
}
