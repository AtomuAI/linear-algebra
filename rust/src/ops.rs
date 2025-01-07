// Copyright 2024 Bewusstsein Labs

/// The append operator.
///
pub trait Append<Rhs = Self> {
    type Output;

    fn append( self, rhs: Rhs ) -> Self::Output;
}

/// The append assign to operator.
///
pub trait AppendAssignTo<Rhs = Self> {
    type Output;

    fn append_assign_to( self, rhs: Rhs, res: &mut Self::Output );
}

/// The split operator.
///
pub trait Split<const A_SIZE: usize, const B_SIZE: usize> {
    type OutputA;
    type OutputB;

    fn split( self ) -> ( Self::OutputA, Self::OutputB );
}

/// The split assign to operator.
///
pub trait SplitAssignTo<const A_SIZE: usize, const B_SIZE: usize> {
    type OutputA;
    type OutputB;

    fn split_assign_to( self, res: ( &mut Self::OutputA, &mut Self::OutputB ) );
}

/// The contraction operator.
///
pub trait MatrixMul<Rhs = Self> {
    type Output;

    fn mat_mul( self, rhs: Rhs ) -> Self::Output;
}

/// The contraction assign to operator.
///
pub trait MatrixMulAssignTo<Rhs = Self> {
    type Output;

    fn mat_mul_assign_to( self, rhs: Rhs, res: &mut Self::Output );
}

/// The contraction operator.
///
pub trait Contract<const CTR_ORD: usize, Rhs = Self, Res = Self> {
    fn contract( self, lhs_dims: [ usize; CTR_ORD ], rhs_dims: [ usize; CTR_ORD ], rhs: Rhs ) -> Res;
}

/// The contraction assign to operator.
///
pub trait ContractAssignTo<const CTR_ORD: usize, Rhs = Self, Res = Self> {
    fn contract_assign_to( self, lhs_dims: [ usize; CTR_ORD ], rhs_dims: [ usize; CTR_ORD ], rhs: Rhs, res: &mut Res );
}

/// The inner product operator.
///
pub trait InnerProduct<Rhs = Self> {
    type Output;

    /// Performs the inner product operation.
    ///
    fn inner_product( self, rhs: Rhs ) -> Self::Output;
}

/// The inner product assign to operator.
///
pub trait InnerProductAssignTo<Rhs = Self> {
    type Output;

    /// Performs the inner product assign to operation.
    ///
    fn inner_product_assign_to( self, rhs: Rhs, res: &mut Self::Output );
}

/// The outer product operator.
///
pub trait OuterProduct<Rhs = Self> {
    type Output;

    /// Performs the outer product operation.
    ///
    fn outer_product( self, rhs: Rhs ) -> Self::Output;
}

/// The outer product assign to operator.
///
pub trait OuterProductAssignTo<Rhs = Self> {
    type Output;

    /// Performs the outer product assign to operation.
    ///
    fn outer_product_assign_to( self, rhs: Rhs, res: &mut Self::Output );
}

/// The transpose operator.
///
pub trait Transpose {
    type Output;

    /// Performs the transpose operation.
    ///
    fn transpose( self ) -> Self::Output;
}

/// The transpose assign operator.
///
pub trait TransposeAssign {
    /// Performs the transpose assign operation.
    ///
    fn transpose_assign( &mut self );
}

/// The transpose assign to operator.
///
pub trait TransposeAssignTo {
    type Output;

    /// Performs the transpose assign to operation.
    ///
    fn transpose_assign_to( self, res: &mut Self::Output );
}

/// The tensor product operator.
///
pub trait TensorProduct<Rhs = Self> {
    type Output;

    /// Performs the tensor product operation.
    ///
    fn tensor_product( self, rhs: Rhs ) -> Self::Output;
}

/// The tensor product assign to operator.
///
pub trait TensorProductAssignTo<Rhs = Self> {
    type Output;

    /// Performs the tensor product assign to operation.
    ///
    fn tensor_product_assign_to( self, rhs: Rhs, res: &mut Self::Output );
}

/// The kronecker product operator.
///
pub trait KroneckerProduct<Rhs = Self> {
    type Output;

    /// Performs the kronecker product operation.
    ///
    fn kronecker_product( self, rhs: Rhs ) -> Self::Output;
}

/// The kronecker product assign to operator.
///
pub trait KroneckerProductAssignTo<Rhs = Self> {
    type Output;

    /// Performs the kronecker product assign to operation.
    ///
    fn kronecker_product_assign_to( self, rhs: Rhs, res: &mut Self::Output );
}
