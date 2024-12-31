
/// The contraction operator.
///
pub trait Contract<Rhs = Self> {
    type Output;

    fn contract( self, rhs: Rhs ) -> Self::Output;
}

/// The contraction assign to operator.
///
pub trait ContractAssignTo<Rhs = Self> {
    type Output;

    fn contract_assign_to( self, rhs: Rhs, res: &mut Self::Output );
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

/// The tensor (kronecker) product operator.
///
pub trait TensorProduct<Rhs = Self> {
    type Output;

    /// Performs the outer product operation.
    ///
    fn tensor_product( self, rhs: Rhs ) -> Self::Output;
}

/// The tensor (kronecker) product assign to operator.
///
pub trait TensorProductAssignTo<Rhs = Self> {
    type Output;

    /// Performs the outer product assign to operation.
    ///
    fn tensor_product_assign_to( self, rhs: Rhs, res: &mut Self::Output );
}
