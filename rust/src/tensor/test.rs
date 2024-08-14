// Copyright 2024 Bewusstsein Labs

use crate::tensor::{ Tensor, Error };

#[test]
fn test() {
    let mut tensor = Tensor::<f32, 3>::new( [ 2, 3, 4 ] );
    assert_eq!( tensor.dim(), 3 );
    assert_eq!( tensor.size(), 24 );
    assert_eq!( tensor.shape(), [ 2, 3, 4 ] );
    tensor.fill( 1.0 );
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                println!( "tensor[ {}, {}, {} ] = {}", i, j, k, tensor[ [ i, j, k ] ] );
            }
        }
    }
}