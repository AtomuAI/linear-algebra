// Copyright 2024 Bewusstsein Labs

use bewusstsein::memory::memory::heap::Heap;
use bewusstsein::memory::storage::owned::Owned;

use crate::tensor::Tensor;

#[test]
fn test() {
    let mut tensor = Tensor::<f32, 3, Heap, Owned>::new( [ 2, 3, 4 ] );
    assert_eq!( tensor.dim(), 3_usize );
    assert_eq!( tensor.size(), 24_usize );
    assert_eq!( tensor.shape(), [ 2_usize, 3_usize, 4_usize ] );
    tensor.fill( 1.0 );
    for k in 0..4 {
        for j in 0..3 {
            for i in 0..2 {
                tensor[ [ i, j, k ] ] = ( i * 3 * 4 + j * 4 + k ) as f32;
                print!( "{},", tensor[ [ i, j, k ] ] );
            }
            print!( "\n" );
        }
        print!( "\n" );
    }
    let slice = tensor.slice( [ 0, 0, 0 ], [ 2, 3, 4 ], [ 2, 1, 2 ] );
    assert_eq!( slice.dim(), 3_usize );
    assert_eq!( slice.size(), 6_usize );
    assert_eq!( slice.shape(), [ 1_usize, 3_usize, 2_usize ] );
    //slice.zero();
    for k in 0..2 {
        for j in 0..3 {
            for i in 0..1 {
                print!( "{},", slice[ [ i, j, k ] ] );
            }
            print!( "\n" );
        }
        print!( "\n" );
    }
    let tensor_2 = slice.tensor()?;
    for k in 0..2 {
        for j in 0..3 {
            for i in 0..1 {
                print!( "{},", tensor_2[ [ i, j, k ] ] );
            }
            print!( "\n" );
        }
        print!( "\n" );
    }
    Ok( () )
}