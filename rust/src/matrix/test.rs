// Copyright 2024 Bewusstsein Labs

use crate::tensor::{ Error };
use crate::matrix::Matrix3x3;

#[test]
fn test() -> Result<(), Error>{
    let mut matrix = Matrix3x3::<f32>::new( [ 2, 3 ] );
    assert_eq!( matrix.dim(), 2_usize );
    assert_eq!( matrix.size(), 6_usize );
    assert_eq!( matrix.shape(), [ 2_usize, 3_usize ] );
    matrix.fill( 1.0 );
    for j in 0..3 {
        for i in 0..2 {
            matrix[ [ i, j ] ] = ( i * 3 + j ) as f32;
            print!( "{},", matrix[ [ i, j ] ] );
        }
        print!( "\n" );
    }
    let slice = matrix.slice( [ 0, 0 ], [ 2, 3 ], [ 2, 1 ] );
    assert_eq!( slice.dim(), 2_usize );
    assert_eq!( slice.size(), 3_usize );
    assert_eq!( slice.shape(), [ 1_usize, 3_usize ] );
    //slice.zero();
    for j in 0..3 {
        for i in 0..1 {
            print!( "{},", slice[ [ i, j ] ] );
        }
        print!( "\n" );
    }
    let matrix_2 = slice.tensor()?;
    for j in 0..3 {
        for i in 0..1 {
            print!( "{},", matrix_2[ [ i, j ] ] );
        }
        print!( "\n" );
    }
    Ok( () )
}