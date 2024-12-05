// Copyright 2024 Bewusstsein Labs

use crate::vector::Vector;

#[test]
fn test() {
    let mut vector = Vector::<f32, 2>::new( [ 2 ] );
    assert_eq!( vector.dim(), 1_usize );
    assert_eq!( vector.size(), 2_usize );
    assert_eq!( vector.shape(), [ 2_usize ] );
    vector.fill( 1.0 );
    for i in 0..2 {
        vector[ [ i ] ] = ( i ) as f32;
        print!( "{},", vector[ [ i ] ] );
    }
    let slice = vector.slice( [ 0 ], [ 2 ], [ 2 ] );
    assert_eq!( slice.dim(), 1_usize );
    assert_eq!( slice.size(), 1_usize );
    assert_eq!( slice.shape(), [ 1_usize ] );
    //slice.zero();
    for i in 0..1 {
        print!( "{},", slice[ [ i ] ] );
    }
    let vector_2 = slice.tensor()?;
    for i in 0..1 {
        print!( "{},", vector_2[ [ i ] ] );
    }
    Ok( () )
}