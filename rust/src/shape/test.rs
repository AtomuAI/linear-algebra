// Copyright 2024 Bewusstsein Labs

use bewusstsein::memory::{ Memory, heap::Heap, stack::Stack };

use crate::shape::Shape;

#[test]
fn stack_test() {
    let shape = Shape::<Stack<usize, 9>>::from( [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ] );
    assert_eq!( shape.dim(), 9 );
    assert_eq!( shape.vol(), 0 * 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 );
    assert_eq!( shape[ 0 ], 0 );
    assert_eq!( shape[ 1 ], 1 );
    assert_eq!( shape[ 2 ], 2 );
    assert_eq!( shape[ 3 ], 3 );
    assert_eq!( shape[ 4 ], 4 );
    assert_eq!( shape[ 5 ], 5 );
    assert_eq!( shape[ 6 ], 6 );
    assert_eq!( shape[ 7 ], 7 );
    assert_eq!( shape[ 8 ], 8 );
}