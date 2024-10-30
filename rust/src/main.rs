use linear_algebra::tensor::{ Tensor, Error };
fn main() {
    let mut tensor = Tensor::<f64, 2>::new([2, 2]);
    tensor[[0, 0]] = 1.0;
    tensor[[0, 1]] = 2.0;
    tensor[[1, 0]] = 3.0;
    tensor[[1, 1]] = 4.0;
    let mut slice = tensor.slice([0, 0], [2, 2], [1, 1]);
    let tensor = slice.tensor().unwrap();
    assert_eq!(tensor[[0, 0]], 1.0);
    assert_eq!(tensor[[0, 1]], 2.0);
    assert_eq!(tensor[[1, 0]], 3.0);
    assert_eq!(tensor[[1, 1]], 4.0);
}