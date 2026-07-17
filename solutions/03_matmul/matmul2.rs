// matmul2.rs — SOLUTION · Chapter 3: Matmul & the Shape Rule
// A matrix times a vector is a matrix-vector product. `matmul` wants two rank-2
// tensors; for M·v use `linalg::matvec(matrix, vector)`.

use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn mat_vec() -> Tensor<Backend, 1> {
    let device = Default::default();
    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let v = Tensor::<Backend, 1>::from_floats([1.0, 1.0], &device);
    linalg::matvec(m, v)
}

fn main() {
    println!("{}", mat_vec().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_times_vector() {
        let out = mat_vec();
        assert_eq!(out.dims(), [2]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert_eq!(v, vec![3.0, 7.0]);
    }
}
