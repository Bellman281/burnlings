// matmul2.rs — Burnlings · Chapter 3: Matmul & the Shape Rule
//
// `matmul` multiplies two MATRICES (both rank 2). A matrix times a VECTOR is a
// different call: `linalg::matvec(matrix, vector)`. Passing a rank-1 vector to
// `matmul` doesn't type-check — the ranks don't line up.
//
// `mat_vec` should compute M·v = [3, 7]. As written it calls `m.matmul(v)` with
// a vector, so it does not compile.
//
// TODO: Compute the matrix-vector product with `linalg::matvec`.
//       (Compare book example chapter03/e02_transpose_matvec_outer.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn mat_vec() -> Tensor<Backend, 1> {
    let device = Default::default();
    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let v = Tensor::<Backend, 1>::from_floats([1.0, 1.0], &device);
    // ⬇️ matmul wants a rank-2 tensor, not a vector
    m.matmul(v)
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
