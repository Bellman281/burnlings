// gram1.rs — Burnlings · Chapter 4: Norms & Gram
//
// The Gram matrix of a design matrix X ([n_samples, n_features]) is X^T X:
// transpose FIRST, then multiply. Order matters — `X X^T` and `X^T X` are
// different shapes. For a [3, 2] matrix, X^T X is [2, 2] but X X^T is [3, 3].
//
// `gram` should return X^T X (shape [2, 2], value [[3, 6], [6, 14]]). The body
// multiplies the other way around, so the shape is wrong and the test fails.
//
// TODO: Multiply the transpose by X (in that order): X^T X.
//       (Compare book example ch4_02_gram_matrix.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn gram() -> Tensor<Backend, 2> {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], &device);
    let xt = x.clone().transpose(); // [2, 3]
    // ⬇️ this is X X^T (shape [3, 3]); the Gram matrix is X^T X
    x.matmul(xt)
}

fn main() {
    println!("X^T X =\n{}", gram().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gram_is_2x2() {
        let g = gram();
        assert_eq!(g.dims(), [2, 2]);
        let v: Vec<f32> = g.into_data().to_vec().unwrap();
        assert_eq!(v, vec![3.0, 6.0, 6.0, 14.0]);
    }
}
