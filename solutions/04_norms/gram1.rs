// gram1.rs — SOLUTION · Chapter 4: Norms & Gram
// The Gram matrix of a design matrix X ([n_samples, n_features]) is X^T X: you
// transpose FIRST, then multiply. Order matters — X X^T is a different shape.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn gram() -> Tensor<Backend, 2> {
    let device = Default::default();
    // X: 3 samples, 2 features
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], &device);
    let xt = x.clone().transpose(); // [2, 3]
    xt.matmul(x) // X^T X -> [2, 2]
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
