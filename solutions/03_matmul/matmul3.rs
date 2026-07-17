// matmul3.rs — SOLUTION · Chapter 3: Matmul & the Shape Rule
// Batched matmul multiplies matching STACKS of matrices: the leading dim is the
// batch, the last two are the matrices. [b, m, k] @ [b, k, n] -> [b, m, n], and
// the inner dims (k) must match.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn batched() -> Tensor<Backend, 3> {
    let device = Default::default();
    let a = Tensor::<Backend, 3>::ones([4, 2, 3], &device);
    let b = Tensor::<Backend, 3>::ones([4, 3, 5], &device);
    a.matmul(b)
}

fn main() {
    println!("dims = {:?}", batched().dims());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_shape_and_values() {
        let out = batched();
        assert_eq!(out.dims(), [4, 2, 5]);
        // each output element sums k=3 products of ones -> 3.0
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| (*x - 3.0).abs() < 1e-6));
    }
}
