// ops2.rs — SOLUTION · Chapter 2: Tensor Ops
// Burn will not broadcast a rank-1 vector against a rank-2 matrix implicitly.
// Raise the vector to [1, 3] with `unsqueeze()`; then it broadcasts over rows.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn add_bias() -> Tensor<Backend, 2> {
    let device = Default::default();
    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let bias = Tensor::<Backend, 1>::from_floats([100.0, 200.0, 300.0], &device);
    m + bias.unsqueeze()
}

fn main() {
    println!("{}", add_bias().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn per_column_bias() {
        let out = add_bias();
        assert_eq!(out.dims(), [2, 3]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert_eq!(v, vec![101.0, 202.0, 303.0, 104.0, 205.0, 306.0]);
    }
}
