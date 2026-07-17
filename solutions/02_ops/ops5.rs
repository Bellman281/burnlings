// ops5.rs — SOLUTION · Chapter 2: Tensor Ops
// Z-score standardisation: subtract the per-feature mean, divide by the
// per-feature standard DEVIATION (the square root of the variance).

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn standardize() -> Tensor<Backend, 2> {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let mean = x.clone().mean_dim(0);
    let std = x.clone().var(0).sqrt();
    (x - mean) / std
}

fn main() {
    println!("{}", standardize().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_mean_unit_std() {
        let v: Vec<f32> = standardize().into_data().to_vec().unwrap();
        let expected = [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0];
        for (g, w) in v.iter().zip(expected.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }
}
