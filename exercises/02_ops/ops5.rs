// ops5.rs — Burnlings · Chapter 2: Tensor Ops
//
// Z-score standardisation subtracts the per-feature mean and divides by the
// per-feature standard DEVIATION. Standard deviation is the SQUARE ROOT of the
// variance — dividing by the raw variance is a classic bug.
//
// Here var = 4 and std = 2 per column, so dividing by the variance makes every
// z half of what it should be, and the test fails.
//
// TODO: Divide by the standard deviation, not the variance. (`var` gives the
//       variance; take its square root.) Compare chapter02/e05_standardize_features.
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn standardize() -> Tensor<Backend, 2> {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let mean = x.clone().mean_dim(0);
    // ⬇️ this is the variance; standardisation needs the standard deviation
    let std = x.clone().var(0);
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
