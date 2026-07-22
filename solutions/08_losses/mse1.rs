// mse1.rs — SOLUTION · Chapter 8: Losses
// Mean Squared Error with `Reduction::Mean` averages the squared errors.
// Squared errors here are [1, 1, 0, 4]; their mean is 1.5 (their sum is 6).

use burn::backend::NdArray;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::Tensor;

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let pred = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let target = Tensor::<Backend, 2>::from_floats([[2.0, 1.0], [3.0, 2.0]], &device);
    MseLoss::new().forward(pred, target, Reduction::Mean).into_scalar()
}

fn main() { println!("MSE = {}", loss()); }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mean_squared_error() {
        assert!((loss() - 1.5).abs() < 1e-4, "loss = {}", loss());
    }
}
