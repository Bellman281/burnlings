// mse1.rs — Burnlings · Chapter 8: Losses
//
// Mean Squared Error averages the squared errors when you ask for
// `Reduction::Mean`. `Reduction::Sum` adds them up instead — a different number.
// The squared errors here are [1, 1, 0, 4]: their MEAN is 1.5, their SUM is 6.
//
// TODO: Use `Reduction::Mean` so you get the *mean* squared error (1.5).
//       (Compare book example chapter08/e01_mean_squared_error.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::Tensor;

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let pred = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let target = Tensor::<Backend, 2>::from_floats([[2.0, 1.0], [3.0, 2.0]], &device);
    // ⬇️ Sum gives 6; the MEAN squared error is 1.5
    MseLoss::new().forward(pred, target, Reduction::Sum).into_scalar()
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
