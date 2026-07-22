// huber1.rs — Burnlings · Chapter 8: Losses
//
// Huber loss is the robust cousin of MSE: quadratic for small errors, linear for
// large ones, so an outlier can't dominate. On this data Huber (delta = 1.0) is
// 0.625, while plain MSE is 1.5 — the body uses MSE, so the test fails.
//
// TODO: Use Huber loss (`HuberLossConfig::new(1.0)`), not MSE.
//       (Compare book example chapter08/e03_huber_loss.)
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
    // ⬇️ this is MSE (1.5); we want the robust Huber loss (0.625)
    MseLoss::new().forward(pred, target, Reduction::Mean).into_scalar()
}

fn main() { println!("Huber = {}", loss()); }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn huber_delta_one() {
        assert!((loss() - 0.625).abs() < 1e-4, "loss = {}", loss());
    }
}
