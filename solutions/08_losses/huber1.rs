// huber1.rs — SOLUTION · Chapter 8: Losses
// Huber loss is the robust cousin of MSE: quadratic for small errors, linear for
// large ones, so a single outlier can't dominate. With delta = 1.0 on this data
// it is 0.625 — smaller than the MSE (1.5), because the outlier is down-weighted.

use burn::backend::NdArray;
use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::tensor::Tensor;

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let pred = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let target = Tensor::<Backend, 2>::from_floats([[2.0, 1.0], [3.0, 2.0]], &device);
    HuberLossConfig::new(1.0)
        .init()
        .forward(pred, target, Reduction::Mean)
        .into_scalar()
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
