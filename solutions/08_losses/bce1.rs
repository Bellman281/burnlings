// bce1.rs — SOLUTION · Chapter 8: Losses
// Binary cross-entropy for yes/no outputs. `.with_logits(true)` tells Burn the
// inputs are RAW logits and applies the sigmoid internally (fused, stable). Feed
// it logits directly. On this data the loss is ~0.294.

use burn::backend::NdArray;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let logits = Tensor::<Backend, 1>::from_floats([2.0, -1.5, 0.3], &device);
    let targets = Tensor::<Backend, 1, Int>::from_ints([1, 0, 1], &device);
    BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init(&device)
        .forward(logits, targets)
        .into_scalar()
}

fn main() { println!("BCE = {}", loss()); }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bce_from_logits() {
        assert!((loss() - 0.29423222).abs() < 1e-4, "loss = {}", loss());
    }
}
