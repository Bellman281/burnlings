// bce1.rs — Burnlings · Chapter 8: Losses
//
// Binary cross-entropy for yes/no outputs. `.with_logits(true)` tells Burn the
// inputs are RAW logits and fuses the sigmoid inside (stable). Leave it off and
// the loss treats the raw logits as if they were probabilities — wrong, since
// logits aren't in [0, 1].
//
// TODO: Configure the loss for logits with `.with_logits(true)`.
//       (Compare book example chapter08/e04_binary_cross_entropy.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let logits = Tensor::<Backend, 1>::from_floats([2.0, -1.5, 0.3], &device);
    let targets = Tensor::<Backend, 1, Int>::from_ints([1, 0, 1], &device);
    // ⬇️ missing `.with_logits(true)` — the loss won't apply the sigmoid
    BinaryCrossEntropyLossConfig::new()
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
