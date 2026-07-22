// ce1.rs — Burnlings · Chapter 8: Losses
//
// Cross-entropy applies softmax INTERNALLY, so it expects RAW logits. If you
// softmax the logits yourself first, the loss softmaxes them AGAIN — a
// double-softmax that flattens the distribution and gives the wrong number
// (with no compiler error to warn you).
//
// TODO: Pass the raw `logits` straight to `.forward(...)` — don't pre-softmax.
//       (Compare book example chapter08/e02_cross_entropy.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::activation::softmax;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let logits = Tensor::<Backend, 2>::from_floats([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], &device);
    let targets = Tensor::<Backend, 1, Int>::from_data(TensorData::from([0i64, 1]), &device);
    // ⬇️ pre-softmaxing causes a double-softmax inside the loss
    CrossEntropyLossConfig::new()
        .init(&device)
        .forward(softmax(logits, 1), targets)
        .into_scalar()
}

fn main() { println!("cross-entropy = {}", loss()); }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_entropy_of_raw_logits() {
        assert!((loss() - 0.31853974).abs() < 1e-4, "loss = {}", loss());
    }
}
