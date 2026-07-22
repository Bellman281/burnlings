// ce1.rs — SOLUTION · Chapter 8: Losses
// Cross-entropy for classification. It applies softmax INTERNALLY, so you pass
// RAW logits — softmaxing them yourself first causes a double-softmax that
// flattens the distribution and gives the wrong loss. Targets are class indices.

use burn::backend::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let logits = Tensor::<Backend, 2>::from_floats([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], &device);
    let targets = Tensor::<Backend, 1, Int>::from_data(TensorData::from([0i64, 1]), &device);
    CrossEntropyLossConfig::new()
        .init(&device)
        .forward(logits, targets)
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
