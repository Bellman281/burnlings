// kldiv1.rs — SOLUTION · Chapter 8: Losses
// KL divergence compares two distributions. It is asymmetric about its inputs:
// the PREDICTION must be LOG-probabilities (`log_softmax`), the TARGET must be
// plain probabilities (`softmax`). Getting the prediction's log wrong silently
// produces the wrong number.

use burn::backend::NdArray;
use burn::nn::loss::{KLDivLossConfig, Reduction};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Tensor, TensorData};

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let student_logits = Tensor::<Backend, 2>::from_data(
        TensorData::from([[2.0, 0.5, -1.0, 0.0], [0.1, 1.2, 0.3, -0.5], [-0.2, 0.4, 1.5, 0.8]]),
        &device,
    );
    let teacher_logits = Tensor::<Backend, 2>::from_data(
        TensorData::from([[2.5, 0.2, -0.8, 0.1], [-0.1, 1.6, 0.2, -0.3], [0.0, 0.7, 1.2, 0.4]]),
        &device,
    );
    let student = log_softmax(student_logits, 1); // prediction = LOG-probabilities
    let teacher = softmax(teacher_logits, 1); // target = probabilities
    KLDivLossConfig::new()
        .init()
        .forward(student, teacher, Reduction::Auto)
        .into_scalar()
}

fn main() { println!("KL = {}", loss()); }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kl_divergence() {
        assert!((loss() - 0.034823496).abs() < 1e-4, "loss = {}", loss());
    }
}
