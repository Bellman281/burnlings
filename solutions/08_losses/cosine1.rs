// cosine1.rs — SOLUTION · Chapter 8: Losses
// Cosine embedding loss compares direction, not magnitude. The target y says
// what each pair SHOULD be: +1 = similar (point the same way), -1 = dissimilar.
// With the correct targets here the loss is ~0.003.

use burn::backend::NdArray;
use burn::nn::loss::CosineEmbeddingLossConfig;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [1.0, 0.0]], &device);
    let b = Tensor::<Backend, 2>::from_floats([[0.9, 0.1], [-1.0, 0.0]], &device);
    // pair 1 should match (+1), pair 2 should differ (-1)
    let y = Tensor::<Backend, 1, Int>::from_ints([1, -1], &device);
    CosineEmbeddingLossConfig::new().init().forward(a, b, y).into_scalar()
}

fn main() { println!("cosine = {}", loss()); }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cosine_embedding() {
        assert!((loss() - 0.0030581355).abs() < 1e-4, "loss = {}", loss());
    }
}
