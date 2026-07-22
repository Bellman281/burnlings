// cosine1.rs — Burnlings · Chapter 8: Losses
//
// Cosine embedding loss compares DIRECTION. The target y says what each pair
// should be: +1 = similar, -1 = dissimilar. Here pair 1 (`[1,0]` vs `[0.9,0.1]`)
// nearly matches and pair 2 (`[1,0]` vs `[-1,0]`) is opposite — so the correct
// targets are [1, -1]. Flip them and you penalise the wrong pairs.
//
// TODO: Set the targets to [1, -1] (pair 1 similar, pair 2 dissimilar).
//       (Compare book example chapter08/e06_cosine_embedding.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::nn::loss::CosineEmbeddingLossConfig;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn loss() -> f32 {
    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [1.0, 0.0]], &device);
    let b = Tensor::<Backend, 2>::from_floats([[0.9, 0.1], [-1.0, 0.0]], &device);
    // ⬇️ targets are flipped
    let y = Tensor::<Backend, 1, Int>::from_ints([-1, 1], &device);
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
