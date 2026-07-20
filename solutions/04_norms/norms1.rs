// norms1.rs — SOLUTION · Chapter 4: Norms & Gram
// The L2 (Euclidean) norm is a vector's length: sqrt(sum of squares).
// For [3, 4] that's sqrt(9 + 16) = 5 — not the plain sum (7).

use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn magnitude() -> f32 {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([3.0, 4.0], &device);
    linalg::l2_norm(x, 0).into_scalar()
}

fn main() {
    println!("|x| = {}", magnitude());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn euclidean_length() {
        assert!((magnitude() - 5.0).abs() < 1e-6);
    }
}
