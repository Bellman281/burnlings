// act2.rs — SOLUTION · Chapter 7: Activations
// Sigmoid squashes to (0, 1) and gives exactly 0.5 at x = 0. (Tanh squashes to
// (-1, 1) and gives 0 at x = 0 — a different curve.) Note: sigmoid lives only in
// `activation::sigmoid` — there is no `.sigmoid()` method on a tensor.

use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type Backend = NdArray;

fn squash() -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([-1.0, 0.0, 1.0], &device);
    activation::sigmoid(x).into_data().to_vec().unwrap()
}

fn main() {
    println!("sigmoid = {:?}", squash());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sigmoid_midpoint_is_half() {
        let v = squash();
        let expected = [0.26894143, 0.5, 0.7310586];
        for (g, w) in v.iter().zip(expected.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }
}
