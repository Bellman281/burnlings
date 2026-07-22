// act2.rs — Burnlings · Chapter 7: Activations
//
// Sigmoid squashes to (0, 1) and gives exactly 0.5 at x = 0. Tanh is the other
// saturating curve — it squashes to (-1, 1) and gives 0 at x = 0. They are not
// interchangeable: the body uses `tanh`, so the midpoint comes out 0 instead of
// 0.5 and the test fails.
//
// TODO: Use sigmoid. Note there is no `.sigmoid()` method — it lives only in
//       `activation::sigmoid(x)`. (Compare book example chapter07/e02_sigmoid_and_tanh.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type Backend = NdArray;

fn squash() -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([-1.0, 0.0, 1.0], &device);
    // ⬇️ tanh gives 0 at x=0, not the 0.5 that sigmoid does
    x.tanh().into_data().to_vec().unwrap()
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
