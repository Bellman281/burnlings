// bp2.rs — Burnlings · Chapter 10: Backprop
//
// Gradients flow THROUGH the hidden layer back to the first weight matrix — but
// only if that matrix opts in with `.require_grad()`. Here `w1` doesn't, so no
// gradient reaches it: `w1.grad(&g)` is `None` and the `.unwrap()` panics.
//
// TODO: Mark `w1` with `.require_grad()` so the backprop'd gradient reaches it.
//       (Compare book example chapter10/e02_backprop_through_a_hidden_layer.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::tensor::activation::relu;
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn grads() -> (Vec<f32>, Vec<f32>) {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0]], &device);
    let target = Tensor::<Backend, 2>::from_floats([[1.0]], &device);
    // ⬇️ missing `.require_grad()` — the first layer won't receive a gradient
    let w1 = Tensor::<Backend, 2>::from_floats([[0.1, 0.2], [0.3, 0.4]], &device);
    let w2 = Tensor::<Backend, 2>::from_floats([[0.5], [0.6]], &device).require_grad();

    let h = relu(x.matmul(w1.clone()));
    let out = h.matmul(w2.clone());
    let diff = out - target;
    let loss = (diff.clone() * diff).sum();

    let g = loss.backward();
    let gw1 = w1.grad(&g).unwrap();
    let gw2 = w2.grad(&g).unwrap();
    (
        gw1.into_data().to_vec().unwrap(),
        gw2.into_data().to_vec().unwrap(),
    )
}

fn main() {
    let (gw1, gw2) = grads();
    println!("gW1 = {gw1:?}\ngW2 = {gw2:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gradient_reaches_first_layer() {
        let (gw1, gw2) = grads();
        let e1 = [-0.05, -0.06, -0.10, -0.12];
        let e2 = [-0.07, -0.10];
        for (g, w) in gw1.iter().zip(e1.iter()) { assert!((g - w).abs() < 1e-3, "gW1 {g} vs {w}"); }
        for (g, w) in gw2.iter().zip(e2.iter()) { assert!((g - w).abs() < 1e-3, "gW2 {g} vs {w}"); }
    }
}
