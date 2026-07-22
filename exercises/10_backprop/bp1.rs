// bp1.rs — Burnlings · Chapter 10: Backprop
//
// Backprop IS the chain rule run backwards. Here loss = (a*b)^2 with a=2, b=3.
// The square matters: dloss/da = 2(a*b)*b = 36 and dloss/db = 2(a*b)*a = 24.
// Without squaring, loss = a*b and the gradients are just b=3 and a=2.
//
// TODO: Square `u` so that loss = (a*b)^2 (set `v = u * u`).
//       (Compare book example chapter10/e01_the_chain_rule_checked_by_hand.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn grads() -> (f32, f32) {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([2.0], &device).require_grad();
    let b = Tensor::<Backend, 1>::from_floats([3.0], &device).require_grad();
    let u = a.clone() * b.clone();
    // ⬇️ not squared: this makes loss = a*b, so the gradients are wrong
    let v = u.clone();
    let loss = v.sum();
    let g = loss.backward();
    (
        a.grad(&g).unwrap().into_scalar(),
        b.grad(&g).unwrap().into_scalar(),
    )
}

fn main() {
    let (da, db) = grads();
    println!("da = {da}, db = {db}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chain_rule() {
        let (da, db) = grads();
        assert!((da - 36.0).abs() < 1e-3, "da = {da}");
        assert!((db - 24.0).abs() < 1e-3, "db = {db}");
    }
}
