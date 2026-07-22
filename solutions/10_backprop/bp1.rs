// bp1.rs — SOLUTION · Chapter 10: Backprop
// Backprop IS the chain rule run backwards. loss = (a*b)^2 with a=2, b=3.
// dloss/da = 2(a*b)*b = 2*6*3 = 36; dloss/db = 2(a*b)*a = 2*6*2 = 24.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn grads() -> (f32, f32) {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([2.0], &device).require_grad();
    let b = Tensor::<Backend, 1>::from_floats([3.0], &device).require_grad();
    let u = a.clone() * b.clone(); // u = a*b = 6
    let v = u.clone() * u.clone(); // v = u^2 = 36  (this square is the chain-rule twist)
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
