// net3.rs — SOLUTION · Chapter 11: A Network From Scratch
// The whole thing from scratch: four weight tensors, autodiff for gradients, and
// a MANUAL SGD step — param <- param - lr * grad (descent SUBTRACTS the gradient).
// `inner()` drops to the base backend for the arithmetic; `from_inner(..)
// .require_grad()` lifts the result back to a trainable leaf. Learns XOR. (Seeded.)

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

type B = Autodiff<NdArray>;

fn final_loss() -> f32 {
    let device = Default::default();
    B::seed(&device, 1);
    let x = Tensor::<B, 2>::from_floats([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], &device);
    let y = Tensor::<B, 2>::from_floats([[0.0], [1.0], [1.0], [0.0]], &device);
    let hidden = 8;
    let mut w1 = Tensor::<B, 2>::random([2, hidden], Distribution::Uniform(-1.0, 1.0), &device).require_grad();
    let mut b1 = Tensor::<B, 2>::zeros([1, hidden], &device).require_grad();
    let mut w2 = Tensor::<B, 2>::random([hidden, 1], Distribution::Uniform(-1.0, 1.0), &device).require_grad();
    let mut b2 = Tensor::<B, 2>::zeros([1, 1], &device).require_grad();
    let lr = 0.1;
    for _ in 0..20000 {
        let h = (x.clone().matmul(w1.clone()) + b1.clone()).tanh();
        let out = h.matmul(w2.clone()) + b2.clone();
        let diff = out - y.clone();
        let loss = (diff.clone() * diff).mean();
        let g = loss.backward();
        let g1 = w1.grad(&g).unwrap();
        let gb1 = b1.grad(&g).unwrap();
        let g2 = w2.grad(&g).unwrap();
        let gb2 = b2.grad(&g).unwrap();
        // manual SGD: subtract lr * grad
        w1 = Tensor::from_inner(w1.inner() - g1.mul_scalar(lr)).require_grad();
        b1 = Tensor::from_inner(b1.inner() - gb1.mul_scalar(lr)).require_grad();
        w2 = Tensor::from_inner(w2.inner() - g2.mul_scalar(lr)).require_grad();
        b2 = Tensor::from_inner(b2.inner() - gb2.mul_scalar(lr)).require_grad();
    }
    let h = (x.clone().matmul(w1.clone()) + b1.clone()).tanh();
    let out = h.matmul(w2.clone()) + b2.clone();
    (out - y).powf_scalar(2.0).mean().into_scalar()
}

fn main() {
    println!("final XOR loss = {}", final_loss());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learns_xor_from_scratch() {
        let l = final_loss();
        assert!(l < 0.05, "final loss = {l}");
    }
}
