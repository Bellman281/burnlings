// grad1.rs — SOLUTION · Chapter 5: Autodiff
// `Autodiff<NdArray>` wraps a backend so it can track a graph. Mark the inputs
// you want gradients for with `.require_grad()`, call `.backward()` on the
// output, then look up each gradient. For f = sum(x^2), df/dx = 2x.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn gradient() -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let f = (x.clone() * x.clone()).sum();
    let grads = f.backward();
    let dx = x.grad(&grads).unwrap();
    dx.into_data().to_vec().unwrap()
}

fn main() {
    println!("dx = {:?}", gradient());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn derivative_is_2x() {
        assert_eq!(gradient(), vec![2.0, 4.0, 6.0]);
    }
}
