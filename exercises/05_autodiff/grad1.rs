// grad1.rs — Burnlings · Chapter 5: Autodiff
//
// `Autodiff<NdArray>` is a backend DECORATOR: wrap a backend and it can track a
// computation graph. But it only tracks the inputs you explicitly opt in with
// `.require_grad()`. Without that, there is no gradient to look up — `x.grad(..)`
// returns `None`, and the `.unwrap()` below panics.
//
// For f = sum(x^2), the gradient df/dx is 2x, i.e. [2, 4, 6].
//
// TODO: Mark `x` as requiring a gradient with `.require_grad()`.
//       (Compare book example ch5_01_backward_gradient.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn gradient() -> Vec<f32> {
    let device = Default::default();
    // ⬇️ no `.require_grad()` — nothing gets tracked
    let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
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
