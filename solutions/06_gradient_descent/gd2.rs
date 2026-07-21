// gd2.rs — SOLUTION · Chapter 6: Gradient Descent
// Same regression, but let autodiff compute the gradient. The loss is the MEAN
// squared error. At w = 0 the loss is mean([9,25,49,81]) = 41 and the gradient
// is [-12, -35].

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn mse_and_grad() -> (f32, Vec<f32>) {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
        &device,
    );
    let y = Tensor::<Backend, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device);

    let w = Tensor::<Backend, 2>::zeros([2, 1], &device).require_grad();

    let pred = x.matmul(w.clone());
    let diff = pred - y;
    let loss = (diff.clone() * diff).mean(); // MEAN squared error

    let grads = loss.backward();
    let gw = w.grad(&grads).unwrap();

    (loss.into_scalar(), gw.into_data().to_vec().unwrap())
}

fn main() {
    let (loss, g) = mse_and_grad();
    println!("loss = {loss}, grad = {g:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mse_and_its_gradient() {
        let (loss, g) = mse_and_grad();
        assert!((loss - 41.0).abs() < 1e-4, "loss = {loss}");
        assert!((g[0] - (-12.0)).abs() < 1e-4, "g0 = {}", g[0]);
        assert!((g[1] - (-35.0)).abs() < 1e-4, "g1 = {}", g[1]);
    }
}
