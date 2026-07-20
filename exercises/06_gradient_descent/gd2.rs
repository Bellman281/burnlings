// gd2.rs — Burnlings · Chapter 6: Gradient Descent
//
// Same regression, but autodiff computes the gradient for us. The loss is the
// MEAN squared error — the MEAN of the squared residuals, not their sum. Using
// the sum scales both the loss and the gradient by n (here 4), so the numbers
// come out wrong.
//
// At w = 0 the MSE is mean([9, 25, 49, 81]) = 41 and the gradient is [-12, -35].
// The body sums instead of averaging, so the test fails.
//
// TODO: Use the MEAN of the squared residuals, not the sum.
//       (Compare book example ch6_02_autodiff_gradient.)
//
// I AM NOT DONE

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
    // ⬇️ this is the SUM of squared errors; MSE needs the mean
    let loss = (diff.clone() * diff).sum();

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
