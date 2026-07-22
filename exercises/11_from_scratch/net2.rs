// net2.rs — Burnlings · Chapter 11: A Network From Scratch
//
// Backprop by hand should match autodiff exactly. A weight gradient is the
// INPUT (transposed) matmul the incoming error: dW1 = xᵀ @ dh_pre. Forgetting the
// transpose (using `x` instead of `xᵀ`) still type-checks here but computes the
// wrong gradient, so it stops matching autodiff.
//
// TODO: Transpose `x` in the layer-1 weight gradient (`x.transpose().matmul(...)`).
//       (Compare book example chapter11/e02_gradients_by_hand_and_by_autodiff.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::tensor::activation::relu;
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn manual_vs_autodiff_gap() -> f32 {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let t = Tensor::<Backend, 2>::from_floats([[1.0], [0.0]], &device);
    let n = x.dims()[0] as f32;

    let w1 = Tensor::<Backend, 2>::from_floats([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], &device).require_grad();
    let w2 = Tensor::<Backend, 2>::from_floats([[0.7], [0.8], [0.9]], &device).require_grad();

    let h_pre = x.clone().matmul(w1.clone());
    let h = relu(h_pre.clone());
    let out = h.clone().matmul(w2.clone());
    let diff = out - t;
    let loss = (diff.clone() * diff.clone()).mean();

    let dout = diff.mul_scalar(2.0 / n);
    let manual_gw1 = {
        let dh = dout.clone().matmul(w2.clone().transpose());
        let dh_pre = dh.mask_fill(h_pre.lower_equal_elem(0.0), 0.0);
        // ⬇️ missing the transpose: the weight gradient needs xᵀ, not x
        x.clone().matmul(dh_pre)
    };

    let g = loss.backward();
    let auto_gw1 = w1.grad(&g).unwrap();

    (manual_gw1.inner() - auto_gw1).abs().max().into_scalar()
}

fn main() {
    println!("max |manual - auto| = {}", manual_vs_autodiff_gap());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hand_gradients_match_autodiff() {
        let gap = manual_vs_autodiff_gap();
        assert!(gap < 1e-4, "manual backprop disagrees with autodiff by {gap}");
    }
}
