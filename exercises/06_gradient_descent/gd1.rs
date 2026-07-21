// gd1.rs — Burnlings · Chapter 6: Gradient Descent
//
// Gradient DESCENT minimises the loss by walking DOWNHILL: each step moves
// AGAINST the gradient, so you SUBTRACT `gradient * learning_rate` from the
// weights. Add it instead and you climb uphill — the weights diverge instead of
// settling on the fit.
//
// The data is y = 2x + 1, so the weights [bias, slope] should converge to
// [1, 2]. The update below has the wrong sign, so it blows up and the test fails.
//
// TODO: Fix the update step so it steps AGAINST the gradient (subtract, not add).
//       (Compare book example ch6_01_gradient_descent.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn train() -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
        &device,
    );
    let y = Tensor::<Backend, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device);

    let mut w = Tensor::<Backend, 2>::zeros([2, 1], &device);
    let n = 4.0;
    let lr = 0.05;

    for _ in 0..3000 {
        let pred = x.clone().matmul(w.clone());
        let err = pred - y.clone();
        let grad = x.clone().transpose().matmul(err).mul_scalar(2.0 / n);
        // ⬇️ wrong sign: this climbs UPHILL and diverges
        w = w + grad.mul_scalar(lr);
    }
    w.into_data().to_vec().unwrap()
}

fn main() {
    println!("w = {:?}", train());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn converges_to_bias1_slope2() {
        let w = train();
        assert!((w[0] - 1.0).abs() < 1e-3, "bias = {}", w[0]);
        assert!((w[1] - 2.0).abs() < 1e-3, "slope = {}", w[1]);
    }
}
