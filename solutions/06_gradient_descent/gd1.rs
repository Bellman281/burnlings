// gd1.rs — SOLUTION · Chapter 6: Gradient Descent
// Fit y = bias + slope*x by hand. Gradient DESCENT walks downhill: subtract the
// gradient times the learning rate. The data comes from y = 2x + 1, so the
// weights [bias, slope] converge to [1, 2].

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
        w = w - grad.mul_scalar(lr); // descent: step against the gradient
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
