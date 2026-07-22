// net1.rs — Burnlings · Chapter 11: A Network From Scratch
//
// A "layer" is just an affine map (x @ W) plus a nonlinearity. This tiny net has
// two layers: h = relu(x @ W1); out = h @ W2. The body stops at the hidden layer
// and returns `h` (shape [2, 3]) instead of the network output (shape [2, 1]).
//
// TODO: Apply the output layer — pass `h` through `W2` (`h.matmul(w2)`).
//       (Compare book example chapter11/e01_a_layer_is_a_matmul.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::activation::relu;
use burn::tensor::Tensor;

type Backend = NdArray;

fn forward() -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let w1 = Tensor::<Backend, 2>::from_floats([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], &device);
    let w2 = Tensor::<Backend, 2>::from_floats([[0.7], [0.8], [0.9]], &device);

    let h = relu(x.matmul(w1)); // hidden layer  [2, 3]
    // ⬇️ this returns the hidden activations, not the network output
    let _ = &w2;
    h.into_data().to_vec().unwrap()
}

fn main() {
    println!("out = {:?}", forward());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_layer_forward() {
        let out = forward();
        assert_eq!(out.len(), 2, "output should be [2, 1] = 2 values");
        assert!((out[0] - 2.94).abs() < 1e-3, "out0 = {}", out[0]);
        assert!((out[1] - 6.38).abs() < 1e-3, "out1 = {}", out[1]);
    }
}
