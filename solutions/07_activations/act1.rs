// act1.rs — SOLUTION · Chapter 7: Activations
// ReLU = max(0, x): it flattens EVERY negative value to zero. (Leaky ReLU keeps
// a small slope for negatives; ReLU does not.)

use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type Backend = NdArray;

fn rectify() -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([-2.0, -0.5, 0.0, 1.0, 3.0], &device);
    activation::relu(x).into_data().to_vec().unwrap()
}

fn main() {
    println!("relu = {:?}", rectify());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zeros_all_negatives() {
        assert_eq!(rectify(), vec![0.0, 0.0, 0.0, 1.0, 3.0]);
    }
}
