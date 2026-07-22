// act1.rs — Burnlings · Chapter 7: Activations
//
// ReLU is `max(0, x)`: it flattens EVERY negative value to zero, giving
// [0, 0, 0, 1, 3] for the input below. Leaky ReLU is the lookalike that KEEPS a
// small slice of the negatives (slope 0.1 here → -0.2, -0.05, ...), so it is a
// different function and fails this test.
//
// TODO: Use plain ReLU (`activation::relu`), which zeroes all negatives.
//       (Compare book example chapter07/e01_the_relu_family.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Tensor, activation};

type Backend = NdArray;

fn rectify() -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([-2.0, -0.5, 0.0, 1.0, 3.0], &device);
    // ⬇️ leaky_relu keeps a slice of the negatives; relu zeroes them
    activation::leaky_relu(x, 0.1).into_data().to_vec().unwrap()
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
