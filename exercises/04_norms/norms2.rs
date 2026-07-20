// norms2.rs — Burnlings · Chapter 4: Norms & Gram
//
// Normalising a vector scales it to length 1 by dividing by its L2 norm.
// [3, 4] has length 5, so its unit vector is [0.6, 0.8].
//
// `unit` should return that unit vector. Right now it returns `x` unchanged
// (still length 5), so the test fails.
//
// TODO: Normalise `x` to unit length with
//       `linalg::vector_normalize(x, 2.0, 0, 1e-12)` (p=2, dim 0, tiny epsilon).
//       (Compare book example ch4_01_vector_norm.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn unit() -> Tensor<Backend, 1> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([3.0, 4.0], &device);
    // ⬇️ this is still the original vector, not normalised
    x
}

fn main() {
    println!("unit = {}", unit().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn has_unit_length() {
        let v: Vec<f32> = unit().into_data().to_vec().unwrap();
        let expected = [0.6, 0.8];
        for (g, w) in v.iter().zip(expected.iter()) {
            assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
        }
    }
}
