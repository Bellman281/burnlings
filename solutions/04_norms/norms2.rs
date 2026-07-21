// norms2.rs — SOLUTION · Chapter 4: Norms & Gram
// Normalising a vector scales it to length 1 by dividing by its L2 norm.
// [3, 4] has length 5, so the unit vector is [0.6, 0.8].

use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn unit() -> Tensor<Backend, 1> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([3.0, 4.0], &device);
    // p=2 (L2), over dim 0, tiny epsilon to avoid divide-by-zero
    linalg::vector_normalize(x, 2.0, 0, 1e-12)
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
