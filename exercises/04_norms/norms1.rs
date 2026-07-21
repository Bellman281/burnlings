// norms1.rs — Burnlings · Chapter 4: Norms & Gram
//
// The L2 (Euclidean) norm is a vector's LENGTH: the square root of the sum of
// squares. For [3, 4] that's sqrt(9 + 16) = 5. That is NOT the same as the
// plain sum of the elements (3 + 4 = 7).
//
// `magnitude` should return the L2 norm of [3, 4], i.e. 5.0. The body returns
// the sum instead, so the test fails.
//
// TODO: Return the L2 norm instead of the sum, using `linalg::l2_norm(x, 0)`.
//       (Compare book example ch4_01_vector_norm.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Tensor, linalg};

type Backend = NdArray;

fn magnitude() -> f32 {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([3.0, 4.0], &device);
    // ⬇️ the sum is 7, not the length 5
    x.sum().into_scalar()
}

fn main() {
    println!("|x| = {}", magnitude());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn euclidean_length() {
        assert!((magnitude() - 5.0).abs() < 1e-6);
    }
}
