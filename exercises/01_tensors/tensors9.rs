// tensors9.rs — Burnlings · Chapter 1: Tensors
//
// Floats are rarely EXACTLY equal, so you don't test tensors with `==`. You
// take the largest element-wise difference and compare it to a tolerance —
// exactly what the book's `check_closeness` helper reports.
//
// `within_tolerance` should return `true` when the biggest difference is
// SMALLER than `tol`. The comparison below is backwards, so it returns false
// and the test fails.
//
// TODO: Fix the comparison so "close enough" means the difference is below the
//       tolerance. (Compare book example ch1_09_display_and_closeness.rs.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn within_tolerance() -> bool {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([1.001, 1.999, 3.002], &device);
    let tol = 0.01;

    let max_diff = (a - b).abs().max().into_scalar();

    // ⬇️ backwards: this is true only when they are FAR apart
    max_diff > tol
}

fn main() {
    println!("within tolerance? {}", within_tolerance());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensors_are_close() {
        assert!(within_tolerance());
    }
}
