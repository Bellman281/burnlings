// tensors9.rs — SOLUTION · Chapter 1: Tensors
// Floats are rarely exactly equal. You compare tensors by checking the largest
// element-wise difference against a tolerance (this is what check_closeness does).

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn within_tolerance() -> bool {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([1.001, 1.999, 3.002], &device);
    let tol = 0.01;

    let max_diff = (a - b).abs().max().into_scalar();

    // close ENOUGH means the biggest difference is below the tolerance
    max_diff < tol
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
