// tensors8.rs — SOLUTION · Chapter 1: Tensors
// Tensor arithmetic CONSUMES its operands. When a tensor is used more than
// once, clone it. This is min-max normalisation to [0, 1].

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn normalize() -> Tensor<Backend, 1> {
    let device = Default::default();
    let t = Tensor::<Backend, 1>::from_floats([2.0, 4.0, 6.0, 8.0], &device);

    let mn = t.clone().min();
    let mx = t.clone().max();
    (t - mn.clone()).div(mx - mn)
}

fn main() {
    println!("normalized = {}", normalize());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_to_unit_range() {
        let t = normalize();
        let d = t.into_data();
        let v: &[f32] = d.as_slice().unwrap();
        // (x - 2) / 6  ->  [0, 1/3, 2/3, 1]
        let expected = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }
}
