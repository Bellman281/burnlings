// tensors8.rs — Burnlings · Chapter 1: Tensors
//
// Tensor arithmetic CONSUMES its operands: `a - b` moves both `a` and `b`.
// So does `min`/`max` (they take `self`). When you need a tensor more than
// once, clone it. This function min-max normalises to [0, 1].
//
// The body uses `t` three times and `mn` twice with no clones, so it won't
// compile.
//
// TODO: Add `.clone()` wherever a tensor is reused. (Compare book example
//       ch1_08_ownership_cloning.rs.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn normalize() -> Tensor<Backend, 1> {
    let device = Default::default();
    let t = Tensor::<Backend, 1>::from_floats([2.0, 4.0, 6.0, 8.0], &device);

    // ⬇️ every one of these moves a tensor that is needed again
    let mn = t.min();
    let mx = t.max();
    (t - mn).div(mx - mn)
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
