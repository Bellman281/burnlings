// tensors7.rs — SOLUTION · Chapter 1: Tensors
// `into_data` CONSUMES the tensor. If you still need the tensor afterwards,
// clone it first (`t.clone().into_data()`), or read the data last.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn sum_via_slice() -> f32 {
    let device = Default::default();
    let t = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device);

    // clone so `t` survives for the println below
    let data = t.clone().into_data();
    let values: &[f32] = data.as_slice().unwrap();
    let total: f32 = values.iter().sum();

    println!("tensor was still usable here: {t}");
    total
}

fn main() {
    println!("sum = {}", sum_via_slice());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sums_to_60() {
        assert_eq!(sum_via_slice(), 60.0);
    }
}
