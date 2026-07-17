// tensors7.rs — Burnlings · Chapter 1: Tensors
//
// `into_data` CONSUMES the tensor (it takes `self`, not `&self`). After you
// call it, the tensor is gone. The code reads the data out of `t` and then
// tries to print `t` — but `t` has already been moved.
//
// TODO: Let the tensor survive the data read. Clone it before consuming it
//       (`t.clone().into_data()`), or move the read to the very end.
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn sum_via_slice() -> f32 {
    let device = Default::default();
    let t = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device);

    // ⬇️ this MOVES `t` away
    let data = t.into_data();
    let values: &[f32] = data.as_slice().unwrap();
    let total: f32 = values.iter().sum();

    // ⬇️ ...so this use of `t` won't compile
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
