// ops6.rs — SOLUTION · Chapter 2: Tensor Ops
// A comparison builds a boolean mask; `mask_where` keeps the original where the
// mask is FALSE and takes the replacement where the mask is TRUE.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn replace_positives() -> Tensor<Backend, 1> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([-1.0, 0.0, 2.0], &device);
    let nines = Tensor::<Backend, 1>::from_floats([9.0, 9.0, 9.0], &device);

    // true only where x > 0  ->  [false, false, true]
    let mask = x.clone().greater_elem(0.0);
    x.mask_where(mask, nines)
}

fn main() {
    println!("{}", replace_positives().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replaces_only_positives() {
        let v: Vec<f32> = replace_positives().into_data().to_vec().unwrap();
        assert_eq!(v, vec![-1.0, 0.0, 9.0]);
    }
}
