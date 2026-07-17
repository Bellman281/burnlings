// ops6.rs — Burnlings · Chapter 2: Tensor Ops
//
// A comparison such as `greater_elem` / `lower_elem` builds a boolean MASK.
// `mask_where(mask, replacement)` keeps the original where the mask is FALSE
// and takes the replacement where the mask is TRUE.
//
// `replace_positives` should replace every element GREATER THAN 0 with 9.0, so
// [-1, 0, 2] becomes [-1, 0, 9]. The mask below is built with the wrong
// comparison, so the wrong element gets replaced.
//
// TODO: Build the mask that is true where x is greater than 0.
//       (Compare book example chapter02/e06_elementwise_masking.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn replace_positives() -> Tensor<Backend, 1> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([-1.0, 0.0, 2.0], &device);
    let nines = Tensor::<Backend, 1>::from_floats([9.0, 9.0, 9.0], &device);

    // ⬇️ wrong comparison: this is true where x < 0
    let mask = x.clone().lower_elem(0.0);
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
