// tensors1.rs — SOLUTION · Burnlings · Chapter 1: Tensors
//
// The five numbers are one axis of length 5, so the tensor is RANK 1.
// The only change needed was the rank in the type annotation: 2 -> 1.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn one_row() -> Tensor<Backend, 1> {
    let device = Default::default();
    let floats = [1.0, 2.0, 3.0, 4.0, 5.0];

    // Rank matches the data now: one axis, shape [5].
    let tensor = Tensor::<Backend, 1>::from_floats(floats, &device);

    tensor
}

fn main() {
    let t = one_row();
    println!("tensor = {t}");
    println!("rank   = {}", t.dims().len());
    println!("shape  = {:?}", t.dims());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_rank_1_with_five_elements() {
        let t = one_row();
        assert_eq!(t.dims(), [5]);
    }
}
