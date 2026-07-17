// tensors1.rs — Burnlings · Chapter 1: Tensors
//
// In Burn, a tensor's RANK (its number of dimensions) is part of its TYPE.
// `Tensor<B, 1>` and `Tensor<B, 2>` are DIFFERENT types, and the compiler
// will not let you mix them. This is the "3 a.m. shape error becomes a
// compile error" promise from the Learning Burn book, chapter 1.
//
// The five numbers below sit in a single row: one axis, length 5.
// That is RANK 1 — shape `[5]` — not rank 2.
//
// TODO: Fix the rank in the type annotation so the code compiles.
//       Change ONLY the rank number. Do not touch the data.
//
// Run it:    cargo run  --example tensors1
// Check it:  cargo test --example tensors1
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn one_row() -> Tensor<Backend, 1> {
    let device = Default::default();
    let floats = [1.0, 2.0, 3.0, 4.0, 5.0];

    // ⬇️ this annotation claims rank 2, but the data is a single row (rank 1)
    let tensor = Tensor::<Backend, 2>::from_floats(floats, &device);

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
        // one axis, of length 5
        assert_eq!(t.dims(), [5]);
    }
}
