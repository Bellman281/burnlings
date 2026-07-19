// ops2.rs — Burnlings · Chapter 2: Tensor Ops
//
// PyTorch silently broadcasts a length-3 vector across a [2, 3] matrix. Burn
// does NOT: a `Tensor<B, 2>` and a `Tensor<B, 1>` are different types and won't
// add. You must first raise the vector to rank 2 with `unsqueeze()` (giving
// shape [1, 3]); then it broadcasts over the 2 rows.
//
// As written, `m + bias` mixes rank 2 and rank 1, so it does not compile.
//
// TODO: Raise `bias` to rank 2 so it can broadcast across the rows of `m`.
//       (Compare book example chapter02/e02_broadcasting.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn add_bias() -> Tensor<Backend, 2> {
    let device = Default::default();
    let m = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let bias = Tensor::<Backend, 1>::from_floats([100.0, 200.0, 300.0], &device);
    // ⬇️ rank 2 + rank 1 won't compile
    m + bias
}

fn main() {
    println!("{}", add_bias().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn per_column_bias() {
        let out = add_bias();
        assert_eq!(out.dims(), [2, 3]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert_eq!(v, vec![101.0, 202.0, 303.0, 104.0, 205.0, 306.0]);
    }
}
