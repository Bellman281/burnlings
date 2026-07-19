// tensors2.rs — SOLUTION · Chapter 1: Tensors
// A NESTED array gives you a rank-2 tensor. A flat array is rank 1, and
// asking for shape [2, 3] from 6 flat values is not the same thing.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn matrix() -> Tensor<Backend, 2> {
    let device = Default::default();
    // nested arrays -> two axes, shape [2, 3]
    Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)
}

fn main() {
    let m = matrix();
    println!("m = {m}");
    println!("shape = {:?}", m.dims());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn is_2x3_summing_to_21() {
        let m = matrix();
        assert_eq!(m.dims(), [2, 3]);
        assert_eq!(m.sum().into_scalar(), 21.0);
    }
}
