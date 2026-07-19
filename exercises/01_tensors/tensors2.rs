// tensors2.rs — Burnlings · Chapter 1: Tensors
//
// There are two common ways to build a tensor: `from_floats` (ergonomic for
// float literals) and `from_data` (the general path). The SHAPE comes from how
// the data is nested: a flat array is rank 1; a nested array is rank 2.
//
// `matrix` must return a 2x3 tensor holding [[1,2,3],[4,5,6]] — two rows of
// three. The code below hands `from_floats` a FLAT array of 6 numbers, which is
// rank 1, so it cannot become a [2, 3] matrix.
//
// TODO: Give `from_floats` the data nested as two rows of three so the tensor
//       is genuinely 2x3. (Compare book example ch1_02_basic_creation.rs.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn matrix() -> Tensor<Backend, 2> {
    let device = Default::default();
    // ⬇️ flat array = rank 1; we need two rows of three
    Tensor::<Backend, 2>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
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
