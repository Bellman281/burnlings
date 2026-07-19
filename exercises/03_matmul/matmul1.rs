// matmul1.rs — Burnlings · Chapter 3: Matmul & the Shape Rule
//
// `matmul` is the real MATRIX product: [m, k] @ [k, n] -> [m, n], and the inner
// dimensions must match. Element-wise `*` (chapter 2) is a completely different
// operation — and for these shapes it can't even apply: `[2, 3]` and `[3, 2]`
// don't line up element-wise, so `*` blows up at runtime.
//
// `product` should compute the matrix product of `a` ([2, 3]) and `b` ([3, 2]),
// which is [[4, 5], [10, 11]]. The body uses `*` instead.
//
// TODO: Use the matrix product instead of element-wise `*`.
//       (Compare book example ch3_01_matmul.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn product() -> Tensor<Backend, 2> {
    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device); // [2, 3]
    let b = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &device); // [3, 2]
    // ⬇️ element-wise product — wrong operation, and these shapes don't align
    a * b
}

fn main() {
    println!("{}", product().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_product() {
        let c = product();
        assert_eq!(c.dims(), [2, 2]);
        let v: Vec<f32> = c.into_data().to_vec().unwrap();
        assert_eq!(v, vec![4.0, 5.0, 10.0, 11.0]);
    }
}
