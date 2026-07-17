// matmul1.rs — Burnlings · Chapter 3: Matmul & the Shape Rule
//
// `matmul` is the real MATRIX product: [m, k] @ [k, n] -> [m, n], and the inner
// dimensions must match. Element-wise `*` (chapter 2) is a completely different
// operation — it just multiplies matching positions.
//
// `product` should compute the matrix product of `a` and `b`, which is
// [[3, 2], [7, 4]]. The body uses `*`, giving the element-wise product instead,
// so the test fails.
//
// TODO: Use the matrix product instead of element-wise `*`.
//       (Compare book example chapter03/e01_matmul.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn product() -> Tensor<Backend, 2> {
    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let b = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [1.0, 1.0]], &device);
    // ⬇️ element-wise product, not the matrix product
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
        assert_eq!(v, vec![3.0, 2.0, 7.0, 4.0]);
    }
}
