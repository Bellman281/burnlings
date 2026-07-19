// matmul3.rs — Burnlings · Chapter 3: Matmul & the Shape Rule
//
// Batched matmul multiplies matching STACKS of matrices: the leading dim is the
// batch, the last two are the matrices. The rule is
//     [b, m, k] @ [b, k, n]  ->  [b, m, n]
// the batch `b` and the inner `k` must match; the result's last dim is `n`.
//
// `batched` should produce shape [4, 2, 5]: a stack of 4 matrices, each
// [2, 3] @ [3, 5]. The shape of `b` below is wrong, so the result is not
// [4, 2, 5] and the test fails.
//
// TODO: Fix the shape of `b` so the batched product is [4, 2, 5].
//       (Compare book example chapter03/e03_structured_and_batched.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn batched() -> Tensor<Backend, 3> {
    let device = Default::default();
    let a = Tensor::<Backend, 3>::ones([4, 2, 3], &device);
    // ⬇️ wrong last dim: this makes the product [4, 2, 2], not [4, 2, 5]
    let b = Tensor::<Backend, 3>::ones([4, 3, 2], &device);
    a.matmul(b)
}

fn main() {
    println!("dims = {:?}", batched().dims());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_shape_and_values() {
        let out = batched();
        assert_eq!(out.dims(), [4, 2, 5]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| (*x - 3.0).abs() < 1e-6));
    }
}
