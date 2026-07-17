// ops1.rs — Burnlings · Chapter 2: Tensor Ops
//
// On two tensors, `*` is ELEMENT-WISE multiplication (pair up matching
// positions), NOT matrix multiplication. `+` adds element-wise too.
//
// `products` should multiply `a` and `b` element-wise to get [10, 40, 90].
// The body adds them instead, so the test fails.
//
// TODO: Multiply the two tensors element-wise instead of adding them.
//       (Compare book example chapter02/e01_arithmetic.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn products() -> Tensor<Backend, 1> {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device);
    // ⬇️ this ADDS; we want the element-wise product
    a + b
}

fn main() {
    println!("products = {}", products().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn elementwise_product() {
        let v: Vec<f32> = products().into_data().to_vec().unwrap();
        assert_eq!(v, vec![10.0, 40.0, 90.0]);
    }
}
