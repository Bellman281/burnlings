// ops1.rs — SOLUTION · Chapter 2: Tensor Ops
// `*` on two tensors is ELEMENT-WISE, not matrix multiplication.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn products() -> Tensor<Backend, 1> {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let b = Tensor::<Backend, 1>::from_floats([10.0, 20.0, 30.0], &device);
    a * b
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
