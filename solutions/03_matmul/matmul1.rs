// matmul1.rs — SOLUTION · Chapter 3: Matmul & the Shape Rule
// `matmul` is the real matrix product: [m, k] @ [k, n] -> [m, n]. It is NOT the
// same as element-wise `*`.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn product() -> Tensor<Backend, 2> {
    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let b = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [1.0, 1.0]], &device);
    a.matmul(b)
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
