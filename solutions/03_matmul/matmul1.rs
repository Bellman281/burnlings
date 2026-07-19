// matmul1.rs — SOLUTION · Chapter 3: Matmul & the Shape Rule
// `matmul` is the real matrix product: [m, k] @ [k, n] -> [m, n]. Element-wise
// `*` is a different operation (and here the shapes don't even line up for it).

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn product() -> Tensor<Backend, 2> {
    let device = Default::default();
    let a = Tensor::<Backend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device); // [2, 3]
    let b = Tensor::<Backend, 2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &device); // [3, 2]
    a.matmul(b) // [2, 3] @ [3, 2] -> [2, 2] = [[4, 5], [10, 11]]
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
