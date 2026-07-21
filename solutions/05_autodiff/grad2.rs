// grad2.rs — SOLUTION · Chapter 5: Autodiff
// Burn's `backward()` returns ALL gradients in one container; you look each up
// by the tensor it belongs to. For L = sum(a * b): dL/da = b, dL/db = a.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn grads() -> (Vec<f32>, Vec<f32>) {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([2.0, 3.0], &device).require_grad();
    let b = Tensor::<Backend, 1>::from_floats([4.0, 5.0], &device).require_grad();

    let loss = (a.clone() * b.clone()).sum();
    let g = loss.backward();

    let da = a.grad(&g).unwrap(); // dL/da = b = [4, 5]
    let db = b.grad(&g).unwrap(); // dL/db = a = [2, 3]
    (
        da.into_data().to_vec().unwrap(),
        db.into_data().to_vec().unwrap(),
    )
}

fn main() {
    let (da, db) = grads();
    println!("dL/da = {da:?}, dL/db = {db:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn product_rule() {
        let (da, db) = grads();
        assert_eq!(da, vec![4.0, 5.0]); // = b
        assert_eq!(db, vec![2.0, 3.0]); // = a
    }
}
