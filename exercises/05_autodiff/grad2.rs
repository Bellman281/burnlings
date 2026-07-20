// grad2.rs — Burnlings · Chapter 5: Autodiff
//
// Burn's `backward()` returns ALL gradients in one container, and you look each
// one up by the tensor it belongs to — `a.grad(&g)` gives dL/da, `b.grad(&g)`
// gives dL/db. For L = sum(a * b): dL/da = b and dL/db = a.
//
// The two lookups below are swapped, so `da` and `db` come out backwards and the
// test fails.
//
// TODO: Fix the lookups so `da` uses `a` and `db` uses `b`.
//       (Compare book example ch5_02_gradient_container.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn grads() -> (Vec<f32>, Vec<f32>) {
    let device = Default::default();
    let a = Tensor::<Backend, 1>::from_floats([2.0, 3.0], &device).require_grad();
    let b = Tensor::<Backend, 1>::from_floats([4.0, 5.0], &device).require_grad();

    let loss = (a.clone() * b.clone()).sum();
    let g = loss.backward();

    // ⬇️ swapped: da should look up `a`, db should look up `b`
    let da = b.grad(&g).unwrap();
    let db = a.grad(&g).unwrap();
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
