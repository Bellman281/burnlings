// ops3.rs — SOLUTION · Chapter 2: Tensor Ops
// `reshape` lays a flat range into a matrix; `slice` takes a rectangular block
// with half-open ranges [start..end) per axis.

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn block() -> Tensor<Backend, 2, Int> {
    let device = Default::default();
    let t = Tensor::<Backend, 1, Int>::arange(0..12, &device);
    let m = t.reshape([3, 4]);
    // rows 0-1, cols 1-2  ->  [[1, 2], [5, 6]]
    m.slice([0..2, 1..3])
}

fn main() {
    println!("{}", block());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slices_the_middle_block() {
        let out = block();
        assert_eq!(out.dims(), [2, 2]);
        let v: Vec<i64> = out.into_data().to_vec().unwrap();
        assert_eq!(v, vec![1, 2, 5, 6]);
    }
}
