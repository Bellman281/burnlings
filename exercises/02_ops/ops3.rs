// ops3.rs — Burnlings · Chapter 2: Tensor Ops
//
// `reshape` lays a flat range into a matrix. `slice` takes a rectangular block
// using a half-open range [start..end) for each axis (end is EXCLUSIVE).
//
// `block` should return rows 0-1 and columns 1-2 of the 3x4 matrix, i.e.
// [[1, 2], [5, 6]]. The ranges below grab the wrong block, so the test fails.
//
// TODO: Fix the two ranges so you select rows 0..2 and columns 1..3.
//       (Compare book example chapter02/e03_reshaping_slicing.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type Backend = NdArray;

fn block() -> Tensor<Backend, 2, Int> {
    let device = Default::default();
    let t = Tensor::<Backend, 1, Int>::arange(0..12, &device);
    let m = t.reshape([3, 4]);
    // ⬇️ wrong block: these ranges are not rows 0-1, cols 1-2
    m.slice([0..3, 0..2])
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
