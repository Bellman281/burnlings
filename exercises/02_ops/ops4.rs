// ops4.rs — Burnlings · Chapter 2: Tensor Ops
//
// A dimension reduction collapses ONE axis. `sum_dim(0)` sums DOWN each column
// (collapsing the rows); `sum_dim(1)` sums ACROSS each row. Burn keeps the
// reduced axis, so summing a [2, 2] on dim 0 gives shape [1, 2].
//
// `col_sums` should sum down the columns of [[1,2],[3,4]] to get [[4, 6]].
// The body reduces the wrong axis, so both the shape and values are wrong.
//
// TODO: Reduce the axis that leaves you with per-column sums.
//       (Compare book example chapter02/e04_reductions.)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn col_sums() -> Tensor<Backend, 2> {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    // ⬇️ wrong axis: this sums across each row, not down each column
    x.sum_dim(1)
}

fn main() {
    println!("{}", col_sums().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sums_down_columns() {
        let out = col_sums();
        assert_eq!(out.dims(), [1, 2]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert_eq!(v, vec![4.0, 6.0]);
    }
}
