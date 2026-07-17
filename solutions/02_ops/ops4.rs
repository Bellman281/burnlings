// ops4.rs — SOLUTION · Chapter 2: Tensor Ops
// Dimension reductions take the axis to collapse. `sum_dim(0)` sums DOWN each
// column. Burn keeps the reduced axis, so a [2, 2] summed on dim 0 is [1, 2].

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn col_sums() -> Tensor<Backend, 2> {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    x.sum_dim(0)
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
