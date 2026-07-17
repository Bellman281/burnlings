// tensors4.rs — SOLUTION · Chapter 1: Tensors
// Burn ships constructors for common fills: zeros, ones, full, eye, arange.
// `full` fills a shape with one value.

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn sevens() -> Tensor<Backend, 2> {
    let device = Default::default();
    // a 2x3 tensor where every element is 7.0
    Tensor::<Backend, 2>::full([2, 3], 7.0, &device)
}

fn main() {
    let t = sevens();
    println!("sevens = {t}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn six_sevens_sum_to_42() {
        let t = sevens();
        assert_eq!(t.dims(), [2, 3]);
        assert_eq!(t.sum().into_scalar(), 42.0);
    }
}
