// tensors4.rs — Burnlings · Chapter 1: Tensors
//
// Burn ships constructors for common fills: `zeros`, `ones`, `full`, `eye`,
// `arange`. `sevens` should be a 2x3 tensor where EVERY element is 7.0.
//
// TODO: Pick the right constructor so every element is 7.0.
//       (Hint: which one fills a shape with a chosen value?)
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::Tensor;

type Backend = NdArray;

fn sevens() -> Tensor<Backend, 2> {
    let device = Default::default();
    // ⬇️ zeros gives all 0.0 — we want all 7.0
    Tensor::<Backend, 2>::zeros([2, 3], &device)
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
