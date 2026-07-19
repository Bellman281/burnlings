// tensors5.rs — SOLUTION · Chapter 1: Tensors
// `Distribution::Default` is uniform on [0, 1). `Distribution::Normal(m, s)`
// is a bell curve whose values are NOT bounded to [0, 1).

use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};

type Backend = NdArray;

fn uniform_sample() -> Tensor<Backend, 1> {
    let device = Default::default();
    // 64 samples uniform on [0, 1)
    Tensor::<Backend, 1>::random([64], Distribution::Default, &device)
}

fn main() {
    let t = uniform_sample();
    println!("shape = {:?}", t.dims());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_values_in_unit_interval() {
        let t = uniform_sample();
        assert_eq!(t.dims(), [64]);
        let d = t.into_data();
        let v: &[f32] = d.as_slice().unwrap();
        assert!(v.iter().all(|x| *x >= 0.0 && *x < 1.0));
    }
}
