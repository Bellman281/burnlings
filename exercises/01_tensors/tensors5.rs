// tensors5.rs — Burnlings · Chapter 1: Tensors
//
// `random` takes a `Distribution`. `Distribution::Default` is UNIFORM on
// [0, 1). `Distribution::Normal(mean, std)` is a bell curve whose samples are
// NOT bounded — plenty will fall outside [0, 1).
//
// `uniform_sample` must return samples that all lie in [0, 1). The code below
// uses a normal distribution, so the test (which checks the bounds) fails.
//
// TODO: Choose the distribution that is uniform on [0, 1).
//
// I AM NOT DONE

use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};

type Backend = NdArray;

fn uniform_sample() -> Tensor<Backend, 1> {
    let device = Default::default();
    // ⬇️ normal samples escape [0, 1)
    Tensor::<Backend, 1>::random([64], Distribution::Normal(0.0, 1.0), &device)
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
