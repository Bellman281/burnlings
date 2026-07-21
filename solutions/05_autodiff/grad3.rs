// grad3.rs — SOLUTION · Chapter 5: Autodiff
// There is no `no_grad` block in Burn. For inference, call `.inner()` to peel
// the autodiff wrapper off and drop to the plain base backend (no graph, no
// tracking). Note the RETURN type is the base backend `NdArray`, not `Autodiff`.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn infer() -> Tensor<NdArray, 1> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    // drop autodiff, then do the cheap inference op
    x.inner().add_scalar(5.0)
}

fn main() {
    println!("{}", infer().to_data());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plain_inference() {
        let v: Vec<f32> = infer().into_data().to_vec().unwrap();
        assert_eq!(v, vec![6.0, 7.0, 8.0]);
    }
}
