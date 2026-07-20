// grad3.rs — Burnlings · Chapter 5: Autodiff
//
// There is no `no_grad` block in Burn. For inference you call `.inner()` to peel
// the `Autodiff` wrapper off and drop to the plain base backend — no graph, no
// tracking. Notice the return type is the BASE backend `NdArray`, not `Autodiff`.
//
// The body never drops the wrapper, so it returns an `Autodiff<NdArray>` tensor
// where an `NdArray` tensor is expected — and it won't compile.
//
// TODO: Drop the autodiff wrapper with `.inner()` before the inference op.
//       (Compare book example ch5_03_inner_inference.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type Backend = Autodiff<NdArray>;

fn infer() -> Tensor<NdArray, 1> {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    // ⬇️ still an Autodiff tensor — the return type wants a plain NdArray tensor
    x.add_scalar(5.0)
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
