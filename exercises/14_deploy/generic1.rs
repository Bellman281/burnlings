// generic1.rs — Burnlings · Chapter 14: Backend-agnostic inference
//
// Burn's core idea: your model and inference code are generic over the Backend
// trait `B`. The exact same code compiles for CPU (Flex), GPU (Wgpu/CUDA), or
// an embedded backend — you pick the concrete type only at the call site. That's
// what lets you develop on a laptop and deploy to a very different device.
//
// The bug: `run_inference` is hard-wired to `Flex`, so it is NOT generic. The
// call site uses a turbofish (`::<Flex>`) to choose the backend — which only
// compiles if the function actually has a backend type parameter.
//
// TODO: Make `run_inference` generic over the backend:
//       `fn run_inference<B: Backend>(device: &B::Device) -> Tensor<B, 2>`
//       and use `B` throughout (`Classifier::<B>`, `Tensor::<B, 2>`).
//       (Compare book example chapter14/e02_inference_generic_over_the_backend.)
//
// I AM NOT DONE

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    linear: Linear<B>,
}
impl<B: Backend> Classifier<B> {
    fn new(device: &B::Device) -> Self {
        Self { linear: LinearConfig::new(3, 2).init(device) }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

// ⬇️ NOT generic: hard-wired to Flex, so the ::<Flex> call site won't compile
fn run_inference() -> Tensor<Flex, 2> {
    let device = Default::default();
    let model = Classifier::<Flex>::new(&device);
    let x = Tensor::<Flex, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    model.forward(x)
}

fn output_dims() -> Vec<usize> {
    // Pick the concrete backend at the call site — the same function should work
    // for any backend.
    let out = run_inference::<Flex>();
    out.dims().to_vec()
}

fn main() {
    println!("output shape = {:?}", output_dims());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_generic_inference() {
        assert_eq!(output_dims(), vec![1, 2], "3 features -> 2 classes");
    }
}
