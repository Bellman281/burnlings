// generic1.rs — Burnlings · Chapter 14: Backend-agnostic inference (SOLUTION)
//
// Inference written once, generic over B; the backend is chosen at the call site.

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

// Generic over B — nothing here names a concrete backend.
fn run_inference<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    let model = Classifier::<B>::new(device);
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], device);
    model.forward(x)
}

fn output_dims() -> Vec<usize> {
    let out = run_inference::<Flex>(&Default::default());
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
