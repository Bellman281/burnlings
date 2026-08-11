// xor2.rs — Burnlings · Chapter 10: Backprop
//
// XOR is not linearly separable. Stacking two Linear layers WITHOUT a
// nonlinearity between them is still just one linear map, so it can never solve
// XOR — the loss stalls around 0.25. The `tanh` between the layers is what makes
// backprop through a hidden layer meaningful.
//
// TODO: Complete the network, implement more than 2 layers!!
//
// I AM NOT DONE

use burn::backend::{Autodiff, Flex};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
// use necessary layers



type MyBackend = Autodiff<Flex>;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    // You can optionally experiment here.
}
impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self { 
            // You can optionally experiment here.
         }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // You can optionally experiment here.
    }
}

fn final_loss() -> f32 {
    let device = Default::default();
    MyBackend::seed(&device, 1);
    let x = Tensor::<MyBackend, 2>::from_floats([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], &device);
    let y = Tensor::<MyBackend, 2>::from_floats([[0.0], [1.0], [1.0], [0.0]], &device);
    let mut model = Mlp::new(&device);
    let mut optim = SgdConfig::new().init();
    let lr = 0.1;
    for _ in 0..20000 {
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }
    let pred = model.forward(x.clone());
    (pred - y).powf_scalar(2.0).mean().into_scalar()
}

fn main() {
    println!("final XOR loss = {}", final_loss());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learns_xor() {
        let l = final_loss();
        assert!(l < 0.05, "final loss = {l} (did the net solve XOR?)");
    }

    // Counts submodules by type as burn walks the module tree.
    struct LayerCounter {
        linear: usize,
    }
    impl<B: Backend> burn::module::ModuleVisitor<B> for LayerCounter {
        fn enter_module(&mut self, name: &str, container_type: &str) {
            if container_type == "Struct:Linear" && name == "weight" {
                self.linear += 1;
            }
        }
    }

    #[test]
    fn model_has_more_than_two_layers() {
        let model = Mlp::<MyBackend>::new(&Default::default());
        let mut counter = LayerCounter { linear: 0 };
        model.visit(&mut counter);
        println!("later counts = {}", counter.linear);
        assert!(
            counter.linear > 2,
            "found {} Linear layers — the net needs more depth",
            counter.linear
        );
    }
}
