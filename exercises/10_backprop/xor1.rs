// xor1.rs — Burnlings · Chapter 10: Backprop
//
// XOR is not linearly separable. Stacking two Linear layers WITHOUT a
// nonlinearity between them is still just one linear map, so it can never solve
// XOR — the loss stalls around 0.25. The `tanh` between the layers is what makes
// backprop through a hidden layer meaningful.
//
// TODO: Put a nonlinearity between the layers — `self.l1.forward(x).tanh()`.
//       (Compare book example chapter10/e03_learning_xor.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

type MyBackend = Autodiff<NdArray>;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}
impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self { l1: LinearConfig::new(2, 8).init(device), l2: LinearConfig::new(8, 1).init(device) }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // ⬇️ no nonlinearity: two Linear layers collapse into one linear map
        let h = self.l1.forward(x);
        self.l2.forward(h)
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
}
