// sgd1.rs — Burnlings · Chapter 9: Training
//
// A real training loop, four beats: forward -> loss -> backward -> step.
// Burn's optimizer is FUNCTIONAL: `optim.step(...)` CONSUMES the old model and
// RETURNS a new, updated one. If you don't capture that return value, the model
// is moved away and gone — the next iteration can't use it, and it never learns.
//
// Trained on y = 2x + 1, a correct loop makes the model predict f(5) ~ 11.
//
// TODO: Capture the updated model: `model = optim.step(lr, model, grads);`.
//       (Compare book example chapter09/e01 — a real SGD training loop.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

type MyBackend = Autodiff<NdArray>;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn new(device: &B::Device) -> Self {
        Self { linear: LinearConfig::new(1, 1).init(device) }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

fn train_and_predict() -> f32 {
    let device = Default::default();
    let x = Tensor::<MyBackend, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
    let y = Tensor::<MyBackend, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device);

    let mut model = Model::new(&device);
    let mut optim = SgdConfig::new().init();
    let lr = 0.02;

    for _ in 0..3000 {
        let pred = model.forward(x.clone());
        let loss = MseLoss::new().forward(pred, y.clone(), Reduction::Mean);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        // ⬇️ the updated model is thrown away (and `model` is moved out)
        optim.step(lr, model, grads);
    }

    let test = Tensor::<MyBackend, 2>::from_floats([[5.0]], &device);
    model.forward(test).into_scalar()
}

fn main() {
    println!("f(5) = {}", train_and_predict());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learns_the_line() {
        let p = train_and_predict();
        assert!((p - 11.0).abs() < 0.3, "f(5) = {p}");
    }
}
