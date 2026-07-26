// save1.rs — Burnlings · Chapter 13: Saving & Reloading
//
// Training and inference are usually separate programs: you train once, write
// the weights to disk, then a lightweight process loads them and serves
// predictions. Here we do the whole round trip in one file (fit y = 2x + 1).
//
// The bug is in step 3: the inference model is created FRESH and never actually
// loads the saved weights, so it predicts from random init instead of the
// trained fit — nowhere near 11 for x = 5.
//
// TODO: Load the saved weights into the inference model by chaining
//       `.load_file(path, &recorder, &infer_device).expect("load")`.
//       (Compare book example chapter13/e01_train_save_reload_predict.)
//
// I AM NOT DONE

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

type Train = Autodiff<NdArray>;
type Infer = NdArray;

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

fn train_reload_predict() -> f32 {
    let device = Default::default();
    let path = std::env::temp_dir().join("burnlings_ch13_save1");
    let recorder = CompactRecorder::new();

    // 1. TRAIN (fit y = 2x + 1)
    let x = Tensor::<Train, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
    let y = Tensor::<Train, 2>::from_floats([[3.0], [5.0], [7.0], [9.0]], &device);
    let mut model: Model<Train> = Model::new(&device);
    let mut optim = SgdConfig::new().init();
    for _ in 0..3000 {
        let loss = MseLoss::new().forward(model.forward(x.clone()), y.clone(), Reduction::Mean);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(0.02, model, grads);
    }

    // 2. SAVE weights to disk
    model.save_file(path.clone(), &recorder).expect("save");

    // 3. LOAD into a fresh inference model (plain backend, no autodiff)
    let infer_device = Default::default();
    // ⬇️ BUG: fresh model — the saved weights are never loaded into it
    let loaded: Model<Infer> = Model::new(&infer_device);
    let _ = (&path, &recorder);

    // 4. INFER at x = 5 (should be ~11 = 2*5 + 1)
    let test = Tensor::<Infer, 2>::from_floats([[5.0]], &infer_device);
    loaded.forward(test).into_scalar()
}

fn main() {
    println!("prediction for x=5 after reload: {}", train_reload_predict());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reloaded_model_predicts() {
        let pred = train_reload_predict();
        assert!((pred - 11.0).abs() < 0.1, "x=5 -> {pred}, expected ~11 (weights not reloaded?)");
    }
}
