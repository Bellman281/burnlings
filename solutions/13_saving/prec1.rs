// prec1.rs — Burnlings · Chapter 13: Precision (SOLUTION)
//
// Full-precision (f32) recorder gives an exact round trip.

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

type MyBackend = Flex;

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

fn round_trip_gap() -> f32 {
    let device = Default::default();
    let x = Tensor::<MyBackend, 2>::from_floats([[2.0]], &device);
    let model: Model<MyBackend> = Model::new(&device);
    let original = model.forward(x.clone());

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let path = std::env::temp_dir().join("burnlings_ch13_prec1_full");
    model.clone().save_file(path.clone(), &recorder).expect("save");
    let reloaded: Model<MyBackend> = Model::new(&device)
        .load_file(path, &recorder, &device)
        .expect("load");
    let pred = reloaded.forward(x);

    (pred - original).abs().into_scalar()
}

fn main() {
    println!("|reload - original| = {}", round_trip_gap());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_precision_round_trip_is_exact() {
        let gap = round_trip_gap();
        assert!(gap < 1e-9, "reload differs by {gap}");
    }
}
