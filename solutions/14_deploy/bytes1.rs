// bytes1.rs — Burnlings · Chapter 14: Bare-metal deployment (SOLUTION)
//
// Load the recorded bytes back into the model with load_record.

use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

type MyBackend = NdArray;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    linear: Linear<B>,
}
impl<B: Backend> Model<B> {
    fn new(device: &B::Device) -> Self {
        Self { linear: LinearConfig::new(4, 1).init(device) }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

fn round_trip_gap() -> f32 {
    let device = Default::default();
    let x = Tensor::<MyBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let model: Model<MyBackend> = Model::new(&device);
    let before = model.forward(x.clone());

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes: Vec<u8> = recorder.record(model.into_record(), ()).expect("record");
    let record = recorder.load(bytes.clone(), &device).expect("load");

    let loaded = Model::<MyBackend>::new(&device).load_record(record);

    let after = loaded.forward(x);
    (after - before).abs().into_scalar()
}

fn main() {
    println!("|after - before| = {}", round_trip_gap());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_round_trip_is_exact() {
        let gap = round_trip_gap();
        assert!(gap < 1e-9, "after != before by {gap}");
    }
}
