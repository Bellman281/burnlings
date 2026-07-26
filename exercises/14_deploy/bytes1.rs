// bytes1.rs — Burnlings · Chapter 14: Bare-metal deployment
//
// A microcontroller has no filesystem. So instead of load_file(), you serialise
// the weights to a byte buffer, bake that buffer into the firmware with
// `include_bytes!("model.bin")`, and load it back from RAM — no disk involved.
// BinBytesRecorder does exactly this: record() -> Vec<u8>, load(bytes) -> record.
//
// The bug: the record is loaded from the bytes, but then a FRESH model is used
// and the record is never applied — so the "after" prediction isn't the same
// model as "before".
//
// TODO: Apply the loaded record: `Model::<MyBackend>::new(&device).load_record(record)`.
//       (Compare book example chapter14/e01_weights_baked_into_the_binary.)
//
// I AM NOT DONE

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

// Gap between the prediction before and after the in-RAM byte round trip.
fn round_trip_gap() -> f32 {
    let device = Default::default();
    let x = Tensor::<MyBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let model: Model<MyBackend> = Model::new(&device);
    let before = model.forward(x.clone());

    // Serialise weights to an in-memory byte buffer — NO filesystem.
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::new();
    let bytes: Vec<u8> = recorder.record(model.into_record(), ()).expect("record");
    let record: ModelRecord<MyBackend> = recorder.load(bytes.clone(), &device).expect("load");

    // ⬇️ BUG: fresh model — the loaded record is never applied to it
    let loaded = Model::<MyBackend>::new(&device);
    let _ = record;

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
        assert!(gap < 1e-9, "after != before by {gap} — record not applied?");
    }
}
