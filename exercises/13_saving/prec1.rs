// prec1.rs — Burnlings · Chapter 13: Precision
//
// A "recorder" decides how weights are serialised. Two msgpack recorders differ
// only in numeric precision:
//   * NamedMpkFileRecorder<FullPrecisionSettings> -> f32: weights reload exactly
//   * CompactRecorder (f16) -> weights are rounded to half precision, so the
//     reloaded prediction shifts slightly
//
// This checks an EXACT round trip (reload must equal the original bit-for-bit).
// The bug saves and loads with the compact (f16) recorder, so a little precision
// is lost and the exact check fails.
//
// TODO: Use a full-precision recorder for an exact round trip:
//       `NamedMpkFileRecorder::<FullPrecisionSettings>::new()`
//       (add the import `use burn::record::{FullPrecisionSettings,
//        NamedMpkFileRecorder};`).
//       (Compare book example chapter13/e02_same_weights_two_precisions.)
//
// I AM NOT DONE

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::record::CompactRecorder;
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

// Gap between the original prediction and the reloaded one.
fn round_trip_gap() -> f32 {
    let device = Default::default();
    let x = Tensor::<MyBackend, 2>::from_floats([[2.0]], &device);
    let model: Model<MyBackend> = Model::new(&device);
    let original = model.forward(x.clone());

    // ⬇️ compact (f16) recorder rounds the weights — not an exact round trip
    let recorder = CompactRecorder::new();
    let path = std::env::temp_dir().join("burnlings_ch13_prec1");
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
        assert!(gap < 1e-9, "reload differs by {gap} — not a full-precision round trip");
    }
}
