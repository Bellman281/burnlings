// count1.rs — Burnlings · Chapter 14: model footprint
//
// On a microcontroller, memory is THE constraint. Before deploying you want to
// know: how many parameters, and how many bytes at each precision? `Module` gives
// you `num_params()`; the rest is arithmetic — one value costs 4 bytes at f32,
// 2 bytes at f16, and 1 byte at int8.
//
// The bug computes the f16 footprint with the f32 factor (x4), so it reports no
// saving over f32 at all.
//
// TODO: An f16 value is 2 bytes, so f16 bytes = `params * 2`.
//       (Compare book example chapter14/e04_counting_parameters_and_bytes.)
//
// I AM NOT DONE

use burn::backend::Flex;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;

type MyBackend = Flex;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>, // 64 -> 32
    l2: Linear<B>, // 32 -> 10
}
impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(64, 32).init(device),
            l2: LinearConfig::new(32, 10).init(device),
        }
    }
}

// Returns (params, f32 bytes, f16 bytes, int8 bytes).
fn footprint() -> (usize, usize, usize, usize) {
    let device = Default::default();
    let model: Mlp<MyBackend> = Mlp::new(&device);

    let params = model.num_params();
    let f32_bytes = params * 4;
    // ⬇️ wrong factor: an f16 value is 2 bytes, not 4
    let f16_bytes = params * 4;
    let int8_bytes = params;
    (params, f32_bytes, f16_bytes, int8_bytes)
}

fn main() {
    let (params, f32b, f16b, int8b) = footprint();
    println!("parameters   = {params}");
    println!("f32  weights = {f32b} bytes");
    println!("f16  weights = {f16b} bytes");
    println!("int8 weights = {int8b} bytes");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn footprint_at_each_precision() {
        let (params, f32b, f16b, int8b) = footprint();
        assert_eq!(params, 2410, "64*32+32 + 32*10+10 = 2410");
        assert_eq!(f32b, 9640);
        assert_eq!(f16b, 4820, "f16 is 2 bytes per value");
        assert_eq!(int8b, 2410);
    }
}
